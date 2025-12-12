import atexit
import os
from collections import namedtuple
from pathlib import Path
from typing import Dict, Optional, OrderedDict

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from core import get_dataloaders, get_model
from dataset_split import DatasetSplit
from dino import encode_reference_frames
from einops import rearrange, repeat
from features import *
from omegaconf import DictConfig, OmegaConf
from seed import seed_everything
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchcfm import TargetConditionalFlowMatcher
from torchdiffeq import odeint
from torchinfo import summary
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from vae import decode_tracks_latte, encode_tracks_latte
from visualizations import plot_trajectories_on_reference_frame_with_titles

from animations import (animate_motion, concatenate_animations,
                        save_frames_as_gif)
from nn_latte import Latte, log_normal_timestep
from nn_latte_vae import LatteVAE
from sample import pixel_center_grid
from schedulers import (CosineAnnealingLRWLinearWarmupScheduler,
                        LinearWarmupScheduler)


def get_lr_scheduler(optimizer: AdamW, cfg: DictConfig) -> (
    LinearWarmupScheduler |
    CosineAnnealingLRWLinearWarmupScheduler
):
    if "lr_scheduler" in cfg.training:
        if "cosine" in cfg.training.lr_scheduler:
            return CosineAnnealingLRWLinearWarmupScheduler(
                optimizer,
                warmup_lr=cfg.training.warmup_lr,
                warmup_steps=cfg.training.warmup_steps,
                cosine_annealing_steps=cfg.training.annealing_steps
            )
        else:
            return LinearWarmupScheduler(
                optimizer,
                cfg.training.warmup_steps,
            )


def load(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: AdamW,
    lr_scheduler: LinearWarmupScheduler,
    should_restore_optimizer: bool = True
):
    checkpoint = torch.load(
        checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if should_restore_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded optimizer state.")
    else:
        print("Skipping loading optimizer state.")
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    return (
        model,
        optimizer,
        lr_scheduler,
        checkpoint["step"],
        checkpoint["epoch"],
        checkpoint["wandb"]
    )


def save(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: AdamW,
    lr_scheduler: LinearWarmupScheduler,
    step: int,
    epoch: int,
    cfg: DictConfig,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            **(
                {
                    "wandb": {
                        "run": {
                            "id": wandb.run.id,
                            "name": wandb.run.name
                        }
                    }
                } if cfg.training.use_wandb
                else {}
            )
        },
        str(checkpoint_path)
    )


DistConfig = namedtuple(
    "DistConfig", ["world_size", "global_rank", "local_rank", "device"])


@torch.inference_mode()
def sample(
    v: nn.Module,
    cfg: OrderedDict[str, torch.Tensor],
    query_points: torch.Tensor,
    frames: Optional[torch.Tensor],
    trajectory_length: int,
    device: torch.device,
    rng: torch.Generator,
    return_sampling_trajectory: bool = False,
    features: Optional[Features] = None,
):
    (b, _, n, _) = query_points.shape
    # Forward pass with autocast
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        print("\nsampling...")
        # Set up the initial state
        noise = torch.randn(
            (b, trajectory_length, n, trajectory_length, 2),
            device=device,
            generator=rng
        )
        x0 = noise
        x0[:, :, :, 0, :] = query_points
        # Mask to select non-initial part of the trajectory.
        m = torch.ones(
            (b, trajectory_length, n, trajectory_length, 2),
            dtype=x0.dtype,
            device=x0.device
        )
        m[:, :, :, 0, :] = 0.0
        # Solve ODE given by v (Integrate f)
        # Define the integrand

        def f(t: float, x: torch.Tensor) -> torch.Tensor:
            # Set up the initial position
            x[:, :, :, 0, :] = query_points
            # Set up the time
            t = t * torch.ones(
                size=(b, ),
                device=x.device
            )
            return m * v(
                trajectory=x,
                frames=frames,
                t=t,
                features=features
            )
        # Integrate
        xt = odeint(
            f,
            x0,
            torch.linspace(0, 1, cfg.sampling.steps, device=device),
            atol=cfg.sampling.atol,
            rtol=cfg.sampling.rtol,
            method=cfg.sampling.method,
        )
    if return_sampling_trajectory:
        return xt
    else:
        return xt[-1, ...]


@torch.inference_mode()
def sample_latte_latte_vae(
    model: Latte,
    vae: LatteVAE,
    vae_scale: torch.Tensor,
    dino: AutoModel,
    processor: AutoImageProcessor,
    reference_frames: torch.Tensor,
    device: torch.device,
    trajectory_stride: int,
    atol=1e-5,
    rtol=1e-5,
    method="euler",
    steps=10,
    rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    print(
        f"sampling method={method}, steps={steps}, atol={atol}, rtol={rtol} ...")
    model.eval()
    vae.eval()
    dino.eval()
    model = model.to(device)
    vae = vae.to(device)
    dino = dino.to(device)
    # query_points: # [b, h, w, c]
    batch_size, h, w, _ = reference_frames.shape
    query_points = repeat(
        pixel_center_grid(h, w, device)[
            ::trajectory_stride, ::trajectory_stride, :],
        "h w d -> b h w d",
        b=batch_size
    ).to(device)
    query_points = repeat(
        query_points,
        "b h w d -> b t h w d",
        t=24
    )
    cond_latent = encode_tracks_latte(
        vae,
        vae_scale,
        dino,
        processor,
        query_points,
        reference_frames.unsqueeze(1),
        device
    )[:, [0], ...]  # [b, 1, c, h, w]
    # Encode reference frames
    features = encode_reference_frames(
        reference_frames,
        processor,
        dino,
        device
    )  # [b, n, d]
    # Encode frames
    _, _, latent_channels, latent_h, latent_w = cond_latent.shape
    # Generate noise
    noise = torch.randn(
        (batch_size, 23, latent_channels, latent_h, latent_w),
        device=device,
        generator=rng
    )
    x0 = torch.cat([cond_latent, noise], dim=1)  # [b, t, c, h, w]
    # Mask to select non-initial part of the trajectory.
    m = torch.ones_like(x0)
    m[:, 0, ...] = 0.0

    def f(t: float, x: torch.Tensor) -> torch.Tensor:
        # Set up the conditioning
        x[:, 0, ...] = cond_latent[:, 0, ...]
        # Set up the time
        t = t * torch.ones(
            size=(batch_size, ),
            device=x.device
        )
        return m * model(x, t, features)
    # Integrate the field
    xt = odeint(
        f,
        x0,
        torch.linspace(0, 1, steps, device=device),
        atol=atol,
        rtol=rtol,
        method=method,
    )
    # Pick the final solution
    x = xt[-1, ...]
    # Initial state is clean
    x[:, 0, ...] = cond_latent[:, 0, ...]
    return x


def visualize_samples_with_segmentation_info(
    frames: torch.Tensor,
    gt_traj: torch.Tensor,
    pred_traj: torch.Tensor,
    segmentations: torch.Tensor,
    samples_path: Path,
    step: int,
):
    _, traj_len, *_ = frames.shape
    motion_animation_per_example = []
    output = {}
    for (fr, gt, pred, seg) in zip(frames, gt_traj, pred_traj, segmentations):
        # frames: [t, h, w, c]
        # gt: [n, t, 2]
        # pred: [n, t, 2]
        # seg: [t, h, w]
        motion_animation_per_example.append(
            animate_motion(
                ground_truth_trajectories=gt[0],
                predicted_trajectories=torch.clip(pred, 0, 255),
                frames=rearrange(fr, "t h w c -> t c h w"),
                segmentations=seg,
            )
        )

    motion_animation = concatenate_animations(
        motion_animation_per_example,
        np.vstack
    )
    save_frames_as_gif(
        motion_animation,
        samples_path / f"step={step}" / f"motion_length={traj_len}.gif"
    )
    output[traj_len] = {
        "motion": samples_path / f"step={step}" / f"motion_length={traj_len}.gif",
    }

    return output


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def generate_samples(
    v: Latte,
    vae: LatteVAE,
    vae_scale: torch.Tensor,
    processor: AutoImageProcessor,
    dino: AutoModel,
    batch,
    cfg,
    step,
    sampling_rng,
    samples_path,
    device
) -> Dict[int, Path]:
    (samples_path / f"step={step}").mkdir(exist_ok=True, parents=True)
    # frames:  [b, t, h, w, c]
    # tracks: [b, t, h, w, 2]
    # segs: [b, t, h, w]
    # visibility: [b, t, h, w]
    batch_size, traj_length, *_ = batch.trajectories.shape
    batch_indices = torch.randperm(
        batch_size)[:cfg.sampling.num_samples]  # [b, ]
    # [b, t, h, w, 2]
    selected_trajectories = batch.trajectories[batch_indices]
    # Subsample
    stride = cfg.model.nn.trajectory_stride
    # [b, t, h, w, 2]
    selected_trajectories = selected_trajectories[:, :, ::stride, ::stride, :]
    selected_frames = batch.frames[batch_indices]  # [b, t, h, w, c]
    selected_reference_frames = selected_frames[:, 0, ...]  # [b, h, w, c]
    selected_segmentations = batch.segmentations[batch_indices]
    print("sampling...")
    pred_latents = sample_latte_latte_vae(
        v,
        vae,
        vae_scale,
        dino,
        processor,
        selected_reference_frames,
        device,
        trajectory_stride=8,  # cfg.model.nn.trajectory_stride
        method=cfg.sampling.method,
        steps=cfg.sampling.steps,
        atol=cfg.sampling.atol,
        rtol=cfg.sampling.rtol
    )  # [b, t, c, h, w]
    pred_tracks = decode_tracks_latte(
        vae,
        vae_scale,
        dino,
        processor,
        pred_latents,
        selected_reference_frames.unsqueeze(1),
        device
    ).cpu()  # [b, h, w, t, 2]
    selected_segmentations = batch.segmentations[batch_indices]
    gt_and_samples = torch.cat([
        rearrange(
            selected_trajectories.cpu(),
            "b t h w d -> b () (h w) t d"
        ),
        rearrange(
            pred_tracks.cpu(),
            "b t h w d -> b () (h w) t d"
        ),
    ], dim=1)
    trajectories = plot_trajectories_on_reference_frame_with_titles(
        trajectories=gt_and_samples,
        frames=rearrange(
            selected_reference_frames,
            "b h w c -> b () h w c"
        ),
        preprocess_cfg=None,
        denormalize=False,
        titles=["gt", "sample"]
    )
    gt, samples = gt_and_samples.chunk(2, dim=1)

    motion = visualize_samples_with_segmentation_info(
        frames=selected_frames,
        gt_traj=gt,
        pred_traj=samples,
        segmentations=selected_segmentations,
        samples_path=samples_path,
        step=step,
    )

    return {
        traj_length: {
            "reference": trajectories,
            "motion": motion[traj_length]["motion"]
        }
    }


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# def setup_distributed():
#   rank = int(os.environ['SLURM_PROCID'])  # Global rank
#   node_rank = int(os.environ["SLURM_NODEID"])  # Node ID
#   world_size = int(os.environ["WORLD_SIZE"])  # Total number of ranks
#   local_rank = rank % torch.cuda.device_count()
#   os.environ["RANK"] = str(rank)  # Required for NCCL.
#   # Force GPU selection to avoid duplicates
#   torch.cuda.set_device(local_rank)
#   print(
#       f"Rank {rank}/{world_size}, Node {node_rank}, Local Rank {local_rank}, GPU: {torch.cuda.current_device()}"
#   )
#   # Initialize distributed training
#   dist.init_process_group(backend="nccl")
#   atexit.register(cleanup_distributed)
#   return DistConfig(
#       world_size=world_size,
#       global_rank=rank,
#       local_rank=local_rank,
#       device=torch.device(f"cuda:{local_rank}")
#   )

def setup_distributed():
    # Works for torchrun
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    print(f"global rank {rank}/{world_size} initialized.")
    torch.cuda.set_device(device)
    atexit.register(cleanup_distributed)
    return DistConfig(
        world_size=world_size,
        global_rank=rank,
        local_rank=local_rank,
        device=device
    )


config_path = os.environ.get(
    "CONFIG_PATH", "/users/gabrijel/projects/track-generator/configs/training/kubric/movi_a/latte/")
config_name = os.environ.get(
    "CONFIG_NAME", "small_latte.yaml"
)


def train_step(
    model: Latte,
    vae: LatteVAE,
    vae_scale: torch.Tensor,
    fm: TargetConditionalFlowMatcher,
    dino: AutoModel,
    processor: AutoImageProcessor,
    batch,
    device: torch.device,
    trajectory_stride: int,
    use_amp: bool
) -> torch.Tensor:
    frames = batch.frames                   # [b, t, h, w, c]
    tracks = batch.trajectories[:, :, ::trajectory_stride,
                                ::trajectory_stride, :]   # [b, t, h, w, 2]
    reference_frames = frames[:, 0, ...]    # [b, h, w, c]
    batch_size, num_frames, *_ = tracks.shape
    # Encode reference frames
    features = encode_reference_frames(
        reference_frames,
        processor,
        dino,
        device
    )  # [b, n, d]
    # Encode query points
    query_points = tracks[:, 0, ...]  # [b, h, w, 2]
    query_points = repeat(
        query_points,
        "b h w d -> b t h w d",
        t=num_frames
    )
    query_points_latents = encode_tracks_latte(
        vae,
        vae_scale,
        dino,
        processor,
        query_points,
        frames,
        device
    )[:, 0, ...]  # [b, c, h, w]
    # Encode frames
    latents = encode_tracks_latte(
        vae,
        vae_scale,
        dino,
        processor,
        tracks,
        frames,
        device
    )  # [b, t, c, h, w]
    # Generate noise
    noise = fm.sample_noise_like(latents)
    x0 = noise
    x1 = latents
    # Sample time
    t = log_normal_timestep(batch_size).to(device)
    # Get conditional flow
    _, xt, ut = fm.sample_location_and_conditional_flow(x0, x1, t=t)
    # Condition on the inital position
    # [b, t, c, h, w]
    xt[:, 0, :, :, :] = query_points_latents
    # Move to device
    xt = xt.to(device)
    ut = ut.to(device)
    # Create mask
    masks = torch.ones_like(latents)
    masks[:, 0, :, :, :] = 0  # We are not denoising the initial frame!
    if use_amp:
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # Get field estimate
            vt = model(xt, t, features)
            # Mask out the initial position
            vt = masks * vt
            ut = masks * ut
            # Regress the conditional flow
            return F.mse_loss(vt, ut, reduction="none")
    else:
        # Get field estimate
        vt = model(xt, t, features)
        # Mask out the initial position
        vt = masks * vt
        ut = masks * ut
        # Regress the conditional flow
        return F.mse_loss(vt, ut, reduction="none")


@hydra.main(
    version_base=None,
    config_path=config_path,
    config_name=config_name
)
def main(cfg: DictConfig):
    # DDP setup
    dist_config = setup_distributed()
    device = dist_config.device
    rank = dist_config.global_rank
    # Seed
    seed_worker_fn = seed_everything(
        dist_config.global_rank + cfg.training.seed)
    # Setup checkpoints path
    checkpoints_path = Path(cfg.training.checkpoints_path)
    train_samples_path = Path(cfg.training.samples_path)
    test_samples_path = Path(cfg.evaluation.samples_path)
    checkpoints_path.mkdir(exist_ok=True, parents=True)
    train_samples_path.mkdir(exist_ok=True, parents=True)
    test_samples_path.mkdir(exist_ok=True, parents=True)
    # Dataset setup
    sampling_rng = torch.Generator(
        device).manual_seed(cfg.training.seed + rank)
    # Dataloaders
    loaders = get_dataloaders(
        cfg,
        dist_config.global_rank,
        seed_worker_fn,
        "training"
    )
    train_dataloader = loaders[DatasetSplit.TRAIN]
    val_dataloader = loaders[DatasetSplit.VALIDATION]
    val_iterator = iter(val_dataloader)
    # Initialize model, loss function, and optimizer
    model = DDP(get_model(cfg).to(device), device_ids=[dist_config.local_rank])
    fm = {
        "target_conditional_flow_matcher": TargetConditionalFlowMatcher()
    }[cfg.model.flow_matcher]
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr
    )
    lr_scheduler = get_lr_scheduler(optimizer, cfg)
    if rank == 0:
        summary(model)
        print(OmegaConf.to_yaml(cfg))
    initial_epoch = 0
    step = initial_step = 0
    checkpoint_wandb = None
    should_restore = False
    if hasattr(cfg.training, "restart_from_ckpt"):
        (
            model,
            optimizer,
            lr_scheduler,
            initial_step,
            initial_epoch,
            checkpoint_wandb
        ) = load(
            checkpoint_path=cfg.training.restart_from_ckpt,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            should_restore_optimizer=(
                cfg.training.should_restore_optimizer if hasattr(
                    cfg.training, "should_restore_optimizer") else True
            )
        )
        print(f"loaded ckpt {cfg.training.restart_from_ckpt}.")
        step = initial_step
        should_restore = True

    # Load VAE
    vae = LatteVAE(
        encoder_depth=12,
        decoder_depth=12,
        hidden_size=384,
        patch_size=2,
        num_heads=6,
        in_channels=2,
        latent_channels=8,
        input_size=32,
        num_frames=24,
        learn_sigma=False
    )
    # vae_ckpt = "/scratch/shared/beegfs/gabrijel/experiments/track-compression/kubric_dense_full/sd3.5-vae-mse+decoder_only/epoch=16.pth"  # noqa: E501
    ckpt = torch.load(
        cfg.model.vae.ckpt,
        map_location=device,
        weights_only=True
    )
    vae.load_state_dict(
        state_dict={
            k.replace("module.", ""): v
            for (k, v) in ckpt["model_state_dict"].items()
        }
    )
    vae = vae.to(device)
    vae.eval()
    vae_scale = 1.0 / torch.tensor(
        cfg.model.vae.scale,
        device=device
    )
    # Load DINO
    processor = AutoImageProcessor.from_pretrained(
        "/scratch/shared/beegfs/gabrijel/hf/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056"  # noqa: E501
    )
    dino = AutoModel.from_pretrained(
        "/scratch/shared/beegfs/gabrijel/hf/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056"  # noqa: E501
    )
    dino = dino.to(device)
    dino.eval()
    model.train()
    # Required for calling scheduler after the optimizer
    optimizer_called: bool = False
    # Initialize wandb (only rank 0 should log)
    if rank == 0 and cfg.training.use_wandb:
        if checkpoint_wandb and cfg.training.log_to_existing_wandb_run:
            wandb.init(
                project="track-generator",
                config=OmegaConf.to_container(cfg),
                id=checkpoint_wandb["run"]["id"],
                resume="must"
            )
        else:
            wandb.init(
                project="track-generator",
                config=OmegaConf.to_container(cfg)
            )
        wandb.watch(
            model,
            log="all",
            log_freq=cfg.training.log_every_steps
        )
    # Training loop
    for epoch in range(initial_epoch, cfg.training.epochs):
        running_loss = 0.0
        avg_loss = 0.0
        grad_norm = 0.0
        if rank == 0:
            batch_loader = tqdm(
                iter(train_dataloader),
                desc=f"epoch {epoch+1}/{cfg.training.epochs}"
            )
        else:
            batch_loader = iter(train_dataloader)
        for batch_idx, batch in enumerate(batch_loader):
            # Advance to the checkpoint step
            if should_restore:
                if step < initial_step:
                    step += 1
                    continue
                if step == cfg.training.num_steps_per_epoch:
                    should_restore = False
                    break
            loss = train_step(
                model,
                vae,
                vae_scale,
                fm,
                dino,
                processor,
                batch,
                device,
                cfg.model.nn.trajectory_stride,
                cfg.training.use_amp
            )
            if "max_train_loss" in cfg.training:
                # Filter out using the absolute value
                loss = loss[loss < cfg.training.max_train_loss]
                if loss.numel() == 0:
                    loss = torch.zeros((1, ), device=loss.device)
            loss = loss.mean()
            # Backward pass
            loss.backward()
            if cfg.training.gradient_accumulation_steps == 1:
                # Get rid of possible NaNs
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                        if torch.isnan(param.grad).any():  # Check if any element is NaN
                            print(
                                f"NaN detected in gradient of parameter: {param}")
                        if torch.isinf(param.grad).any():
                            print(
                                f"Inf detected in gradient of parameter: {param}")
                        param.grad = torch.nan_to_num(
                            param.grad,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0
                        )
                grad_norm = grad_norm ** 0.5
                # Clip gradients
                if "clip_grad_norm" in cfg.training:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()   # Clean up.
                optimizer_called = True
            else:
                if (step > 0) and (step % cfg.training.gradient_accumulation_steps) == 0:
                    # Get rid of possible NaNs
                    grad_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2).item() ** 2
                            # Check if any element is NaN
                            if torch.isnan(param.grad).any():
                                print(
                                    f"NaN detected in gradient of parameter: {param}")
                            if torch.isinf(param.grad).any():
                                print(
                                    f"Inf detected in gradient of parameter: {param}")
                            param.grad = torch.nan_to_num(
                                param.grad,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0
                            )
                    grad_norm = grad_norm ** 0.5
                    # Clip gradients
                    if "clip_grad_norm" in cfg.training:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_called = True
            # Scheduler LR
            if optimizer_called:
                lr_scheduler.step()
            # Keep track of losses
            step_loss = loss * cfg.training.gradient_accumulation_steps
            running_loss += step_loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            # Log metrics
            if rank == 0 and (step > 0) and (step % (cfg.training.log_every_steps) == 0):
                batch_loader.set_postfix({
                    "train_loss": step_loss.item(),
                    "running_avg_train_loss": avg_loss,
                })
                if cfg.training.use_wandb:
                    for param_group in optimizer.param_groups:
                        lr = param_group["lr"]
                    wandb.log({
                        "grad_norm": grad_norm,
                        "train_loss": step_loss.item(),
                        "lr": lr
                    })
            # Log samples
            if rank == 0 and (step % (cfg.training.sample_every_steps) == 0):
                train_batch = batch
                try:
                    val_batch = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    val_batch = next(val_iterator)
                # Sample
                model.eval()
                train_samples = generate_samples(
                    model,
                    vae,
                    vae_scale,
                    processor,
                    dino,
                    train_batch,
                    cfg,
                    step,
                    sampling_rng,
                    train_samples_path,
                    device
                )
                validation_samples = generate_samples(
                    model,
                    vae,
                    vae_scale,
                    processor,
                    dino,
                    val_batch,
                    cfg,
                    step,
                    sampling_rng,
                    test_samples_path,
                    device
                )
                if cfg.training.use_wandb:
                    for length in cfg.sampling.sampling_trajectory_lengths:
                        if "motion" in train_samples[length]:
                            wandb.log({
                                f"train_samples_motion(l={length})": wandb.Video(
                                    str(train_samples[length]["motion"])
                                ),
                                f"train_samples_trajectories(l={length})": wandb.Image(
                                    train_samples[length]["reference"]
                                ),
                                f"validation_samples_motion(l={length})": wandb.Video(
                                    str(validation_samples[length]["motion"])
                                ),
                                f"validation_samples_trajectories(l={length})": wandb.Image(
                                    validation_samples[length]["reference"]
                                )
                            })
                        else:
                            wandb.log({
                                f"train_samples_trajectories(l={length})": wandb.Image(
                                    train_samples[length]
                                ),
                                f"validation_samples_trajectories(l={length})": wandb.Image(
                                    validation_samples[length]
                                )
                            })
                model.train()
            # Advance the step
            step += 1
        # Log epoch loss
        if rank == 0 and cfg.training.use_wandb:
            wandb.log({
                "epoch_train_loss": avg_loss,
                "epoch": epoch + 1,
            })
        if rank == 0 and (epoch % (cfg.training.checkpoint_every_epochs) == 0):
            save(
                checkpoint_path=str(checkpoints_path /
                                    f"ckpt_epoch={epoch}.pth"),
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                step=step,
                epoch=epoch,
                cfg=cfg
            )
            print("model checkpoint saved.")
        # Save the final model checkpoint only from the main process
        if rank == 0:
            save(
                checkpoint_path=str(checkpoints_path / "ckpt_current.pth"),
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                step=step,
                epoch=epoch,
                cfg=cfg
            )
            print("model checkpoint saved.")
    if rank == 0:
        save(
            checkpoint_path=str(checkpoints_path / "ckpt_final.pth"),
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            step=step,
            epoch=epoch,
            cfg=cfg
        )
        print("final checkpoint saved.")


if __name__ == "__main__":
    main()
