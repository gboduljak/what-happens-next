import atexit
import os
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple, Optional

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.profiler import record_function
from torchcfm import TargetConditionalFlowMatcher
from torchdiffeq import odeint
from torchinfo import summary
from tqdm import tqdm

from core import get_dataloaders, get_model
from dataset_split import DatasetSplit
from ema import (ensure_params_compatible, freeze, get_trainable_params,
                 update_ema)
from nn_latte_rectangular import Latte, log_normal_timestep
from preprocessing import normalize_points
from sample import pixel_center_grid
from schedulers import (CosineAnnealingLRWLinearWarmupScheduler,
                        LinearWarmupScheduler)
from seed import seed_everything

DistConfig = namedtuple("DistConfig", ["world_size", "global_rank", "local_rank", "device"])

config_path = os.environ.get(
    "CONFIG_PATH",
    "/users/gabrijel/projects/track-generator/configs/training/kubric/movi_a/latte/"
)
config_name = os.environ.get(
    "CONFIG_NAME",
    "small_latte.yaml"
)


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
        
def load_model(
    checkpoint_path: Path,
    model: Latte,
    ema: Latte,
):
  checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
  model_state_dict = {
      k.replace("module._orig_mod.", ""): v
      for (k, v) in checkpoint["model_state_dict"].items()
  }
  model.load_state_dict(model_state_dict)
  ema.load_state_dict(checkpoint["ema_state_dict"])
  return (
      model,
      ema,
      checkpoint["step"],
      checkpoint["epoch"],
      checkpoint["wandb"]
  )


def load_optimizer(
    checkpoint_path: Path,
    optimizer: AdamW,
    lr_scheduler: LinearWarmupScheduler | CosineAnnealingLRWLinearWarmupScheduler,
    should_restore_optimizer: bool = True
):
  checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

  if should_restore_optimizer:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("Loaded optimizer state.")
  else:
    print("Skipping loading optimizer state.")
  lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

  return (
      optimizer,
      lr_scheduler,
      checkpoint["step"],
      checkpoint["epoch"],
      checkpoint["wandb"]
  )


def save(
    checkpoint_path: str,
    model: Latte,
    ema: Latte,
    optimizer: AdamW,
    lr_scheduler: LinearWarmupScheduler,
    step: int,
    epoch: int,
    cfg: DictConfig,
):
  torch.save(
      {
          "model_state_dict": model.state_dict(),
          "ema_state_dict": ema.state_dict(),
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


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Batch(NamedTuple):
  frames: torch.Tensor
  trajectories: torch.Tensor
  frame_features: torch.Tensor


def cleanup_distributed():
  if dist.is_initialized():
    dist.destroy_process_group()


def setup_distributed():
  rank = int(os.environ['SLURM_PROCID'])  # Global rank
  node_rank = int(os.environ["SLURM_NODEID"])  # Node ID
  world_size = int(os.environ["WORLD_SIZE"])  # Total number of ranks
  local_rank = rank % torch.cuda.device_count()
  os.environ["RANK"] = str(rank)  # Required for NCCL.
  # Force GPU selection to avoid duplicates
  torch.cuda.set_device(local_rank)
  print(
      f"Rank {rank}/{world_size}, Node {node_rank}, Local Rank {local_rank}, GPU: {torch.cuda.current_device()}"
  )
  # Initialize distributed training
  dist.init_process_group(backend="nccl")
  atexit.register(cleanup_distributed)
  return DistConfig(
      world_size=world_size,
      global_rank=rank,
      local_rank=local_rank,
      device=torch.device(f"cuda:{local_rank}")
  )

# def setup_distributed():
#   # Works for torchrun
#   dist.init_process_group(
#       backend="nccl",
#       init_method="env://"
#   )
#   rank = dist.get_rank()
#   world_size = dist.get_world_size()
#   local_rank = rank % torch.cuda.device_count()
#   device = torch.device(f"cuda:{local_rank}")
#   print(f"global rank {rank}/{world_size} initialized.")
#   torch.cuda.set_device(device)
#   atexit.register(cleanup_distributed)
#   return DistConfig(
#       world_size=world_size,
#       global_rank=rank,
#       local_rank=local_rank,
#       device=device
#   )


@torch.compile
def construct_target(
    fm: TargetConditionalFlowMatcher,
    tracks: torch.Tensor,
    batch_size: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Compiled version of flow matching target construction.

  Args:
      tracks: Input tracks tensor [b, t, c, h, w]
      fm: Flow model object (assumed to be GPU-efficient)
      batch_size: Batch size
      device: Target device

  Returns:
      tuple: (xt, t, ut, masks)
  """
  # Normalize tracks
  normalized_tracks = normalize_points(
      tracks,
      448,
      224
  )   # [b, t, h, w, 2]
  normalized_tracks = rearrange(
      normalized_tracks,
      "b t h w d -> b t d h w"
  )
  # Get query points
  query_points = normalized_tracks[:, 0, ...]  # [b, d, h, w]
  # Generate noise
  noise = fm.sample_noise_like(normalized_tracks)
  # Coupling
  x0 = noise
  x1 = normalized_tracks
  # Sample timesteps
  t = log_normal_timestep(batch_size, device)
  # Get interpolation and conditional flow target
  _, xt, ut = fm.sample_location_and_conditional_flow(x0, x1, t=t)
  # Condition on the initial position
  xt[:, 0] = query_points
  # Create mask
  masks = torch.ones_like(xt)
  masks[:, 0] = 0.0
  return xt, t, ut, masks


@torch.compile
def sample_latents(
    mean: torch.Tensor,
    std: torch.Tensor,
    scale: torch.Tensor,
    h: int,
    w: int,
    rng: Optional[torch.Generator] = None
):
  gamma = rearrange(scale, "c -> () () c () ()")
  noise = torch.randn(mean.shape, generator=rng, device=mean.device)
  return gamma * rearrange(
      mean + std * noise,
      "b t (h w) c -> b t c h w",
      h=h,
      w=w
  )


def train_step(
    model: Latte,
    fm: TargetConditionalFlowMatcher,
    batch,
    device: torch.device,
    trajectory_stride: int,
    use_amp: bool
) -> torch.Tensor:
  # Slice
  with record_function("prepare_batch"):
    tracks = batch.trajectories[:, :, ::trajectory_stride, ::trajectory_stride, :]   # [b, t, h, w, 2]
    batch_size, *_ = tracks.shape
    frame_features = batch.frame_features[:, 1:, :]  # [b, n, d], skip CLS
  with record_function("transfer to gpu"):
    # Move to GPU
    tracks = tracks.to(device, non_blocking=True)
    frame_features = frame_features.to(device, non_blocking=True)
  # Construct training target using a compiled kernel
  with record_function("construct target"):
    (xt, t, ut, masks) = construct_target(
        fm,
        tracks,
        batch_size,
        device
    )
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
    with record_function("forward"):
      # Get field estimate
      vt = model(
          xt,
          t,
          frame_features,
      )
    # Regress the conditional flow
    with record_function("loss"):
      # Mask out the initial position
      vt = masks * vt
      ut = masks * ut
      return F.mse_loss(vt, ut, reduction="none")


@torch.inference_mode()
def sample_latte_raw(
    model: Latte,
    frame_features: torch.Tensor,
    device: torch.device,
    trajectory_stride: int,
    atol=1e-5,
    rtol=1e-5,
    method="euler",
    steps=10,
    rng: Optional[torch.Generator] = None
) -> torch.Tensor:
  TARGET_WIDTH = 448
  TARGET_HEIGHT = 224
  print(f"sampling method={method}, steps={steps}, atol={atol}, rtol={rtol} ...")
  model.eval()
  model = model.to(device)
  # query_points: # [b, h, w, c]
  batch_size, *_ = frame_features.shape
  query_points = (
      pixel_center_grid(TARGET_WIDTH, TARGET_HEIGHT)[2::4, 2::4][::trajectory_stride, ::trajectory_stride, :]  # type: ignore
  )
  query_points = normalize_points(query_points, TARGET_WIDTH, TARGET_HEIGHT)
  query_points = repeat(
      query_points,
      "h w d -> b () d h w",
      b=batch_size
  ).to(device)
  _, _, _, h, w = query_points.shape
  # Encode frames
  noise = torch.randn(
      (batch_size, 29, 2, h, w),
      device=query_points.device,
      generator=rng
  )
  x0 = torch.cat([query_points, noise], dim=1)  # [b, t, c, h, w]
  # Mask to select non-initial part of the trajectory.
  m = torch.ones_like(x0)
  m[:, 0, ...] = 0.0
  # To device
  frame_features = frame_features.to(device)

  def f(t: float, x: torch.Tensor) -> torch.Tensor:
    # Set up the conditioning
    x[:, 0, ...] = query_points[:, 0, ...]
    # Set up the time
    t = t * torch.ones(
        size=(batch_size, ),
        device=x.device
    )
    return m * model(
        x,
        t,
        frame_features,
    )
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
  x[:, 0, ...] = query_points[:, 0, ...]
  return x


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
  seed_worker_fn = seed_everything(dist_config.global_rank + cfg.training.seed)
  # Setup checkpoints path
  checkpoints_path = Path(cfg.training.checkpoints_path)
  train_samples_path = Path(cfg.training.samples_path)
  test_samples_path = Path(cfg.evaluation.samples_path)
  checkpoints_path.mkdir(exist_ok=True, parents=True)
  train_samples_path.mkdir(exist_ok=True, parents=True)
  test_samples_path.mkdir(exist_ok=True, parents=True)
  # Dataset setup
  sampling_rng = torch.Generator(device).manual_seed(cfg.training.seed + rank)
  # Training rng
  train_rng = torch.Generator(device).manual_seed(cfg.training.seed + rank)
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
  # Create model
  model = get_model(cfg).to(device)

  vanilla_model = model
  if rank == 0:
    summary(model)
    print(OmegaConf.to_yaml(cfg))
    print("----")
  # Create EMA model
  ema_model = deepcopy(vanilla_model).to(device)
  # Restore
  initial_epoch = 0
  step = initial_step = 0
  checkpoint_wandb = None
  should_restore = False
  if hasattr(cfg.training, "restart_from_ckpt"):
    (
        model,
        ema_model,
        initial_step,
        initial_epoch,
        checkpoint_wandb
    ) = load_model(
        checkpoint_path=cfg.training.restart_from_ckpt,
        model=model,
        ema=ema_model,
        # optimizer=optimizer,
        # lr_scheduler=lr_scheduler,
        # should_restore_optimizer=(
        #     cfg.training.should_restore_optimizer if hasattr(cfg.training, "should_restore_optimizer") else True
        # )
    )
    print(f"loaded ckpt {cfg.training.restart_from_ckpt}.")
    step = initial_step
    should_restore = True

  # Verify EMA params
  ema_named_params = get_trainable_params(ema_model)
  model_named_params = get_trainable_params(vanilla_model)
  ensure_params_compatible(ema_named_params, model_named_params)
  # Compile
  if cfg.training.get("compile", False):
    model = torch.compile(
        model,
        fullgraph=True,
        options={"triton.cudagraphs": False}
    )  # type: ignore
  # Create DDP model
  model = DDP(model, device_ids=[dist_config.local_rank])  # type: ignore
  # Create optimizer
  fm = {
      "target_conditional_flow_matcher": TargetConditionalFlowMatcher()
  }[cfg.model.flow_matcher]
  optimizer = AdamW(
      model.parameters(),
      lr=cfg.training.lr
  )
  lr_scheduler = get_lr_scheduler(optimizer, cfg)

  if hasattr(cfg.training, "restart_from_ckpt"):
    (
        optimizer,
        lr_scheduler,
        _,
        _,
        _
    ) = load_optimizer(
        checkpoint_path=cfg.training.restart_from_ckpt,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        should_restore_optimizer=(
            cfg.training.should_restore_optimizer if hasattr(cfg.training, "should_restore_optimizer") else True
        )
    )
    # This means EMA was restored
  else:
    # Ensure all processes are here
    dist.barrier()
    # Set
    update_ema(
        ema_params=[p for (_, p) in ema_named_params],
        model_params=[p for (_, p) in model_named_params],
        decay=0
    )
  # Freeze ema_model!
  freeze(ema_model)
  # Ensure EMA frozen
  for _, p in ema_named_params:
    assert p.requires_grad == False
  ema_model.load_state_dict(
      {
          buffer: param
          for (buffer, param) in ema_model.named_buffers()
      } | {
          name: param
          for (name, param) in ema_named_params
      }
  )
  model.train()
  ema_model.eval()  # EMA model should always be in eval mode
  # Required for calling scheduler after the optimizer
  optimizer_called: bool = False
  first_run: bool = True
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
  # Ensure all processes are synchronized before starting
  if dist.is_initialized():
    dist.barrier()

  # with profiler_context as prof:
  # Training loop
  for epoch in range(initial_epoch, cfg.training.epochs):
    # if step == total_steps:
    #   break
    batch_loader = tqdm(
        iter(train_dataloader),
        desc=f"epoch {epoch+1}/{cfg.training.epochs}",
        disable=rank != 0
    )
    for batch_idx, batch in enumerate(batch_loader):
      # if step == total_steps:
      #   break
      # Advance to the checkpoint step
      if should_restore:
        if step < initial_step:
          step += 1
          continue
        if step == cfg.training.num_steps_per_epoch:
          should_restore = False
          break
      # Get batch
      frames, tracks, frame_features, *_ = batch
      loss = train_step(
          model,
          fm,
          Batch(frames, tracks, frame_features),
          device,
          trajectory_stride=cfg.model.nn.trajectory_stride,
          use_amp=cfg.training.use_amp
      )
      # Make sure that if model is compiled all are here
      if first_run:
        dist.barrier()
        first_run = False
      with record_function("loss"):
        # if "max_train_loss" in cfg.training:
        #   # Filter out using the absolute value
        #   loss = loss[loss < cfg.training.max_train_loss]
        #   if loss.numel() == 0:
        #     loss = torch.zeros((1, ), device=loss.device)
        # Account for gradient accumulation
        loss = loss.mean() / cfg.training.gradient_accumulation_steps

      with record_function("backward"):
        loss.backward()

      should_step_optimizer = (
          step % cfg.training.gradient_accumulation_steps == 0
      )

      if should_step_optimizer:
        # Clip gradients
        with record_function("clip_grad_norm"):
          if "clip_grad_norm" in cfg.training:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
        with record_function("optimizer_step"):
          optimizer.step()
        with record_function("ema_step"):
          update_ema(
              ema_params=[p for (_, p) in ema_named_params],
              model_params=[p for (_, p) in model_named_params],
          )
        optimizer.zero_grad()   # Clean up.
        optimizer_called = True
      with record_function("lr_scheduler_step"):
        # Scheduler LR
        if optimizer_called:
          lr_scheduler.step()
      # Log metrics
      if rank == 0 and (step > 0) and (step % (cfg.training.log_every_steps) == 0):
        # Keep track of losses
        step_loss = loss * cfg.training.gradient_accumulation_steps
        batch_loader.set_postfix({
            "train_loss": step_loss.item(),
        })
        if cfg.training.use_wandb:
          for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            break
          wandb.log({
              "train_loss": step_loss.item(),
              "lr": lr
          }, step=step)
      # Log samples
      if rank == 0 and (step % (cfg.training.sample_every_steps) == 0):
        train_batch = batch
        try:
          val_batch = next(val_iterator)
        except StopIteration:
          val_iterator = iter(val_dataloader)
          val_batch = next(val_iterator)
        # Sample
        # model.eval()
        # frames, tracks, frame_features, *_ = train_batch
        # train_samples = generate_samples(
        #     vanilla_model,
        #     vae,
        #     vae_scale,
        #     Batch(
        #         frames,
        #         tracks,
        #         None,
        #         None,
        #         frame_features,
        #     ),
        #     cfg,
        #     step,
        #     sampling_rng,
        #     train_samples_path,
        #     device
        # )
        # frames, tracks, frame_features, *_ = val_batch
        # validation_samples = generate_samples(
        #     vanilla_model,
        #     vae,
        #     vae_scale,
        #     Batch(
        #         frames,
        #         tracks,
        #         None,
        #         None,
        #         frame_features,
        #     ),
        #     cfg,
        #     step,
        #     sampling_rng,
        #     test_samples_path,
        #     device
        # )
        # if cfg.training.use_wandb:
        #   for length in cfg.sampling.sampling_trajectory_lengths:
        #     if "motion" in train_samples[length]:
        #       wandb.log({
        #           f"train_samples_motion(l={length})": wandb.Video(
        #               str(train_samples[length]["motion"])
        #           ),
        #           f"train_samples_trajectories(l={length})": wandb.Image(
        #               train_samples[length]["reference"]
        #           ),
        #           f"validation_samples_motion(l={length})": wandb.Video(
        #               str(validation_samples[length]["motion"])
        #           ),
        #           f"validation_samples_trajectories(l={length})": wandb.Image(
        #               validation_samples[length]["reference"]
        #           )
        #       })
        #     else:
        #       wandb.log({
        #           f"train_samples_trajectories(l={length})": wandb.Image(
        #               train_samples[length]
        #           ),
        #           f"validation_samples_trajectories(l={length})": wandb.Image(
        #               validation_samples[length]
        #           )
        #       })
        model.train()
      # Advance the step
      step += 1
      # torch.cuda.synchronize()  # ensure CUDA ops are completed before next step
      # Only step profiler on rank 0
      # if rank == 0 and prof is not None:
      #     prof.step()

    # Log epoch loss
    if rank == 0 and cfg.training.use_wandb:
      wandb.log({
          "epoch": epoch + 1,
      }, step=step)
    if rank == 0 and (epoch % (cfg.training.checkpoint_every_epochs) == 0):
      ema_model.load_state_dict(
          {
              buffer: param
              for (buffer, param) in ema_model.named_buffers()
          } | {
              name: param
              for (name, param) in ema_named_params
          }
      )
      save(
          checkpoint_path=str(checkpoints_path / f"ckpt_epoch={epoch}.pth"),
          model=vanilla_model,
          ema=ema_model,
          optimizer=optimizer,
          lr_scheduler=lr_scheduler,
          step=step,
          epoch=epoch,
          cfg=cfg
      )
      print("model checkpoint saved.")
      # Save the final model checkpoint only from the main process
      if rank == 0:
        ema_model.load_state_dict(
            {
                buffer: param
                for (buffer, param) in ema_model.named_buffers()
            } | {
                name: param
                for (name, param) in ema_named_params
            }
        )
        save(
            checkpoint_path=str(checkpoints_path / "ckpt_current.pth"),
            model=vanilla_model,
            ema=ema_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            step=step,
            epoch=epoch,
            cfg=cfg
        )
        print("model checkpoint saved.")

  if rank == 0:
    ema_model.load_state_dict(
        {
            buffer: param
            for (buffer, param) in ema_model.named_buffers()
        } | {
            name: param
            for (name, param) in ema_named_params
        }
    )
    save(
        checkpoint_path=str(checkpoints_path / "ckpt_final.pth"),
        model=vanilla_model,
        ema=ema_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        step=step,
        epoch=epoch,
        cfg=cfg
    )
    print("final checkpoint saved.")


if __name__ == "__main__":
  main()


if __name__ == "__main__":
  main()
