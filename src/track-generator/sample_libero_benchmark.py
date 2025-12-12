import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
from core import get_model
from einops import rearrange
from omegaconf import OmegaConf
from preprocessing import denormalize_points
from seed import seed_everything
from torchinfo import summary
from tqdm import tqdm

from nn_latte_vae import LatteVAE
from train_latent_libero import decode_tracks_latte, sample_latte_latte_vae
from train_raw_libero import sample_latte_raw


def get_sampling_seed(rollout_index: int) -> int:
    import numpy as np
    np.random.seed(42)
    seeds = np.random.randint(0, 2**32, size=64 + 1, dtype=np.uint32)
    return seeds[rollout_index].item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load config, checkpoint, and run sampling.")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config file.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--samples_dir", type=str, required=True,
                        help="Path to the directory where samples will be saved.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run the model on (e.g., 'cuda:2', 'cpu').")
    parser.add_argument("--tar_idx", type=int, default=0,
                        help="Target index for sampling.")
    parser.add_argument("--rollout_idx", type=int, default=0,
                        help="Rollout index for sampling.")
    parser.add_argument("--split", type=str, default="atm",
                        choices=["atm", "tra-moe"], required=False)
    parser.add_argument(
        "--suite", type=str, choices=["libero_10", "libero_90", "libero_goal", "libero_spatial", "libero_object"], required=True,
        help="Which LIBERO suite to use (e.g., libero_10 or libero_90)."
    )
    parser.add_argument("--weights", type=str,
                        choices=["model", "ema"], default="model")
    return parser.parse_args()


def map_to_tuple(sample: Dict[str, Any]):
    return (
        torch.from_numpy(sample['frames.npz']['agent']),
        torch.from_numpy(
            sample['frame_features__agent__last_hidden_state.npy']
        ),
        rearrange(
            torch.from_numpy(sample['trajectories__agent__trajectories.npy']),
            "(h w) t d -> t h w d",
            h=64,
            w=64
        ),
        torch.from_numpy(sample['frames.npz']['gripper']),
        torch.from_numpy(
            sample['frame_features__gripper__last_hidden_state.npy']
        ),
        rearrange(
            torch.from_numpy(
                sample['trajectories__gripper__trajectories.npy']),
            "(h w) t d -> t h w d",
            h=64,
            w=64
        ),
        sample['instruction.txt'],
        torch.from_numpy(
            sample['instruction_features__last_hidden_state.npy']),
        torch.from_numpy(
            sample['instruction_features__text_embeds.npy']),
        sample['__key__']
    )


def sample_rollout_tracks(
    model,
    frames_features,
    instruction_features,
    instruction_pooled,
    rollout_idx,
    trajectory_stride,
    device,
) -> torch.Tensor:

    print("sampling raw ...")
    sampling_seed = get_sampling_seed(rollout_idx)
    sampling_rng = torch.Generator(device).manual_seed(sampling_seed)

    pred_tracks = sample_latte_raw(
        model,
        frames_features[:, 1:, :],
        instruction_features,
        instruction_pooled,
        device,
        trajectory_stride=trajectory_stride,
        method="euler",
        steps=10,
        atol=1e-7,
        rtol=1e-7,
        rng=sampling_rng
    )
    pred_tracks = rearrange(
        pred_tracks,
        "b t c h w -> b t h w c"
    )
    pred_tracks = denormalize_points(pred_tracks, 128, 128)

    return pred_tracks


def sample_rollout_tracks_latent(
    model,
    vae,
    vae_scale,
    frames_features,
    instruction_features,
    instruction_pooled,
    rollout_idx,
    trajectory_stride,
    device,
) -> torch.Tensor:

    print("sampling rollout ...")
    sampling_seed = get_sampling_seed(rollout_idx)
    sampling_rng = torch.Generator(device).manual_seed(sampling_seed)

    pred_latents = sample_latte_latte_vae(
        model,
        vae,
        vae_scale,
        frames_features[:, 1:, :],
        instruction_features,
        instruction_pooled,
        device,
        trajectory_stride=trajectory_stride,
        method="euler",
        steps=10,
        atol=1e-7,
        rtol=1e-7,
        rng=sampling_rng
    )
    pred_tracks = decode_tracks_latte(
        vae,
        vae_scale,
        pred_latents,
        frames_features[:, 1:, :],
        device
    ).cpu()  # [b, t, h, w, 2]
    return pred_tracks


def main():

    args = parse_args()

    cfg = OmegaConf.load(args.config)
    ckpt = args.ckpt
    samples_dir = Path(args.samples_dir)
    device = torch.device(args.device)
    tar_idx = args.tar_idx
    rollout_idx = args.rollout_idx
    suite = args.suite
    weights = args.weights

    print(f"Loaded config from: {args.config}")
    print(f"Checkpoint path: {ckpt}")
    print(f"Weights: {weights}")
    print(f"Suite: {suite}")
    print(f"Samples will be saved to: {samples_dir}")
    print(f"Using device: {device}")
    print(f"Tar index: {tar_idx}, Rollout index: {rollout_idx}")

    trajectory_stride = cfg.model.nn.trajectory_stride

    rollout_dir = samples_dir / f"rollout={rollout_idx:02d}"
    rollout_dir.mkdir(exist_ok=True, parents=True)

    seed = get_sampling_seed(rollout_idx)
    seed_everything(seed)
    # Load model
    model = get_model(cfg).to(device)
    ckpt = torch.load(
        ckpt,
        map_location="cpu"
    )
    state_dict = {
        key.replace("module.", ""): value
        for (key, value) in ckpt[f"{weights}_state_dict"].items()
    }
    state_dict = {
        key.replace("_orig_mod.", ""): value
        for (key, value) in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(summary(model))

    # Load VAE
    # Is latent?
    latent = hasattr(cfg.model, "vae")
    if latent:
        vae = LatteVAE(
            encoder_depth=12,
            decoder_depth=12,
            hidden_size=384,
            patch_size=4,
            num_heads=6,
            in_channels=2,
            latent_channels=8,
            input_size=64,
            num_frames=16,
            num_frame_tokens=81,
            frame_features_dim=1024,
            learn_sigma=False
        )
        ckpt = torch.load(
            cfg.model.vae.ckpt,
            map_location=device,
            weights_only=True
        )
        vae.load_state_dict(
            state_dict={
                k.replace("module.", ""): v
                for (k, v) in ckpt[f"model_state_dict"].items()
            }
        )
        vae = vae.to(device)
        vae.eval()
        vae_scale = 1.0 / torch.tensor(
            cfg.model.vae.scale,
            device=device
        )
    # Setup dataset
    batch_size = 8
    print(args.split)
    if args.split == "tra-moe":
        # Dataset paths
        val_urls = list(
            sorted(
                Path(
                    f"/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_tra-moe/{suite}/validation"
                ).iterdir()
            )
        )
    else:
        val_urls = list(
            Path(
                f"/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_atm/{suite}/validation").iterdir()
        )
    print(val_urls)
    if tar_idx >= len(val_urls):
        print(f"no samples for tar_idx={tar_idx}")
        return
    valset = wds.DataPipeline(
        wds.SimpleShardList(
            urls=[str(x) for x in val_urls][tar_idx]
        ),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode(),
        wds.map(map_to_tuple),
        wds.batched(batch_size),
    )
    for batch in tqdm(iter(valset)):
        (
            agent_frames,
            agent_frames_features,
            agent_tracks,
            gripper_frames,
            gripper_frames_features,
            gripper_tracks,
            instructions,
            instruction_features,
            instruction_pooled,
            key
        ) = batch
        # sample agent
        if latent:
            sampled_agent_tracks = sample_rollout_tracks_latent(
                model,
                vae,
                vae_scale,
                agent_frames_features,
                instruction_features,
                instruction_pooled,
                rollout_idx,
                trajectory_stride,
                device,
            )
        else:
            sampled_agent_tracks = sample_rollout_tracks(
                model,
                agent_frames_features,
                instruction_features,
                instruction_pooled,
                rollout_idx,
                trajectory_stride,
                device,
            )
        # sample gripper
        if latent:
            sampled_gripper_tracks = sample_rollout_tracks_latent(
                model,
                vae,
                vae_scale,
                gripper_frames_features,
                instruction_features,
                instruction_pooled,
                rollout_idx,
                trajectory_stride,
                device,
            )
        else:
            sampled_gripper_tracks = sample_rollout_tracks(
                model,
                gripper_frames_features,
                instruction_features,
                instruction_pooled,
                rollout_idx,
                trajectory_stride,
                device,
            )

        for sampled_agent, sampled_gripper, sample_key in zip(
            sampled_agent_tracks,
            sampled_gripper_tracks,
            key
        ):
            np.savez(
                rollout_dir / f"{sample_key}",
                **{
                    "__key__": np.array(sample_key),
                    "pred_agent_trajectories": sampled_agent.detach().cpu().numpy(),
                    "pred_gripper_trajectories": sampled_gripper.detach().cpu().numpy(),
                }
            )


if __name__ == "__main__":
    main()
    print("done.")
