import argparse
from pathlib import Path

import numpy as np
import torch
from core import get_model
from cotracker_utils import Visualizer
from diffusers import AutoencoderKL, StableVideoDiffusionPipeline
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from preprocessing import denormalize_points
from seed import seed_everything
from transformers import AutoImageProcessor, AutoModel
from vae import decode_latents, decode_tracks_latte, decode_tracks_svd
from visualizations import plot_tracks_on_reference_frame

from nn_latte_vae import LatteVAE
from sample import pixel_center_grid
from train_latent import sample_latte_latte_vae


def get_sampling_seed(rollout_index: int) -> int:
    import numpy as np
    np.random.seed(42)
    seeds = np.random.randint(0, 2**32, size=64 + 1, dtype=np.uint32)
    return seeds[rollout_index].item()


def main():
    parser = argparse.ArgumentParser(
        description="Script with command-line arguments")
    parser.add_argument("--scene_index", type=int,
                        required=True, help="Index of the scene")
    parser.add_argument("--rollout_index", type=int,
                        required=True, help="Index of the rollout")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--input_dir", type=str,
                        required=True, help="Path to input file")
    parser.add_argument("--output_dir", type=str,
                        required=True, help="Directory to save output")
    parser.add_argument("--queries", type=str, choices=[
                        "objtracks", "gridtracks", "gridtracks32x32"], default="objtracks")
    parser.add_argument("--benchmark", type=str,
                        choices=["rval", "rood"], default="rval")
    args = parser.parse_args()
    # Load config
    config = OmegaConf.load(args.config)
    print("Arguments:", args)
    print("Loaded Config:", config)

    device = torch.device("cuda:0")
    seed = get_sampling_seed(args.rollout_index)
    seed_everything(seed)
    rng = torch.Generator(device).manual_seed(seed)

    out_filename = args.queries
    out_folder = Path(args.output_dir) / \
        f"movi_a_{args.benchmark}_movi_a_{args.benchmark}_{args.scene_index:06d}-{args.rollout_index:06d}"
    out_folder.mkdir(parents=True, exist_ok=True)
    rollout_folder = Path(
        args.input_dir) / f"movi_a_{args.benchmark}_movi_a_{args.benchmark}_{args.scene_index:06d}-{args.rollout_index:06d}"
    reference_frame = Image.open(
        rollout_folder / f"movi_a_{args.benchmark}_{args.scene_index:06d}-{args.rollout_index:06d}_00.png")

    reference_frame = torch.from_numpy(np.array(reference_frame))
    h, w, c = reference_frame.shape
    # Load query points
    if args.queries != "gridtracks32x32":
        query_points = torch.load(
            rollout_folder / ({"objtracks": "q42_obj.pt",
                              "gridtracks": "q42_grid.pt"}[args.queries]),
            map_location="cpu",
            weights_only=True
        )
        query_points = torch.round(query_points)
        query_x, query_y = query_points[:, 1].int(), query_points[:, 2].int()
    # Load model
    model = get_model(config)
    ckpt = torch.load(
        args.ckpt,
        map_location="cpu"
    )
    model.load_state_dict({
        key.replace("module.", ""): value
        for (key, value) in ckpt["model_state_dict"].items()
    })
    model = model.to(device)
    model.eval()
    # Load VAE

    if hasattr(config.model, "vae"):
        if config.model.vae.type == "svd":
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "/scratch/shared/beegfs/gabrijel/hf/hub/models--stabilityai--stable-video-diffusion-img2vid-xt",
            )
            ckpt = torch.load(config.model.vae.ckpt, map_location="cpu")[
                "model_state_dict"]
            vae = pipe.vae
            vae.load_state_dict(ckpt)
            vae = vae.to(device)
        if config.model.vae.type == "latte":
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
            ckpt = torch.load(
                config.model.vae.ckpt,
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
        else:
            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                "/scratch/shared/beegfs/gabrijel/hf/stabilityai/stable-diffusion-3.5-large",
                subfolder="vae"
            )
            vae_ckpt = config.model.vae.ckpt
            vae.load_state_dict(
                torch.load(vae_ckpt, map_location="cpu")["model_state_dict"]
            )
            vae = vae.to(device)
        vae.eval()
        vae_scale = 1.0 / torch.tensor(
            config.model.vae.scale,
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
    # Sample
    samples = []

    if hasattr(config.model, "vae"):
        if config.model.vae.type == "latte":
            latents = sample_latte_latte_vae(
                model,
                vae,
                vae_scale,
                dino,
                processor,
                reference_frame[None, ...],
                device,
                trajectory_stride=8 if "cotracker" in config.model.name else config.model.nn.trajectory_stride,  # 8 for cotracker
                method="euler",
                steps=10,
                atol=1e-7,
                rtol=1e-7,
                rng=rng
            )  # [b, t, c, h, w]
        else:
            latents = sample_latte(
                model,
                vae,
                vae_scale,
                dino,
                processor,
                reference_frame[None, ...],
                device,
                method="dopri5",
                steps=1000,
                atol=1e-7,
                rtol=1e-7,
                rng=rng
            )
        if config.model.vae.type == "svd":
            print("decoding SVD ...")
            pred_tracks = decode_tracks_svd(
                vae,
                vae_scale,
                latents,
                device
            )  # [b, t, h, w, 2]
            samples.append(pred_tracks.unsqueeze(1))
        elif config.model.vae.type == "latte":
            print("decoding latte ...")
            pred_tracks = decode_tracks_latte(
                vae,
                vae_scale,
                dino,
                processor,
                latents,
                reference_frame[None, ...].unsqueeze(1),
                device
            ).cpu()  # [b, t, h, w,  2]
            samples.append(pred_tracks.unsqueeze(1))
        else:
            print("decoding SD3.5 ...")
            samples.append(
                decode_latents(
                    vae,
                    vae_scale,
                    latents,
                    device
                ).unsqueeze(1)
            )  # [b, t, h, w, 2]
    else:
        from train_multinode_latte_raw import sample
        print("sampling raw ...")
        trajectory_stride = config.model.nn.trajectory_stride
        query_points = repeat(
            pixel_center_grid(h, w, device)[
                ::trajectory_stride, ::trajectory_stride, :],
            "h w d -> b h w d",
            b=1
        ).to(device)
        pred_tracks = sample(
            model,
            dino,
            processor,
            reference_frame[None, ...],
            query_points,
            device,
            method="euler",
            steps=10,
            atol=1e-7,
            rtol=1e-7,
        )
        pred_tracks = rearrange(
            pred_tracks,
            "b t c h w -> b t h w c"
        )
        pred_tracks = denormalize_points(pred_tracks, 256, 256)
        samples.append(pred_tracks.unsqueeze(1))
    print("decoded")
    samples = torch.cat(samples, dim=1)
    if args.queries != "gridtracks32x32":
        eval_samples = samples[..., query_y, query_x, :]
        eval_samples[..., 0, :, :] = (
            torch.cat([query_x[:, None], query_y[:, None]], dim=-1)
            .float()
        )
    else:
        eval_samples = rearrange(
            samples,
            "b k t h w d -> b k t (h w) d"
        )
    torch.save(
        eval_samples[0].detach().contiguous().cpu(),
        out_folder / f"{out_filename}.pt",
    )
    visualizer = Visualizer(
        save_dir=str(out_folder),
        mode="rainbow",
        linewidth=1,
        fps=12
    )
    Image.fromarray(
        plot_tracks_on_reference_frame(
            tracks=rearrange(
                eval_samples[0, 0, ...],
                "t n d -> () n t d",
            ).detach().cpu().numpy(),
            reference_frames=reference_frame[None, ...].numpy()
        )[0, ...]
    ).save(
        str(out_folder / f"{out_filename}_sample=0.png")
    )
    sample_tracks = eval_samples[0, [0], ...]
    _, _, t, _, _, _ = samples.shape
    h, w, c = np.array(reference_frame).shape
    visualizer.visualize(
        video=torch.zeros((1, t, c, h, w)),
        tracks=sample_tracks.cpu(),
        filename=f"{out_filename}_sample=0"
    )


if __name__ == "__main__":
    main()
