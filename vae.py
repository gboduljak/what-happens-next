import math

import torch
from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder
from dino import encode_reference_frames
from einops import rearrange, repeat
from preprocessing import denormalize_points, normalize_points
from transformers import AutoImageProcessor, AutoModel

from nn_latte_vae import LatteVAE


@torch.no_grad()
def encode_tracks_latte(
    vae: LatteVAE,
    scale: torch.Tensor,
    dino: AutoModel,
    processor: AutoImageProcessor,
    raw_tracks: torch.Tensor,
    raw_frames: torch.Tensor,
    device: torch.device
):
    # raw_tracks: [b, t, h, w, d]
    # raw_frames: [b, t, h, w, c]
    # scale: [c, ]
    # Rearrange tracks
    tracks = rearrange(
        normalize_points(raw_tracks, 256, 256),
        "b t h w d -> b t d h w"
    )
    # Encode features
    features = encode_reference_frames(
        reference_frames=raw_frames[:, 0, ...],
        processor=processor,
        dino=dino,
        device=device
    )
    latents = (
        vae.encode(
            tracks.to(device),
            features.to(device)
        )
        .rsample()
    )  # [b, t, num_tokens, num_latent_channel]
    b, t, num_tokens, _ = latents.shape
    h = w = int(math.sqrt(num_tokens))
    gamma = repeat(
        scale.to(device),
        "c -> b t c h w",
        t=t,
        b=b,
        h=h,
        w=w
    )
    return gamma * rearrange(
        latents,
        "b t (h w) c -> b t c h w",
        h=h,
        w=w
    )


@torch.no_grad()
def decode_tracks_latte(
    vae: LatteVAE,
    scale: torch.Tensor,
    dino: AutoModel,
    processor: AutoImageProcessor,
    latents: torch.Tensor,
    raw_frames: torch.Tensor,
    device: torch.device
):
    batch_size, num_frames, _,  h, w = latents.shape
    # Encode features
    features = encode_reference_frames(
        reference_frames=raw_frames[:, 0, ...],
        processor=processor,
        dino=dino,
        device=device
    )
    gamma = repeat(
        scale.to(device),
        "c -> b t c h w",
        b=batch_size,
        t=num_frames,
        h=h,
        w=w
    )
    tracks = vae.decode(
        rearrange(
            (1.0 / gamma) * latents.to(device),
            "b t c h w -> b t (h w) c"
        ),
        features
    )
    _, _, _, h, w = tracks.shape
    return rearrange(
        denormalize_points(
            rearrange(
                tracks,
                "b t c h w -> b (h w) t c"
            ),
            256,
            256
        ),
        "b (h w) t c -> b t h w c",
        h=h,
        w=w
    )


@torch.no_grad()
def encode_tracks(
    vae: AutoencoderKL | AutoencoderKLTemporalDecoder,
    scale: torch.Tensor,
    raw_tracks: torch.Tensor,
    device: torch.device
):
    # raw_tracks: [b, t, h, w, d]
    # scale: [c, ]
    batch_size, num_frames, h, w, _ = raw_tracks.shape
    raw_tracks_as_rgb = torch.cat(
        [
            rearrange(
                normalize_points(raw_tracks, w, h),
                "b t h w d -> b h w t d"
            ),
            torch.zeros((batch_size, h, w, num_frames, 1),
                        device=raw_tracks.device)
        ],
        dim=-1
    )  # [b, h, w, t, 3]
    tracks = rearrange(
        raw_tracks_as_rgb,
        "b h w t c -> (b t) c h w"
    )  # [(b t), 3, h, w]
    latents = (
        vae.encode(tracks.to(device))
        .latent_dist
        .mode()
    )
    b, c, h, w = latents.shape
    gamma = repeat(
        scale.to(device),
        "c -> b c h w",
        b=b,
        h=h,
        w=w
    )
    return rearrange(
        gamma * latents,
        "(b t) c h w -> b t c h w",
        b=batch_size,
        t=num_frames
    )


@torch.no_grad()
def decode_latents(
    vae: AutoencoderKL,
    scale: torch.Tensor,
    latents: torch.Tensor,
    device: torch.device
):
    batch_size, num_frames, _,  h, w = latents.shape
    gamma = repeat(
        scale.to(device),
        "c -> b t c h w",
        b=batch_size,
        t=num_frames,
        h=h,
        w=w
    )
    frames = rearrange(
        vae.decode(
            rearrange(
                (1.0 / gamma) * latents.to(device),
                "b t c h w -> (b t) c h w",
                b=batch_size,
                t=num_frames
            )
        ).sample,
        "(b t) c h w -> b t c h w",
        b=batch_size,
        t=num_frames
    )
    _, _, _, h, w = frames.shape
    return rearrange(
        denormalize_points(
            rearrange(
                frames[:, :, :2, :, :],
                "b t c h w -> b (h w) t c"
            ),
            w,
            h
        ),
        "b (h w) t c -> b t h w c",
        h=h,
        w=w
    )


@torch.no_grad()
def decode_tracks_svd(
    vae: AutoencoderKLTemporalDecoder,
    scale: torch.Tensor,
    latents: torch.Tensor,
    device: torch.device
):
    batch_size, num_frames, _,  h, w = latents.shape
    gamma = repeat(
        scale.to(device),
        "c -> b t c h w",
        b=batch_size,
        t=num_frames,
        h=h,
        w=w
    )
    tracks = rearrange(
        vae.decode(
            rearrange(
                (1.0 / gamma) * latents.to(device),
                "b t c h w -> (b t) c h w",
                b=batch_size,
                t=num_frames
            ),
            num_frames=num_frames
        ).sample,
        "(b t) c h w -> b t c h w",
        b=batch_size,
        t=num_frames
    )
    _, _, _, h, w = tracks.shape
    return rearrange(
        denormalize_points(
            rearrange(
                tracks[:, :, :2, :, :],
                "b t c h w -> b (h w) t c"
            ),
            w,
            h
        ),
        "b (h w) t c -> b t h w c",
        h=h,
        w=w
    )


@torch.no_grad()
def encode_frames(
    vae: AutoencoderKL | AutoencoderKLTemporalDecoder,
    scale: torch.Tensor,
    raw_frames: torch.Tensor,
    device: torch.device
):
    # raw_frames: [b t h w c]
    # scale: [c, ]
    batch_size, num_frames, *_ = raw_frames.shape
    # normalized_frames
    frames = rearrange(
        2 * (raw_frames / 255.) - 1,
        "b t h w c -> (b t) c h w"
    )
    latents = vae.encode(frames.to(device)).latent_dist.sample()
    b, c, h, w = latents.shape
    gamma = repeat(
        scale.to(device),
        "c -> b c h w",
        b=b,
        h=h,
        w=w
    )
    return rearrange(
        gamma * latents,
        "(b t) c h w -> b t c h w",
        b=batch_size,
        t=num_frames
    )


@torch.no_grad()
def decode_frames(
    vae: AutoencoderKL,
    scale: torch.Tensor,
    latents: torch.Tensor,
    device: torch.device
):
    batch_size, num_frames, _,  h, w = latents.shape
    gamma = repeat(
        scale.to(device),
        "c -> b t c h w",
        b=batch_size,
        t=num_frames,
        h=h,
        w=w
    )
    frames = rearrange(
        vae.decode(
            rearrange(
                (1.0 / gamma) * latents.to(device),
                "b t c h w -> (b t) c h w",
                b=batch_size,
                t=num_frames
            )
        ).sample,
        "(b t) c h w -> b t c h w",
        b=batch_size,
        t=num_frames
    )
    return rearrange(
        torch.clip(
            ((frames + 1.0) / 2) * 255,
            0,
            255
        ).byte(),
        "b t c h w -> b t h w c"
    )


@torch.no_grad()
def decode_frames_svd(
    vae: AutoencoderKLTemporalDecoder,
    scale: torch.Tensor,
    latents: torch.Tensor,
    device: torch.device
):
    batch_size, num_frames, _,  h, w = latents.shape
    gamma = repeat(
        scale.to(device),
        "c -> b t c h w",
        b=batch_size,
        t=num_frames,
        h=h,
        w=w
    )
    frames = rearrange(
        vae.decode(
            rearrange(
                (1.0 / gamma) * latents.to(device),
                "b t c h w -> (b t) c h w",
                b=batch_size,
                t=num_frames
            ),
            num_frames=num_frames
        ).sample,
        "(b t) c h w -> b t c h w",
        b=batch_size,
        t=num_frames
    )
    return rearrange(
        torch.clip(
            ((frames + 1.0) / 2) * 255,
            0,
            255
        ).byte(),
        "b t c h w -> b t h w c"
    )
