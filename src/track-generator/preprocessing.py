

from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import torch
from batch import Batch
from einops import rearrange
from example import Example


@dataclass
class PreprocessingConfig:
    max_width: float
    max_height: float
    pixel_mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406])
    pixel_std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225])
    time: List[Literal["forward", "backward"]] = field(
        default_factory=lambda: ["forward"])


def normalize_points(
    points: torch.Tensor,
    max_width: float,
    max_height: float
) -> torch.Tensor:
    scale = torch.tensor([max_width, max_height], device=points.device)
    return (2 * (points / scale) - 1)


def standardize_frames(
    frames: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    frames = frames / 255
    mean = mean.to(frames.device).view(1, 3, 1, 1)
    std = std.to(frames.device).view(1, 3, 1, 1)
    return (frames - mean) / std


def denormalize_points(
    normalized_points: torch.Tensor,
    max_width: float,
    max_height: float
) -> torch.Tensor:
    scale = torch.tensor(
        [max_width, max_height],
        device=normalized_points.device
    )
    return (
        (normalized_points + 1) * 0.5 * scale
    )


def destandardize_frames(
    frames: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    mean = (
        mean.to(frames.device)
        .view(3, 1, 1)
        .broadcast_to(frames.shape)
    )
    std = (
        std.to(frames.device)
        .view(3, 1, 1)
        .broadcast_to(frames.shape)
    )
    return (
        ((frames * std + mean).clamp(0.0, 1.0) * 255)
        .to(torch.uint8)
    )


def preprocess(item, config: PreprocessingConfig):
    return {
        "query_points": normalize_points(
            torch.from_numpy(item["query_points"].copy()),
            max_width=config.max_width,
            max_height=config.max_height
        ),
        "trajectories": normalize_points(
            torch.from_numpy(item["trajectories"].copy()),
            max_width=config.max_width,
            max_height=config.max_height
        ),
        "frames": standardize_frames(
            rearrange(
                torch.from_numpy(item["frames"].copy()),
                "t h w c -> t c h w"
            ),
            mean=config.pixel_mean,
            std=config.pixel_std
        )
    }


def preprocess_kubric(
    batch: Batch,
    config: PreprocessingConfig
):
    from utils.kubric import Batch
    return Batch(
        frames=standardize_frames(
            rearrange(
                batch.frames,
                "t h w c -> t c h w"
            ),
            mean=config.pixel_mean,
            std=config.pixel_std
        ),
        trajectories=normalize_points(
            batch.trajectories,
            max_width=config.max_width,
            max_height=config.max_height
        ),
        trajectory_masks=batch.trajectory_masks,
        segmentations=batch.segmentations,
        visibility=batch.visibility,
        features=batch.features,
        index=batch.index
    )
