from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


def get_normalized_patch_centers(
    image_size: Tuple[int, int],
    num_horizontal_patches: int,
    num_vertical_patches: int,
    device: torch.device
) -> torch.Tensor:
  [h, w] = image_size
  # Create meshgrid of patch indices
  i, j = torch.meshgrid(
      torch.linspace(
          0.5,
          num_vertical_patches - 0.5,
          num_vertical_patches,
          device=device
      ),
      torch.linspace(
          0.5,
          num_horizontal_patches - 0.5,
          num_horizontal_patches,
          device=device
      ),
      indexing='ij'
  )
  # Calculate centers
  centers_x = j * (w / num_horizontal_patches)
  centers_y = i * (h / num_vertical_patches)
  # Stack coordinates
  centers = (
      torch.stack([centers_x, centers_y], dim=-1).view(
          (num_vertical_patches * num_horizontal_patches, -1)
      )
  )
  scale = torch.tensor([w, h], device=device)
  # Normalize to [-1,1]
  centers = (
      (centers / scale) * 2 - 1
  )  # [(p * p), 2]
  return centers


def get_normalized_patch_coords(num_vertical_patches: int, num_horizontal_patches: int, device: torch.device):
  x = torch.arange(0, num_horizontal_patches, device=device).float()
  y = torch.arange(0, num_vertical_patches, device=device).float()
  xx, yy = torch.meshgrid(x, y, indexing="xy")
  grid = torch.stack(
      (xx.flatten(), yy.flatten()),
      dim=1
  )
  scale = torch.tensor(
      [num_horizontal_patches, num_vertical_patches],
      device=device
  )
  normalized_grid = (
      (grid / (scale - 1)) * 2 - 1
  )  # [(p * p), 2]
  return normalized_grid


def sample(
    features: torch.Tensor,
    coords: torch.Tensor,
    mode: Literal["bilinear", "nearest"] = "bilinear",
    align_corners: bool = True
) -> torch.Tensor:
  # features: BxNxCxHxW
  # coords: BxNx2
  # Assumes coords are normalized, expressed according to 'xy' notation.
  B, N, C, H, W = features.shape
  B, N, _ = coords.shape
  # transpose grid coords (indexes to height width)
  # coords = coords[:, :, [1, 0]]
  features = rearrange(
      features,
      "b n c h w -> (b n) c h w"
  )
  # normalized_coords = (
  #     (coords / (max_coord - 1)) * 2 - 1
  # )  # [B, N, 2]
  sampling_grid = rearrange(
      coords,
      "b n k -> (b n) () () k"
  )  # [(BxN) 1 1 k]
  sampled_features = F.grid_sample(
      features,
      sampling_grid,
      mode=mode,
      align_corners=align_corners
  )  # [(B N), C, 1, 1]
  return (
      sampled_features
      .view((B, N, C))
  )  # [B, N, C]


def sample_along_trajectory(
    features: torch.Tensor,
    trajectory: torch.Tensor,
    channel_first: bool = False
):
  # Assumes trajectory coordinates are normalized, expressed according to 'xy' notation.
  # features: [b, d, h, w]
  # trajectory: [b, n, t, 2]
  if not channel_first:
    features = rearrange(
        features,
        "b h w d -> b d h w"
    )

  batch_size, num_points, trajectory_length, _ = trajectory.shape
  features = (
      features
      .unsqueeze(1)
      .expand(-1, num_points * trajectory_length, -1, -1, -1)
  )  # [b, (n t), d, h, w]
  trajectory_points = rearrange(
      trajectory,
      "b n t d -> b (n t) d",
  )
  trajectory_features = sample(
      features,
      trajectory_points,
  )
  trajectory_features = rearrange(
      trajectory_features,
      "b (n t) d -> b n t d",
      b=batch_size,
      n=num_points,
      t=trajectory_length
  )
  return trajectory_features
