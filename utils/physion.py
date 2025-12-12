
from typing import Dict

import torch
from einops import rearrange


def filter_valid_trajectories(sample):
  (
      _,
      _,
      trajectories
  ) = sample
  # Check if all (x, y) coordinates are within [-384, 384]
  return torch.all((trajectories >= -384) & (trajectories <= 384)).item()


def map_to_tuple(data: Dict[str, torch.Tensor]) -> torch.Tensor:
  return (
      torch.from_numpy(data['frames.npy']),
      torch.from_numpy(data['frame_features__last_hidden_state.npy']),
      rearrange(
          torch.from_numpy(data['trajectories__trajectories.npy']),
          "(h w) t d -> t h w d",
          h=64,
          w=64
      )
  )


def map_to_batch(x):
  return x
