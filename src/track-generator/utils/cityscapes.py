
from typing import Any, Dict

import torch
from einops import rearrange


def filter_valid_trajectories(sample):
  return True


def map_to_tuple():
    def mapper(sample: Dict[str, Any]):
      return (
        torch.from_numpy(sample['frames.npy']),
        rearrange(
           torch.from_numpy(sample['trajectories__trajectories.npy']),
           "(h w) t d -> t h w d",
           h=56,
           w=112
        ),
        torch.from_numpy(sample['features.npy'])[0],
        sample["__key__"]
      )
    return mapper

def map_to_batch(x):
  return x
