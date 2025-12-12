
import random
from typing import Any, Dict

import torch
from einops import rearrange


def filter_valid_trajectories(sample):
    return True


def map_to_tuple(camera_prob: float = 0.5):

    def mapper(sample: Dict[str, Any]):
        if random.random() <= camera_prob:
            return (
                torch.from_numpy(sample['camera_frames.npy']),
                rearrange(
                    torch.from_numpy(
                        sample['trajectories__camera__trajectories.npy']),
                    "(h w) t d -> t h w d",
                    h=64,
                    w=116
                ),
                torch.from_numpy(sample['camera_features.npy']),
                torch.from_numpy(
                    sample["trajectories__camera__latent_mean.npy"]),
                torch.from_numpy(
                    sample["trajectories__camera__latent_std.npy"]),
                torch.from_numpy(
                    sample["query_points__camera__latent_mean.npy"]),
                torch.from_numpy(
                    sample["query_points__camera__latent_std.npy"]),
            )
        else:
            return (
                torch.from_numpy(sample['kinect_frames.npy']),
                rearrange(
                    torch.from_numpy(
                        sample['trajectories__kinect__trajectories.npy']),
                    "(h w) t d -> t h w d",
                    h=64,
                    w=116
                ),
                torch.from_numpy(sample['kinect_features.npy']),
                torch.from_numpy(
                    sample["trajectories__kinect__latent_mean.npy"]),
                torch.from_numpy(
                    sample["trajectories__kinect__latent_std.npy"]),
                torch.from_numpy(
                    sample["query_points__kinect__latent_mean.npy"]),
                torch.from_numpy(
                    sample["query_points__kinect__latent_std.npy"]),
            )

    return mapper


def map_to_batch(x):
    return x
