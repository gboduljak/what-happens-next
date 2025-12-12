from collections import namedtuple
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd
import torch
from dataset_split import DatasetSplit
from einops import rearrange
from preprocessing import (PreprocessingConfig, normalize_points,
                           standardize_frames)

AtmBenchmarkDemo = namedtuple("AtmBenchmarkDemo", [
    "frames",
    "ground_truth_trajectories",
    "raw_ground_truth_trajectories",
    "atm_trajectories",
    "raw_atm_trajectories",
    "visibility",
    "task_embedding",
    "index"
])
Demo = namedtuple("Demo", ["frames", "trajectories",
                  "visibility", "task_embedding"])
Batch = namedtuple("Batch", ["frames", "trajectories",
                   "trajectory_masks", "visibility", "task_embedding"])


def stratified_train_val_split(df: pd.DataFrame, train_ratio=0.9, seed=42) -> pd.DataFrame:
    # Create a PyTorch random generator
    generator = torch.Generator().manual_seed(seed)
    df = df.copy()
    # Identify unique tasks and shuffle them deterministically
    tasks = df["task"].unique()
    shuffled_tasks = torch.randperm(
        len(tasks),
        generator=generator
    ).tolist()
    # Determine the number of tasks to allocate to training
    train_task_count = int(len(tasks) * train_ratio)
    train_tasks = tasks[shuffled_tasks[:train_task_count]]
    # Assign "train" split to rows with tasks in `train_tasks` and "test" to rows with tasks in `test_tasks`
    df["split"] = df["task"].apply(
        lambda x: DatasetSplit.TRAIN if x in train_tasks else DatasetSplit.VALIDATION
    )
    return df


def load_h5(file):
    # return as a dict.
    def h5_to_dict(h5):
        d = {}
        for k, v in h5.items():
            if isinstance(v, h5py._hl.group.Group):
                d[k] = h5_to_dict(v)
            else:
                d[k] = np.array(v)
        return d

    with h5py.File(file, 'r') as f:
        return h5_to_dict(f)


def sample_window(
    example: Demo,
    window_length: int,
    rng: torch.Generator
) -> Demo:
    # frames:        [t c h w]
    # trajectories:  [t n 2]
    # visibility:    [t n]
    video = example.frames
    trajectories = example.trajectories
    visibility = example.visibility
    num_frames, *_ = trajectories.shape

    assert window_length <= num_frames

    starting_frame_idx = (
        torch.randint(
            0,
            num_frames - window_length,
            size=(1, ),
            generator=rng
        ).item()
    )

    video_window = video[
        starting_frame_idx: starting_frame_idx + window_length,
        ...
    ]
    trajectory_window = trajectories[
        starting_frame_idx: starting_frame_idx + window_length,
        visibility[starting_frame_idx],
        ...
    ]
    visibility_window = visibility[
        starting_frame_idx: starting_frame_idx + window_length,
        visibility[starting_frame_idx],
        ...
    ]

    return Demo(
        video_window,
        trajectory_window,
        visibility_window,
        example.task_embedding
    )


def sample_forward_trajectories(
    demo: Demo,
    num_trajectories: int,
    rng: torch.Generator,
) -> Batch:
    # frames:        [t c h w]
    # trajectories:  [t n 2]
    # visibility:    [t n]
    video = demo.frames
    trajectories = demo.trajectories
    visibility = demo.visibility
    task_embedding = demo.task_embedding
    _, num_points, *_ = trajectories.shape

    # Sample trajectory indices
    trajectory_idx = torch.randint(
        low=0,
        high=num_points,
        size=(num_trajectories, ),
        generator=rng
    )  # [N]
    sampled_trajectories = trajectories[:, trajectory_idx, ...]  # [t N 2]
    sampled_visibility = visibility[:, trajectory_idx]  # [t N]

    # Create mask for not denoising the query point
    masks = torch.ones_like(sampled_trajectories)
    masks[0, ...] = 0.0

    return Batch(
        video,
        sampled_trajectories,
        masks,
        sampled_visibility,
        task_embedding
    )


def sample_backward_trajectories(
    demo: Demo,
    num_trajectories: int,
    rng: torch.Generator,
) -> Batch:
    # frames:        [t c h w]
    # trajectories:  [t n 2]
    # visibility:    [t n]
    return sample_forward_trajectories(
        Demo(
            frames=torch.flip(demo.frames, dims=[0]),
            trajectories=torch.flip(demo.trajectories, dims=[0]),
            visibility=torch.flip(demo.visibility, dims=[0]),
            task_embedding=demo.task_embedding
        ),
        num_trajectories,
        rng
    )


def sample_trajectories(
    demo: Demo,
    num_trajectories: int,
    rng: torch.Generator,
    backward_probability: float = 0.5,
) -> Batch:
    # frames:        [t c h w]
    # trajectories:  [t n 2]
    # visibility:    [t n]
    p = torch.rand(1, generator=rng).item()
    if p < backward_probability:
        # print("sampling backward")
        return sample_backward_trajectories(demo, num_trajectories, rng)
    else:
        # print("sampling forward")
        return sample_forward_trajectories(demo, num_trajectories, rng)


def preprocess_libero(batch: Batch, config: PreprocessingConfig):
    return Batch(
        frames=standardize_frames(
            batch.frames,
            mean=config.pixel_mean,
            std=config.pixel_std
        ),
        trajectories=normalize_points(
            rearrange(
                batch.trajectories,
                "t n d -> n t d"
            ),
            max_width=config.max_width,
            max_height=config.max_height
        ),
        trajectory_masks=rearrange(
            batch.trajectory_masks,
            "t n d -> n t d"
        ),
        visibility=rearrange(
            batch.visibility,
            "t n -> n t"
        ),
        task_embedding=batch.task_embedding
    )


def preproces_libero_benchmark(demo: AtmBenchmarkDemo, config: PreprocessingConfig):
    return AtmBenchmarkDemo(
        frames=standardize_frames(
            demo.frames,
            mean=config.pixel_mean,
            std=config.pixel_std
        ),
        ground_truth_trajectories=normalize_points(
            rearrange(
                demo.ground_truth_trajectories,
                "t n d -> n t d"
            ),
            max_width=config.max_width,
            max_height=config.max_height
        ),
        raw_ground_truth_trajectories=demo.ground_truth_trajectories,
        atm_trajectories=normalize_points(
            rearrange(
                demo.atm_trajectories,
                "t n d -> n t d"
            ),
            max_width=config.max_width,
            max_height=config.max_height
        ),
        raw_atm_trajectories=demo.atm_trajectories,
        visibility=rearrange(
            demo.visibility,
            "t n -> n t"
        ),
        task_embedding=demo.task_embedding,
        index=demo.index
    )


def rescale_trajectories(
    demo: AtmBenchmarkDemo,
    config: PreprocessingConfig
) -> AtmBenchmarkDemo:
    return AtmBenchmarkDemo(
        frames=demo.frames,
        ground_truth_trajectories=demo.ground_truth_trajectories * config.max_width,
        raw_ground_truth_trajectories=demo.raw_ground_truth_trajectories,
        atm_trajectories=demo.atm_trajectories * config.max_width,
        raw_atm_trajectories=demo.atm_trajectories,
        visibility=demo.visibility,
        task_embedding=demo.task_embedding,
        index=demo.index
    )


def map_to_tuple(sample: Dict[str, Any], rng: torch.Generator, precomputed_latents: bool):
    if torch.rand((1,), generator=rng).item() < 0.5:
        if precomputed_latents:
            return (
                torch.from_numpy(sample['frames.npz']['agent']),
                rearrange(
                    torch.from_numpy(
                        sample['trajectories__agent__trajectories.npy']),
                    "(h w) t d -> t h w d",
                    h=64,
                    w=64
                ),
                torch.from_numpy(
                    sample['frame_features__agent__last_hidden_state.npy']),
                torch.from_numpy(
                    sample['instruction_features__last_hidden_state.npy']),
                torch.from_numpy(
                    sample['instruction_features__text_embeds.npy']),
                torch.from_numpy(
                    sample['trajectories__agent__latent_mean.npy']),
                torch.from_numpy(
                    sample['trajectories__agent__latent_std.npy']),
                torch.from_numpy(
                    sample['query_points__agent__latent_mean.npy']),
                torch.from_numpy(
                    sample['query_points__agent__latent_std.npy']),
            )
        else:
            return (
                torch.from_numpy(sample['frames.npz']['agent']),
                rearrange(
                    torch.from_numpy(
                        sample['trajectories__agent__trajectories.npy']),
                    "(h w) t d -> t h w d",
                    h=64,
                    w=64
                ),
                torch.from_numpy(
                    sample['frame_features__agent__last_hidden_state.npy']),
                torch.from_numpy(
                    sample['instruction_features__last_hidden_state.npy']),
                torch.from_numpy(
                    sample['instruction_features__text_embeds.npy']),
            )
    else:
        if precomputed_latents:
            return (
                torch.from_numpy(sample['frames.npz']['gripper']),
                rearrange(
                    torch.from_numpy(
                        sample['trajectories__gripper__trajectories.npy']),
                    "(h w) t d -> t h w d",
                    h=64,
                    w=64
                ),
                torch.from_numpy(
                    sample['frame_features__gripper__last_hidden_state.npy']
                ),
                torch.from_numpy(
                    sample['instruction_features__last_hidden_state.npy']),
                torch.from_numpy(
                    sample['instruction_features__text_embeds.npy']),
                torch.from_numpy(
                    sample['trajectories__gripper__latent_mean.npy']),
                torch.from_numpy(
                    sample['trajectories__gripper__latent_std.npy']),
                torch.from_numpy(
                    sample['query_points__gripper__latent_mean.npy']),
                torch.from_numpy(
                    sample['query_points__gripper__latent_std.npy']),
            )
        else:
            return (
                torch.from_numpy(sample['frames.npz']['gripper']),
                rearrange(
                    torch.from_numpy(
                        sample['trajectories__gripper__trajectories.npy']),
                    "(h w) t d -> t h w d",
                    h=64,
                    w=64
                ),
                torch.from_numpy(
                    sample['frame_features__gripper__last_hidden_state.npy']
                ),
                torch.from_numpy(
                    sample['instruction_features__last_hidden_state.npy']),
                torch.from_numpy(
                    sample['instruction_features__text_embeds.npy']),
            )


def map_to_batch(x):
    return x


def filter_valid_trajectories(x):
    return True


class LiberoDataPipeline:
    def __init__(self, batch_rng: torch.Generator, precomputed_latents: bool):
        self.batch_rng = batch_rng
        self.precomputed_latents = precomputed_latents

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return map_to_tuple(x, self.batch_rng, self.precomputed_latents)
