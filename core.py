# flake8: noqa: E501
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, Literal

import torch
import torch.nn as nn
import webdataset as wds
from omegaconf import DictConfig
from torch.utils.data import Dataset

from dataset_split import DatasetSplit
from prefetcher import PrefetchDataLoader
from preprocessing import PreprocessingConfig
from utils.kubric import KubricDataPipeline

SHUFFLE_BUFFER_SIZE = 256


def get_model(cfg) -> nn.Module:
  if "phys101" in cfg.dataset.name or "cityscapes" in cfg.dataset.name:
    from nn_latte_rectangular import Latte
  else: 
    from nn_latte import Latte

  if "latte" in cfg.model.name:
    if "L" in cfg.model.name:
      if "cityscapes" in cfg.dataset.name:
        return Latte(
            depth=16,
            hidden_size=768,
            patch_size=cfg.model.nn.patch_size,
            num_heads=12,
            learn_sigma=False,
            in_channels=cfg.model.nn.in_channels,
            num_frames=cfg.model.nn.get("trajectory_length", 24),
            frame_features_dim=cfg.model.nn.get("frame_features_dim", 384),
            input_size=cfg.model.nn.input_size,
            text_embed_dim=cfg.model.nn.get("text_embed_dim", None),
            num_horizontal_frame_tokens=cfg.model.nn.num_horizontal_frame_tokens, # type: ignore
            num_vertical_frame_tokens=cfg.model.nn.num_vertical_frame_tokens, # type: ignore
            latent=cfg.model.nn.get("latent", True)
        )
      else:
        return Latte(
            depth=16,
            hidden_size=768,
            patch_size=cfg.model.nn.patch_size,
            num_heads=12,
            learn_sigma=False,
            in_channels=cfg.model.nn.in_channels,
            num_frames=cfg.model.nn.get("trajectory_length", 24),
            frame_features_dim=cfg.model.nn.get("frame_features_dim", 384),
            input_size=cfg.model.nn.input_size,
            text_embed_dim=cfg.model.nn.get("text_embed_dim", None),
            num_frame_tokens=cfg.model.nn.get("num_frame_tokens", 256),
            latent=cfg.model.nn.get("latent", True)
        )
    if "XS" in cfg.model.name:
      return Latte(
          depth=8,
          hidden_size=192,
          patch_size=cfg.model.nn.patch_size,
          learn_sigma=False,
          num_heads=3,
          in_channels=cfg.model.nn.in_channels,
          num_frames=cfg.model.nn.get("trajectory_length", 24),
          frame_features_dim=cfg.model.nn.get("frame_features_dim", 384),
          input_size=cfg.model.nn.input_size,
          text_embed_dim=cfg.model.nn.get("text_embed_dim", None),
          num_frame_tokens=cfg.model.nn.get("num_frame_tokens", 256),
          latent=cfg.model.nn.get("latent", True)
      )
    if ("B" in cfg.model.name) or ("S" in cfg.model.name):
      if  "cityscapes" in cfg.dataset.name:
        return Latte(
            depth=12,
            hidden_size=384,
            patch_size=cfg.model.nn.patch_size,
            learn_sigma=False,
            num_heads=6,
            in_channels=cfg.model.nn.in_channels,
            num_frames=cfg.model.nn.get("trajectory_length", 24),
            frame_features_dim=cfg.model.nn.get("frame_features_dim", 384),
            input_size=cfg.model.nn.input_size,
            text_embed_dim=cfg.model.nn.get("text_embed_dim", None),
            num_horizontal_frame_tokens=cfg.model.nn.num_horizontal_frame_tokens,
            num_vertical_frame_tokens=cfg.model.nn.num_vertical_frame_tokens,
            latent=cfg.model.nn.get("latent", True)
        )
      else:
        return Latte(
            depth=12,
            hidden_size=384,
            patch_size=cfg.model.nn.patch_size,
            learn_sigma=False,
            num_heads=6,
            in_channels=cfg.model.nn.in_channels,
            num_frames=cfg.model.nn.get("trajectory_length", 24),
            frame_features_dim=cfg.model.nn.get("frame_features_dim", 384),
            input_size=cfg.model.nn.input_size,
            text_embed_dim=cfg.model.nn.get("text_embed_dim", None),
            num_frame_tokens=cfg.model.nn.get("num_frame_tokens", 256),
            latent=cfg.model.nn.get("latent", True)
        )
    return Latte(
        depth=12,
        hidden_size=384,
        patch_size=cfg.model.nn.patch_size,
        learn_sigma=False,
        num_heads=6,
        in_channels=cfg.model.nn.in_channels,
        num_frames=cfg.model.nn.get("trajectory_length", 24),
        frame_features_dim=cfg.model.nn.get("frame_features_dim", 384),
        input_size=cfg.model.nn.input_size,
        text_embed_dim=cfg.model.nn.get("text_embed_dim", None),
        num_frame_tokens=cfg.model.nn.get("num_frame_tokens", 256),
        latent=cfg.model.nn.get("latent", True),
    )

def get_dataloaders(
    cfg: DictConfig,
    rank: int,
    worker_init_fn: Callable[[int], Any],
    mode: Literal["training", "sampling"] = "training",
) -> Dict[DatasetSplit, Dataset]:
    batch_rng = torch.Generator().manual_seed(cfg[mode].seed + rank)
    preprocess_cfg = PreprocessingConfig(
        max_width=cfg.preprocessing["max_width"],
        max_height=cfg.preprocessing["max_height"],
        pixel_mean=torch.tensor(cfg.preprocessing["pixel_mean"]),
        pixel_std=torch.tensor(cfg.preprocessing["pixel_std"]),
        time=cfg.preprocessing["time"]
    )
    if "kubric" in cfg.dataset.name:
        from utils.kubric import filter_valid_trajectories, map_to_batch
        train_path = str(
            Path(cfg.dataset.path) /
            DatasetSplit.TRAIN /
            f"{cfg.dataset.image_size}x{cfg.dataset.image_size}" /
            "*.tar"
        )
        val_path = str(
            Path(cfg.dataset.path) /
            DatasetSplit.VALIDATION /
            f"{cfg.dataset.image_size}x{cfg.dataset.image_size}" /
            "*.tar"
        )
        train_urls = glob(train_path)
        val_urls = glob(val_path)
        transform = KubricDataPipeline(
            preprocess_cfg,
            cfg,
            mode,
            batch_rng,
            use_pseudo_trajectories="cotracker" in cfg.dataset.name
        )
    elif "libero" in cfg.dataset.name:
        from utils.libero import (LiberoDataPipeline,
                                  filter_valid_trajectories, map_to_batch)
        print(cfg.dataset.name)
        if "tra-moe" in cfg.dataset.name:
            # Dataset paths
            train_urls = []
            val_urls = []

            suffix = "_precomputed_latents" if cfg.training.get(
                "precomputed_latents", False) else ""

            for suite in ["libero_goal", "libero_spatial", "libero_object", "libero_10", "libero_90"]:
                path = f"/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_tra-moe{suffix}/{suite}/train"
                train_urls.extend([
                    p for p in Path(path).glob("*.tar")
                ])

            for suite in ["libero_goal", "libero_spatial", "libero_object", "libero_10"]:
                path = f"/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_tra-moe{suffix}/{suite}/validation"
                val_urls.extend([
                    p for p in Path(path).glob("*.tar")
                ])
        else:
            if cfg.training.get("precomputed_latents", False):
                train_urls = (
                    list(
                        Path(
                            "/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_atm_precomputed_latents/libero_90/train").iterdir()
                    ) + list(
                        Path(
                            "/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_atm_precomputed_latents/libero_10/train").iterdir()
                    )
                )
                val_urls = (
                    list(
                        Path(
                            "/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_atm_precomputed_latents/libero_90/validation").iterdir()
                    ) + list(
                        Path(
                            "/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_atm_precomputed_latents/libero_10/validation").iterdir()
                    )
                )
            else:
                train_urls = (
                    list(
                        Path(
                            "/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_atm/libero_90/train").iterdir()
                    ) + list(
                        Path(
                            "/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_atm/libero_10/train").iterdir()
                    )
                )
                val_urls = (
                    list(
                        Path(
                            "/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_atm/libero_90/validation").iterdir()
                    ) + list(
                        Path(
                            "/scratch/shared/beegfs/gabrijel/datasets/sharded/atm_libero_grid_atm/libero_10/validation").iterdir()
                    )
                )
        transform = LiberoDataPipeline(
            batch_rng, cfg.training.get("precomputed_latents", False))
    elif "physion" in cfg.dataset.name:
        from utils.physion import (filter_valid_trajectories, map_to_batch,
                                   map_to_tuple)
        train_path = str(
            Path(cfg.dataset.path) /
            DatasetSplit.TRAIN /
            "*.tar"
        )
        val_path = str(
            Path(cfg.dataset.path) /
            DatasetSplit.VALIDATION /
            "*.tar"
        )
        train_urls = glob(train_path)
        val_urls = glob(val_path)
        transform = map_to_tuple
    elif "phys101" in cfg.dataset.name:
        from utils.phys101 import (filter_valid_trajectories, map_to_batch,
                                   map_to_tuple)
        train_path = str(
            Path(cfg.dataset.path) /
            DatasetSplit.TRAIN /
            "*.tar"
        )
        val_path = str(
            Path(cfg.dataset.path) /
            DatasetSplit.VALIDATION /
            "*.tar"
        )
        train_urls = glob(train_path)
        val_urls = glob(val_path)
        print(cfg.training.camera_probability)
        transform = map_to_tuple(
            cfg.training.get("camera_probability", 0.75)
        )
    elif "cityscapes" in cfg.dataset.name:
        from utils.cityscapes import (filter_valid_trajectories, map_to_batch,
                                      map_to_tuple)
        train_path = str(
            Path(cfg.dataset.path) /
            DatasetSplit.TRAIN /
            "*.tar"
        )
        val_path = str(
            Path(cfg.dataset.path) /
            DatasetSplit.VALIDATION /
            "*.tar"
        )
        train_urls = glob(train_path)
        val_urls = glob(val_path)
        transform = map_to_tuple()
    else:
        raise NotImplementedError()
    print("train:")
    print(train_urls)
    print("val:")
    print(val_urls)
    # Create webdataset streams
    # trainset = wds.DataPipeline(
    #     wds.SimpleShardList(
    #         urls=[str(x) for x in train_urls] # type: ignore
    #     ),
    #     wds.shuffle(SHUFFLE_BUFFER_SIZE),
    #     wds.split_by_node,
    #     wds.split_by_worker,
    #     wds.repeatedly,  # Make infinite BEFORE filtering
    #     wds.shuffle(SHUFFLE_BUFFER_SIZE),
    #     wds.tarfile_to_samples(),
    #     wds.decode(),
    #     wds.map(transform), # type: ignore
    #     wds.select(filter_valid_trajectories), # type: ignore
    #     wds.batched(cfg.training.batch_size, partial=False),
    #     wds.map(map_to_batch)
    # )

    # Define your WebDataset pipeline
    trainset = (
        wds.WebDataset(
            urls=[str(x) for x in train_urls],
            resampled=True,
            shardshuffle=True,
            nodesplitter=wds.split_by_node
        )
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .decode()
        .map(transform)
        .select(filter_valid_trajectories)
        .batched(cfg.training.batch_size, partial=False)
        .map(map_to_batch)
    )

    valset = wds.DataPipeline(
        wds.SimpleShardList(
            urls=[str(x) for x in val_urls]
        ),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode(),
        wds.map(transform),
        wds.select(filter_valid_trajectories),  # type: ignore
        wds.batched(cfg.training.batch_size, partial=False),
        wds.map(map_to_batch)
    )
    # Pack streams into dataloaders
    return {
        DatasetSplit.TRAIN: PrefetchDataLoader(
            dataloader=(
                wds.WebLoader(
                    trainset,
                    batch_size=None,
                    num_workers=cfg.training.num_workers,
                    worker_init_fn=worker_init_fn,
                    pin_memory=True,
                    persistent_workers=True,
                ).unbatched()
                .shuffle(SHUFFLE_BUFFER_SIZE)  # Shuffle across workers
                .batched(cfg[mode].batch_size, partial=False)  # Rebatch
                .map(map_to_batch)
                # .with_epoch(128)
                .with_epoch(cfg.training.num_steps_per_epoch)
            ),
            prefetch_count=4
        ),  # type: ignore
        DatasetSplit.VALIDATION: (
            wds.WebLoader(
                valset,
                batch_size=None,
                num_workers=1,
                worker_init_fn=worker_init_fn,
                prefetch_factor=4,
                pin_memory=True,
                persistent_workers=True,
            )
        )
    }
