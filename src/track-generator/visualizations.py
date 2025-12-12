from pathlib import Path
from typing import Dict, List, Literal, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from preprocessing import (PreprocessingConfig, denormalize_points,
                           destandardize_frames)
from tqdm import tqdm
from utils import concatenate_gifs


def plot_tracks_comparison(
    rgb: np.array,
    gt_points: np.array,
    pred_points: np.array,
    gt_occluded: np.array,
    pred_occluded: np.array,
    trackgroup=None,
    output_dir: str = "output",
    file_name: str = "tracks_comparison.gif",
    figure_size: Tuple[int, int] = (224, 224)
) -> Path:
    """Plot predicted tracks and ground truth side by side using matplotlib, saving as GIF."""
    disp = []
    cmap = plt.cm.hsv
    image_width, image_height = figure_size

    z_list = np.arange(
        gt_points.shape[0]
    ) if trackgroup is None else np.array(trackgroup)
    z_list = np.random.permutation(np.max(z_list) + 1)[z_list]
    colors = cmap(z_list / (np.max(z_list) + 1))
    figure_dpi = 100

    for i in range(rgb.shape[0]):
        fig, axs = plt.subplots(
            1, 2,
            figsize=(2 * image_width / figure_dpi, image_height / figure_dpi),
            dpi=figure_dpi,
            facecolor='w'
        )
        axs[0].axis('off')
        axs[1].axis('off')

        # Plot predictions on the left
        axs[0].imshow(rgb[i])
        valid_pred = pred_points[:, i, 0] > 0
        valid_pred = np.logical_and(
            valid_pred, pred_points[:, i, 0] < rgb.shape[2] - 1)
        valid_pred = np.logical_and(valid_pred, pred_points[:, i, 1] > 0)
        valid_pred = np.logical_and(
            valid_pred, pred_points[:, i, 1] < rgb.shape[1] - 1)

        colalpha_pred = np.concatenate(
            [colors[:, :-1], 1 - pred_occluded[:, i:i + 1]], axis=1)
        axs[0].scatter(
            pred_points[valid_pred, i, 0] - 0.5,
            pred_points[valid_pred, i, 1] - 0.5,
            s=3,
            c=colalpha_pred[valid_pred]
        )
        axs[0].set_title('Predictions')

        # Plot ground truth on the right
        axs[1].imshow(rgb[i])
        valid_gt = gt_points[:, i, 0] > 0
        valid_gt = np.logical_and(
            valid_gt, gt_points[:, i, 0] < rgb.shape[2] - 1)
        valid_gt = np.logical_and(valid_gt, gt_points[:, i, 1] > 0)
        valid_gt = np.logical_and(
            valid_gt, gt_points[:, i, 1] < rgb.shape[1] - 1)

        colalpha_gt = np.concatenate(
            [colors[:, :-1], 1 - gt_occluded[:, i:i + 1]], axis=1)
        axs[1].scatter(
            gt_points[valid_gt, i, 0] - 0.5,
            gt_points[valid_gt, i, 1] - 0.5,
            s=3,
            c=colalpha_gt[valid_gt]
        )
        axs[1].set_title('Ground Truth')

        plt.subplots_adjust(wspace=0.05, hspace=0)
        fig.canvas.draw()

        # Convert the figure canvas to an image array
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = (
            np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            .reshape(int(height), int(width), 3)
        )
        disp.append(np.copy(img))
        plt.close(fig)

    # Create a GIF using Pillow
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    gif_path = Path(output_dir) / file_name
    images = [Image.fromarray(img) for img in disp]
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   optimize=False, duration=100, loop=0)

    return gif_path


def generate_tracks_comparison(
    v: nn.Module,
    batch: Dict[str, torch.Tensor],
    lengths: List[int],
    cfg: OmegaConf,
    step: int,
    rng: torch.Generator,
    device: torch.device,
):
    b, t, *_ = batch["trajectories"].shape
    sample_idxes = torch.randint(
        low=0, high=b, size=(cfg.training.num_samples,))
    preprocess_cfg = PreprocessingConfig(**cfg.preprocessing)

    trajectories = batch["trajectories"][sample_idxes, ...]  # b t n t 2
    frames = batch["frames"][sample_idxes, ...]              # b t c h w
    query_points = trajectories[:, :, :, 0, :]               # b t n 2

    print("sampling...")
    sampled_trajectories = sample(
        v,
        cfg,
        query_points,
        frames,
        t,
        device,
        rng,
        return_sampling_trajectory=False
    )
    print("plotting...")
    output = {}

    for traj_len in tqdm(lengths):
        samples: List[Path] = []

        for sample_idx in range(cfg.training.num_samples):
            starting_frame_idx = (
                torch.randint(
                    low=0,
                    high=t - traj_len,
                    size=(1, )
                ).item() if t - traj_len > 0
                else 0
            )

            gt_traj = trajectories[sample_idx,
                                   starting_frame_idx, :, :traj_len, ...]
            pred_traj = sampled_trajectories[sample_idx,
                                             starting_frame_idx, :, :traj_len, ...]
            # t' c h w
            rgb = frames[sample_idx,
                         starting_frame_idx:starting_frame_idx + traj_len, ...]
            num_points, *_ = pred_traj.shape

            samples.append(
                plot_tracks_comparison(
                    rgb=rearrange(
                        destandardize_frames(
                            rgb,
                            mean=preprocess_cfg.pixel_mean,
                            std=preprocess_cfg.pixel_std
                        ).detach().cpu().to(torch.float32).numpy() / 255,
                        "t c h w -> t h w c"
                    ),
                    gt_points=(
                        denormalize_points(
                            gt_traj,
                            max_width=preprocess_cfg.max_width,
                            max_height=preprocess_cfg.max_height
                        )
                    ).detach().cpu().to(torch.float32).numpy(),
                    pred_points=(
                        denormalize_points(
                            pred_traj,
                            max_width=preprocess_cfg.max_width,
                            max_height=preprocess_cfg.max_height
                        )
                    ).detach().cpu().to(torch.float32).numpy(),
                    pred_occluded=np.zeros(
                        (num_points, traj_len), dtype=np.float32),
                    gt_occluded=np.zeros(
                        (num_points, traj_len), dtype=np.float32),
                    output_dir=str(Path(cfg.training.samples_path) /
                                   f"step={step}" / f"length={traj_len}"),
                    file_name=f"sample={sample_idx}.gif"
                )
            )

        output[traj_len] = concatenate_gifs(
            samples,
            Path(cfg.training.samples_path) /
            f"step={step}" / f"length={traj_len}.gif"
        )

    return output


def plot_tracks_on_reference_frame(
    tracks: torch.Tensor,
    reference_frames: np.ndarray,
    show_dots=False
) -> np.array:
    """
    tracks: [B, N, T, 2]
    images: [B, H, W, C]
    returns: [B, H, W, C]
    """
    _, h, _, c = reference_frames.shape
    assert c == 3

    images_back = np.clip(reference_frames, 0, 255).astype(np.uint8).copy()
    images_back = images_back.copy()

    tracks = rearrange(tracks, "b n t d -> b t n d")
    color_map = matplotlib.colormaps.get_cmap("cool")
    linewidth = max(int(5 * h / 512), 1)

    results = []
    for traj_set, img in zip(tracks, images_back):
        traj_len = traj_set.shape[0]
        for traj_idx in range(traj_set.shape[1]):
            traj = traj_set[:, traj_idx]  # (T, 2)
            for s in range(traj_len - 1):
                color = np.array(
                    color_map((s) / max(1, traj_len - 2))[:3]) * 255
                cv2.line(
                    img,
                    pt1=(int(traj[s, 0]), int(traj[s, 1])),
                    pt2=(int(traj[s + 1, 0]), int(traj[s + 1, 1])),
                    color=color,
                    thickness=linewidth,
                    lineType=cv2.LINE_AA
                )
                if show_dots:
                    cv2.circle(
                        img, (traj[s, 0], traj[s, 1]), linewidth, color, -1)
        results.append(img)

    results = np.stack(results, dtype=np.uint8)
    return results


def plot_trajectories_on_reference_frame_with_titles(
    trajectories: torch.Tensor,
    frames: torch.Tensor,
    preprocess_cfg: PreprocessingConfig,
    titles: List[str] = [],
    denormalize: bool = True
) -> Image:
    # trajectories: [b k n t 2]
    # frames: [b k c h w]

    def create_title_image(title, width, height=32):
        """Create a title image with given text."""
        title_img = Image.new("RGB", (width, height),
                              color=(0, 0, 0))  # Black background
        draw = ImageDraw.Draw(title_img)
        font = ImageFont.load_default()  # Use default font

        # Calculate text bounding box
        # (left, top, right, bottom)
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Center the text
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2
        draw.text((text_x, text_y), title, fill=(
            255, 255, 255), font=font)  # White text
        return np.array(title_img)

    def annotate_and_concatenate(images, titles, concat: Literal["horizontal", "vertical"]):
        """Annotate images with titles and concatenate"""
        annotated_images = []
        for idx, img in enumerate(images):
            title_image = create_title_image(str(titles[idx]), img.shape[1])
            # Stack title on top of the image
            annotated_img = np.vstack((title_image, img))
            annotated_images.append(annotated_img)
        if concat == "horizontal":
            return np.hstack(annotated_images)
        else:
            return np.vstack(annotated_images)

    n, k, *_ = trajectories.shape
    batch_samples = []

    for batch_idx in range(n):
        samples = []
        if not titles:
            titles = [f"sample={idx}" for idx in range(k)]
        for sample_idx in range(k):
            if denormalize:
                sampled_trajectories = denormalize_points(
                    trajectories[batch_idx, sample_idx, ...],
                    max_width=preprocess_cfg.max_width,
                    max_height=preprocess_cfg.max_height
                )
                reference_frames = rearrange(
                    destandardize_frames(
                        frames[batch_idx, 0, ...],
                        preprocess_cfg.pixel_mean,
                        preprocess_cfg.pixel_std
                    ),
                    "c h w -> h w c"
                )

            else:
                sampled_trajectories = trajectories[batch_idx, sample_idx, ...]
                reference_frames = frames[batch_idx, 0, ...]

            samples.append(
                plot_tracks_on_reference_frame(
                    (
                        sampled_trajectories
                        .detach()
                        .cpu()
                        .numpy()
                    )[None, ...],
                    (
                        reference_frames
                        .detach()
                        .cpu()
                        .numpy()
                    )[None, ...]
                )[0, ...]
            )
        batch_samples.append(
            annotate_and_concatenate(samples, titles, "horizontal")
        )

    return Image.fromarray(np.vstack(batch_samples))


def plot_trajectories_on_reference_frame_with_titles_and_ground_truth(
    sampled_trajectories: torch.Tensor,
    ground_truth_trajectories: torch.Tensor,
    reference_frames: torch.Tensor,
    preprocess_cfg: PreprocessingConfig,
    titles=None
) -> Image:
    # sampled_trajectories: [b k n t 2]
    # ground_truth_trajectories: [b n t 2]
    # reference_frames: [b k c h w]
    samples_grid = plot_trajectories_on_reference_frame_with_titles(
        trajectories=sampled_trajectories,
        frames=reference_frames,
        preprocess_cfg=preprocess_cfg,
        titles=titles
    )
    gt_grid = plot_trajectories_on_reference_frame_with_titles(
        rearrange(
            ground_truth_trajectories,
            "b n t d -> b () n t d",
        ),
        rearrange(
            reference_frames[:, 0, ...],
            "b c h w -> b () c h w"
        ),
        preprocess_cfg=preprocess_cfg,
        titles=["ground truth"]
    )
    return Image.fromarray(np.hstack([gt_grid, samples_grid]))
