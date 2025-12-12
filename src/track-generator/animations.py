from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio
import matplotlib
import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from PIL.Image import Image
from scipy.stats import multivariate_normal


def create_custom_colormap(base_color: str, colormap_name: str = "custom_cmap"):
  # Convert base color to RGB
  base_rgb = to_rgb(base_color)
  # Create brighter and darker variants
  bright_rgb = tuple(min(1, c + 0.4) for c in base_rgb)  # Adjust to create bright variant
  dark_rgb = tuple(max(0, c - 0.4) for c in base_rgb)    # Adjust to create dark variant
  # Define the colormap
  return LinearSegmentedColormap.from_list(colormap_name, [bright_rgb, base_rgb, dark_rgb])


def add_title_to_frames(
    frames: List[np.ndarray | Image],
    title: str,
    extra_space: int = 100,  # Additional space at the top for the title
    font_scale: float = 0.8,  # Slightly smaller font size
    font_thickness: int = 2,
    font_color: Tuple[int, int, int] = (255, 255, 255)
) -> List[np.ndarray]:
  from PIL.Image import Image
  updated_frames: List[np.ndarray] = []

  for frame in frames:
    # Get the frame dimensions
    if isinstance(frame, Image):
      frame = np.array(frame)
    height, width, _ = frame.shape
    # Create a new frame with extra space at the top
    new_frame = np.zeros((height + extra_space, width, 3), dtype=frame.dtype)
    new_frame[extra_space:, :, :] = frame  # Copy the original frame content

    # Determine the text size and position
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (extra_space - text_size[1]) // 2 + text_size[1]

    # Add the title text in the extra space
    cv2.putText(new_frame, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    updated_frames.append(new_frame)

  return updated_frames


def animate_tracks_on_rgb(
    tracks: torch.Tensor,
    reference_frames: np.ndarray,
    output_filename: Optional[str] = None,
    fps: int = 10,
    show_dots: bool = False,
    title: str = "RGB + Tracks",
    title_color: Tuple[int, int, int] = (255, 255, 255),
    transparency: float = 0.5,
    color_map: str = matplotlib.colormaps.get("cool"),
    track_history: Optional[int] = None
):
  b, h, w, c = reference_frames.shape
  assert c == 3, "Reference frames must have 3 color channels (RGB)."
  assert tracks.ndim == 4 and tracks.shape[-1] == 2, "Tracks must be [B, N, T, 2]."

  tracks = rearrange(tracks, "b n t d -> b t n d")
  linewidth = max(int(5 * h / 512), 1)
  all_frames = []

  for traj_set, img in zip(tracks, reference_frames):
    traj_len = traj_set.shape[0]
    base_frames = [img.copy().astype(np.uint8) for _ in range(traj_len)]
    colors = np.array([np.array(color_map(t / max(1, traj_len - 2)))[:3]
                      * 255 for t in range(traj_len)], dtype=np.uint8)

    for s in range(traj_len):
      if s > 0:
        # Calculate start index based on track_history
        if track_history is not None:
          start_idx = max(0, s - track_history)
        else:
          start_idx = 0

        points = traj_set[start_idx:s+1].numpy()
        num_trajectories = points.shape[1]
        num_segments = (s - start_idx) * num_trajectories
        all_lines = np.zeros((num_segments, 2, 2), dtype=np.int32)
        line_colors = np.zeros((num_segments, 3), dtype=np.uint8)

        idx = 0
        for t in range(s - start_idx):
          current_points = points[t:t+2]
          current_points = np.transpose(current_points, (1, 0, 2))
          all_lines[idx:idx+num_trajectories] = current_points
          # Adjust color index to match the truncated history
          color_idx = start_idx + t
          line_colors[idx:idx+num_trajectories] = colors[color_idx]
          idx += num_trajectories

        mask = np.zeros_like(base_frames[s])
        for color in np.unique(line_colors, axis=0):
          color_mask = (line_colors == color).all(axis=1)
          lines = all_lines[color_mask]
          if len(lines) > 0:
            cv2.polylines(mask, lines, False, color.tolist(), linewidth, cv2.LINE_AA)

        if show_dots:
          for t in range(len(points)):
            current_points = points[t]
            color_idx = start_idx + t
            color = colors[color_idx]
            for point in current_points:
              cv2.circle(mask, tuple(point.astype(int)), linewidth, color.tolist(), -1)

        cv2.addWeighted(mask, transparency, base_frames[s], 1, 0, base_frames[s])

    if title:
      title_height = 40
      title_img = np.zeros((title_height, w, 3), dtype=np.uint8)
      text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
      text_x = (w - text_size[0]) // 2
      cv2.putText(
          title_img, title, (text_x, 30),
          cv2.FONT_HERSHEY_DUPLEX,
          0.5,
          title_color,
          1,
          cv2.LINE_AA
      )
      frame_sequence = [np.vstack([title_img, frame]) for frame in base_frames]
    else:
      frame_sequence = base_frames
    all_frames.append(frame_sequence)

  flattened_frames = [frame for batch_frames in all_frames for frame in batch_frames]

  if output_filename:
    if output_filename.endswith(".gif"):
      save_frames_as_gif(flattened_frames, output_filename, fps)
    elif output_filename.endswith(".mp4"):
      height, width, _ = flattened_frames[0].shape
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
      for frame in flattened_frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
      video_writer.release()
    else:
      raise ValueError("Output filename must end with .gif or .mp4")

  return flattened_frames


def animate_tracks_on_rgb_grouped(
    tracks: List[Dict[int, torch.Tensor]],
    reference_frames: np.ndarray,
    output_filename: Optional[str] = None,
    fps: int = 10,
    show_dots: bool = False,
    title: str = "RGB + Tracks",
    title_color: Tuple[int, int, int] = (255, 255, 255),
    transparency: float = 0.5,
    different_color_per_object: bool = False,
    frames: Optional[np.array] = None,
    track_history: Optional[int] = None
):
  """
  Visualize grouped trajectories overlaid on RGB reference frames with batch drawing.

  This function generates an animation of point trajectories (tracks)
  overlaid on corresponding RGB frames. The trajectories are rendered
  with adjustable transparency, and individual points can optionally
  be marked with dots. The animation can be saved to a file or displayed.

  Args:
      tracks (List[Dict[int, torch.Tensor]]): A list of dictionaries where keys are object IDs
          and values are tensors of shape [N, T, 2] representing trajectories:
          - N: Number of points per object.
          - T: Number of timesteps.
          - 2: (x, y) coordinates of the points.
      reference_frames (np.ndarray): A numpy array of shape [B, H, W, 3] containing RGB frames:
          - B: Batch size (number of sequences).
          - H: Height of the frames.
          - W: Width of the frames.
          - 3: RGB color channels.
      output_filename (Optional[str]): Filepath to save the animation. The file format is
          inferred from the extension (e.g., '.gif' or '.mp4'). If None, the animation is
          displayed but not saved.
      fps (int): Frames per second for the output animation.
      show_dots (bool): If True, individual points along the trajectories are marked with dots.
      title (str): Title displayed at the top of the visualization.
      title_color (Tuple[int, int, int]): RGB color of the title text.
      transparency (float): Transparency level for trajectory overlays (0: invisible, 1: opaque).
      different_color_per_object (bool): If True, assigns a unique color to each object.
      frames (np.ndarray): A numpy array of shape [B, T, H, W, 3] containing RGB frames:
          - B: Batch size (number of sequences).
          - T: Number of frames.
          - H: Height of the frames.
          - W: Width of the frames.
          - 3: RGB color channels.
      track_history: Optional[int] = None
  Returns:
      List[np.ndarray]: A list of frames (numpy arrays) representing the animation.
  """
  b, h, w, c = reference_frames.shape
  assert c == 3, "Reference frames must have 3 color channels (RGB)."

  linewidth = max(int(5 * h / 512), 1)
  title_height = 40
  all_frames = []

  for (sample_idx, (grouped_tracks, img)) in enumerate(zip(tracks, reference_frames)):
    for object_tracks in grouped_tracks.values():
      assert object_tracks.ndim == 3 and object_tracks.shape[-1] == 2, \
          "Tracks must have shape [N, T, 2]."
      _, traj_len, _ = object_tracks.shape

    if frames is None:
      frame_sequence = [img.copy().astype(np.uint8) for _ in range(traj_len)]
    else:
      frame_sequence = [frames[sample_idx][idx].copy().astype(np.uint8) for idx in range(traj_len)]

    for object_id, traj_set in grouped_tracks.items():
      traj_set = rearrange(traj_set, "n t d -> t n d")
      if different_color_per_object:
        object_color = matplotlib.colormaps.get("cool")(object_id / (len(grouped_tracks) - 1))
        track_cmap = create_custom_colormap(object_color, f"{object_id}")
      else:
        track_cmap = matplotlib.colormaps.get_cmap("cool")

      # Precompute colors for the entire trajectory length
      color_map = [
          (np.array(track_cmap(t / max(1, traj_len - 1)))[:3] * 255).astype(int).tolist()
          for t in range(traj_len)
      ]

      for s in range(traj_len):
        overlay = frame_sequence[s].copy()

        # Group line segments by color for batch drawing
        color_to_lines = defaultdict(list)
        color_to_points = defaultdict(list)

        for traj_idx in range(traj_set.shape[1]):
            # Calculate the start index based on track_history
          if track_history is not None:
            start_idx = max(0, s + 1 - track_history)
          else:
            start_idx = 0

          traj = traj_set[start_idx:s + 1, traj_idx].numpy().astype(np.int32)

          # Group line segments by their temporal color
          for t in range(len(traj) - 1):
            color_tuple = tuple(color_map[t])  # Convert color to tuple for dict key
            line = np.array([traj[t:t+2]], dtype=np.int32)  # Shape: [1, 2, 2]
            color_to_lines[color_tuple].append(line)

          # Group points by their temporal color for dots
          if show_dots:
            for t, point in enumerate(traj):
              color_tuple = tuple(color_map[t])
              color_to_points[color_tuple].append(tuple(point))

        # Batch draw lines of the same color
        for color, lines in color_to_lines.items():
          if lines:  # Check if there are any lines for this color
            batch_lines = np.concatenate(lines, axis=0)  # Shape: [N, 2, 2]
            cv2.polylines(overlay, batch_lines, isClosed=False,
                          color=color, thickness=linewidth,
                          lineType=cv2.LINE_AA)

        # Batch draw dots of the same color
        if show_dots:
          for color, points in color_to_points.items():
            if points:  # Check if there are any points for this color
              for point in points:
                cv2.circle(overlay, point, linewidth, color, -1)

        cv2.addWeighted(overlay, transparency, frame_sequence[s], 1 - transparency, 0, frame_sequence[s])

    if title:
      # Add title to frames
      for s in range(traj_len):
        title_img = np.zeros((title_height, w, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(
            title_img, title, (text_x, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            title_color,
            1,
            cv2.LINE_AA
        )
        frame_sequence[s] = np.vstack([title_img, frame_sequence[s]])

    all_frames.append(frame_sequence)

  flattened_frames = [frame for batch_frames in all_frames for frame in batch_frames]

  if output_filename:
    if output_filename.endswith(".gif"):
      save_frames_as_gif(flattened_frames, output_filename, fps)
    elif output_filename.endswith(".mp4"):
      height, width, _ = flattened_frames[0].shape
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
      for frame in flattened_frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
      video_writer.release()
    else:
      raise ValueError("Output filename must end with .gif or .mp4")

  return flattened_frames


def animate_tracks_on_rgbs(
    tracks: np.array,
    frames: np.array,
    output_filename: Optional[str] = None,
    fps: int = 10,
    show_dots: bool = False,
    title: str = "RGB + Tracks",
    title_color: Tuple[int, int, int] = (255, 255, 255),
    transparency: float = 0.5
):
  """
    Visualize trajectories overlaid on RGB reference frames.

    This function creates an animation that visualizes the point trajectories
    (tracks) on the corresponding RGB frames. The trajectories are overlaid
    on the frames with adjustable transparency and optional dots indicating
    individual points. The animation can be displayed or saved as a file.

    Args:
        tracks (torch.Tensor): A tensor of shape [B, N, T, 2] containing
            the trajectories of points.
            - B: Batch size (number of sequences).
            - N: Number of points being tracked.
            - T: Number of timesteps in the trajectory.
            - 2: (x, y) coordinates of the points.
        reference_frames (np.ndarray): A numpy array of shape [B, H, W, 3]
            containing the RGB frames for visualization.
            - B: Batch size (number of sequences).
            - T: Number of timesteps in the trajectory.
            - H: Height of the frames.
            - W: Width of the frames.
            - 3: RGB color channels.
        output_filename (Optional[str]): Path to save the output animation.
            The format is determined by the file extension (e.g., '.gif' or '.mp4').
            If None, the animation will only be displayed and not saved.
        fps (int): Frames per second for the animation. Determines the playback speed.
        show_dots (bool): If True, individual points along the trajectories are marked with dots.
        title (str): Title displayed at the top of the visualization.
        title_color (Tuple[int, int, int]): RGB color of the title text.
        transparency (float): Value between 0 and 1 indicating the transparency
            level of the trajectory overlays.
            - 0: Fully transparent (invisible).
            - 1: Fully opaque (no background visible).
  """
  b, t, h, w, c = frames.shape
  assert c == 3, "Reference frames must have 3 color channels (RGB)."
  assert tracks.ndim == 4 and tracks.shape[-1] == 2, "Tracks must be [B, N, T, 2]."

  tracks = rearrange(tracks, "b n t d -> b t n d")
  color_map = matplotlib.colormaps.get_cmap("cool")
  linewidth = max(int(5 * h / 512), 1)
  title_height = 40
  all_frames = []

  for traj_set, video in zip(tracks, frames):
    traj_len = traj_set.shape[0]
    frame_sequence = []

    for s in range(traj_len):
      # Create RGB frame with transparent gradient-colored trajectories
      img_copy = video[s].copy().astype(np.uint8)
      overlay = img_copy.copy()
      for traj_idx in range(traj_set.shape[1]):
        traj = traj_set[:, traj_idx]
        for step in range(s):
          color_rgba = color_map(step / max(1, traj_len - 2))
          color = np.array(color_rgba[:3]) * 255

          cv2.line(
              overlay,
              pt1=(int(traj[step, 0]), int(traj[step, 1])),
              pt2=(int(traj[step + 1, 0]), int(traj[step + 1, 1])),
              color=color,
              thickness=linewidth,
              lineType=cv2.LINE_AA
          )
          if show_dots:
            cv2.circle(
                overlay,
                (int(traj[step, 0]), int(traj[step, 1])),
                linewidth,
                color,
                -1
            )
      # Blend overlay with original image
      cv2.addWeighted(overlay, transparency, img_copy, 1 - transparency, 0, img_copy)
      if title:
        # Add title above the image
        title_img = np.zeros((title_height, w, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(
            title_img, title, (text_x, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            title_color,
            1,
            cv2.LINE_AA
        )
        # Combine title and image
        img_combined = np.vstack([title_img, img_copy])
        frame_sequence.append(img_combined)

    all_frames.append(frame_sequence)

  # Flatten frames across all batches
  flattened_frames = [frame for batch_frames in all_frames for frame in batch_frames]

  if output_filename:
    if output_filename.endswith(".gif"):
      save_frames_as_gif(flattened_frames, output_filename, fps)
    elif output_filename.endswith(".mp4"):
      height, width, _ = flattened_frames[0].shape
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
      for frame in flattened_frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
      video_writer.release()
    else:
      raise ValueError("Output filename must end with .gif or .mp4")

  return flattened_frames


def save_frames_as_gif(
    frames: List[np.ndarray],
    output_filename="output.gif",
    fps=10,
    loop=1
):
  imageio.mimsave(output_filename, frames, fps=fps, loop=loop)


def get_k_nearest_pixels(
    point: torch.Tensor,
    k: int,
    H: int,
    W: int
) -> torch.Tensor:
  """
  Given a point, find the coordinates of its k-nearest pixels in the segmentation mask.

  Args:
      point (torch.Tensor): The float coordinates of the point (2,).
      k (int): The number of nearest pixels to consider.
      H (int): Height of the segmentation mask.
      W (int): Width of the segmentation mask.

  Returns:
      torch.Tensor: Coordinates of the k-nearest pixels (k, 2).
  """
  # Create a grid of pixel indices around the point
  x_min, x_max = max(0, int(point[0]) - k), min(W, int(point[0]) + k + 1)
  y_min, y_max = max(0, int(point[1]) - k), min(H, int(point[1]) + k + 1)

  # Generate grid of pixel coordinates
  xs, ys = torch.meshgrid(
      torch.arange(x_min, x_max),
      torch.arange(y_min, y_max),
      indexing="xy"
  )

  # Flatten the grid into a list of pixel coordinates
  pixel_coords = torch.stack((xs.flatten(), ys.flatten()), dim=1)  # Shape: (num_pixels, 2)

  # Calculate squared distances from the point to each pixel
  distances = torch.sum((pixel_coords - point) ** 2, dim=1)  # Shape: (num_pixels,)

  # Get indices of the k-nearest pixels
  nearest_indices = torch.topk(-distances, k=k, largest=True).indices
  return pixel_coords[nearest_indices]  # Shape: (k, 2)


def group_points_by_segmentation(
    points: torch.Tensor,
    segmentation_mask: torch.Tensor,
    k: int
) -> Tuple[Dict[int, List[List[float]]], Dict[int, int]]:
  """
  Group points by their k-nearest segmentation mask labels.

  Args:
      points (torch.Tensor): Float coordinates of the points (N, 2).
      segmentation_mask (torch.Tensor): Segmentation mask of shape (H, W).
      k (int): Number of nearest pixels to consider.

  Returns:
      Tuple[Dict[int, List[List[float]]], Dict[int, int]]:
          - A dictionary mapping segmentation labels to lists of point indices and coordinates.
          - A dictionary mapping point indices to the most common segmentation label.
  """
  H, W = segmentation_mask.shape
  grouped_points = {}  # Dictionary to hold points grouped by label
  point_to_index_mapping = {}  # Map each point to its segmentation labels

  for i, point in enumerate(points):
    # Get k-nearest pixel coordinates
    nearest_pixels = get_k_nearest_pixels(point, k, H, W)

    # Retrieve segmentation labels of these pixels
    nearest_labels = segmentation_mask[nearest_pixels[:, 1], nearest_pixels[:, 0]]

    # Update grouped points and point-to-index mapping
    for label in nearest_labels:
      label_item = label.item()
      if label_item not in grouped_points:
        grouped_points[label_item] = []
      grouped_points[label_item].append(point.tolist())

      if i not in point_to_index_mapping:
        point_to_index_mapping[i] = []
      point_to_index_mapping[i].append(label_item)

  return grouped_points, {
      key: Counter(value).most_common(1)[0][0]
      for (key, value) in point_to_index_mapping.items()
  }


def group_trajectories(
    trajectories: torch.Tensor,
    index: List[Tuple[int, int]]
) -> Dict[int, torch.Tensor]:
  """
  Group trajectories by their corresponding indices.

  Args:
      trajectories (torch.Tensor): Trajectory data (N, T, 2) where N is the number of points and T is the number of
      time steps.
      index (List[Tuple[int, int]]): A list of tuples where each tuple contains a point index and a group index.

  Returns:
      Dict[int, torch.Tensor]: A dictionary mapping group indices to the stacked trajectories of the corresponding
      points.
  """
  grouped_trajectories = {}
  for (point_idx, group_idx) in index:
    if group_idx not in grouped_trajectories:
      grouped_trajectories[group_idx] = []
    grouped_trajectories[group_idx].append(
        trajectories[[point_idx], ...]
    )
  return {
      group_idx: torch.vstack(tracks)
      for (group_idx, tracks) in grouped_trajectories.items()
  }


def animate_trajectories(
    trajectories: torch.Tensor,
    reference_frames: torch.Tensor,
    initial_frame_segmentations: torch.Tensor,
    title="",
    title_color: Tuple[int, int, int] = (255, 255, 255),
    different_color_per_object: bool = False,
    point_to_index_mapping: Dict[int, int] = None
):
  num_segmentation_neighbours = 5
  if not point_to_index_mapping:
    _, point_to_index_mapping = group_points_by_segmentation(
        trajectories[:, 0, :],
        initial_frame_segmentations,
        num_segmentation_neighbours
    )
  grouped_trajectories = group_trajectories(
      trajectories=trajectories,
      index=point_to_index_mapping.items()
  )
  # trajectories: [n, t, 2]
  # reference_frames: [c, h, w]
  with_rgb_background = animate_tracks_on_rgb_grouped(
      [grouped_trajectories],
      rearrange(reference_frames, "c h w -> () h w c").numpy(),
      fps=10,
      show_dots=True,
      transparency=0.75,
      title=f"{title}",
      title_color=title_color,
      different_color_per_object=different_color_per_object
  )
  with_black_background = animate_tracks_on_rgb_grouped(
      [grouped_trajectories],
      np.zeros_like(rearrange(reference_frames, "c h w -> () h w c").numpy()),
      fps=10,
      show_dots=True,
      transparency=0.75,
      title=f"{title}",
      title_color=title_color,
      different_color_per_object=different_color_per_object
  )
  return {
      "rgb": with_rgb_background,
      "black": with_black_background
  }


def animate_trajectories_black_background_only(
    ground_truth_trajectories: torch.Tensor,
    predicted_trajectories: torch.Tensor,
    frames: torch.Tensor,
    segmentations: torch.Tensor,
    title: str = "",
    different_color_per_object: bool = False,
    track_history: Optional[int] = None
):
  # ground_truth_trajectories: [n, t, 2]
  # predicted_trajectories:    [n, t, 2]
  # frames:                    [t, c, h, w]
  # segmentations:             [t, h, w]
  initial_frame_segmentations = segmentations[0, ...].cpu()
  reference_frames = frames[0, ...].cpu()
  trajectories = ground_truth_trajectories

  num_segmentation_neighbours = 5
  point_to_index_mapping = None
  if not point_to_index_mapping:
    _, point_to_index_mapping = group_points_by_segmentation(
        trajectories[:, 0, :],
        initial_frame_segmentations,
        num_segmentation_neighbours
    )
  grouped_gt_trajectories = group_trajectories(
      trajectories=trajectories,
      index=point_to_index_mapping.items()
  )

  grouped_pred_traj = group_trajectories(
      predicted_trajectories,
      point_to_index_mapping.items()
  )

  all_gt = animate_tracks_on_rgb_grouped(
      [grouped_gt_trajectories],
      reference_frames=np.zeros_like(
          rearrange(reference_frames[None, ...], "b c h w -> b h w c").numpy()
      ),
      different_color_per_object=True,
      title="",
      track_history=track_history
  )
  all_gt = add_title_to_frames(
      all_gt,
      "gt (tracks)",
      extra_space=64,
      font_thickness=1,
      font_scale=0.6
  )
  all_pred = animate_tracks_on_rgb_grouped(
      [grouped_pred_traj],
      reference_frames=np.zeros_like(
          rearrange(reference_frames[None, ...], "b c h w -> b h w c").numpy()
      ),
      different_color_per_object=True,
      title="",
      track_history=track_history
  )
  all_pred = add_title_to_frames(
      all_pred,
      "pred (tracks)",
      extra_space=64,
      font_thickness=1,
      font_scale=0.6
  )

  all = concatenate_animations([all_gt, all_pred], np.hstack)
  per_object_frames = []
  for object_idx in np.unique(initial_frame_segmentations):
    if object_idx > 0:
      if different_color_per_object:
        object_color = matplotlib.colormaps.get("cool")(object_idx / (len(grouped_gt_trajectories) - 1))
        track_cmap = create_custom_colormap(object_color, f"{object_idx}")
      else:
        track_cmap = matplotlib.colormaps.get_cmap("cool")

      obj_gt_trajectories, _ = filter_trajectories(
          ground_truth_trajectories,
          initial_frame_segmentations,
          object_idx
      )
      obj_pred_trajectories, _ = filter_trajectories(
          predicted_trajectories,
          initial_frame_segmentations,
          object_idx
      )
      per_object_frames.append(
          concatenate_animations(
              [
                  add_title_to_frames(
                      frames=animate_tracks_on_rgb(
                          tracks=obj_gt_trajectories[None, ...],
                          reference_frames=np.zeros_like(
                              rearrange(reference_frames[None, ...], "b c h w -> b h w c").numpy()
                          ),
                          color_map=track_cmap,
                          transparency=1,
                          title="",
                          track_history=track_history
                      ),
                      title=f"gt (tracks) #{object_idx}",
                      extra_space=64,
                      font_thickness=1,
                      font_scale=0.6
                  ),
                  add_title_to_frames(
                      frames=animate_tracks_on_rgb(
                          tracks=obj_pred_trajectories[None, ...],
                          reference_frames=np.zeros_like(
                              rearrange(reference_frames[None, ...], "b c h w -> b h w c").numpy()
                          ),
                          color_map=track_cmap,
                          transparency=1,
                          title="",
                          track_history=track_history
                      ),
                      title=f"pred (tracks) #{object_idx}",
                      extra_space=64,
                      font_thickness=1,
                      font_scale=0.6
                  ),
              ],
              np.hstack
          )
      )

  per_object = concatenate_animations(
      per_object_frames,
      np.vstack
  )

  return concatenate_animations(
      [all, per_object],
      np.vstack
  )


def animate_trajectories_splitted_by_segmentation(
    trajectories: torch.Tensor,
    reference_frames: torch.Tensor,
    segmentations: torch.Tensor,
    point_to_index_mapping=None,
    title="",
    num_segmentation_neighbours: int = 5,
    title_color: Tuple[int, int, int] = (255, 255, 255),
    different_color_per_object: bool = False,
):
  # trajectories: [n, t, 2]
  # reference_frames: [c, h, w]
  # segmentations: [h, w]
  _, trajectory_length, _ = trajectories.shape
  if not point_to_index_mapping:
    _, point_to_index_mapping = group_points_by_segmentation(
        trajectories[:, 0, :],
        segmentations,
        num_segmentation_neighbours
    )

  grouped_trajectories = group_trajectories(
      trajectories=trajectories,
      index=point_to_index_mapping.items()
  )

  with_rgb_background_per_object = []
  with_black_background_per_object = []

  for (object_id, object_tracks) in sorted(grouped_trajectories.items()):
    if different_color_per_object:
      object_color = matplotlib.colormaps.get("cool")(object_id / (len(grouped_trajectories) - 1))
      track_cmap = create_custom_colormap(object_color, f"{object_id}")
    else:
      track_cmap = matplotlib.colormaps.get_cmap("cool")
    rgb_tracks = animate_tracks_on_rgb(
        object_tracks[None, ...],
        rearrange(reference_frames, "c h w -> () h w c").numpy(),
        fps=10,
        show_dots=True,
        transparency=0.75,
        title=f"{title} obj #{object_id}",
        title_color=title_color,
        color_map=track_cmap
    )
    black_tracks = animate_tracks_on_rgb(
        object_tracks[None, ...],
        np.zeros_like(rearrange(reference_frames, "c h w -> () h w c").numpy()),
        fps=10,
        show_dots=True,
        transparency=0.75,
        title=f"{title} obj #{object_id}",
        title_color=title_color,
        color_map=track_cmap
    )
    with_rgb_background_per_object.append(rgb_tracks)
    with_black_background_per_object.append(black_tracks)

  with_rgb_background = []
  with_black_background = []

  for t in range(trajectory_length):
    with_rgb_background.append(np.vstack([animation[t] for animation in with_rgb_background_per_object]))
    with_black_background.append(np.vstack([animation[t] for animation in with_black_background_per_object]))

  return {
      "rgb": with_rgb_background,
      "black": with_black_background,
      "index": point_to_index_mapping
  }


def concatenate_animations(
    animations: List[np.array],
    stack: Union[np.hstack, np.vstack]
) -> List[np.array]:
  trajectory_length = len(animations[0])
  frames: List[np.array] = []
  for t in range(trajectory_length):
    frames.append(stack([anim[t] for anim in animations]))
  return frames


def animate_motion_of_all_objects(
    rgb: np.array,
    grouped_trajectories: Dict[int, np.array],
    output_dir: str = ".",
    file_name: Optional[str] = None,
    title: str = 'Predicted Trajectories',
    figure_size: Tuple[int, int] = (224, 224),
    alpha=1.0,
) -> Path:
  from PIL import Image
  """Plot predicted trajectories over the input frames and save as GIF."""
  disp = []
  cmap = plt.cm.cool
  image_width, image_height = figure_size

  num_objects = max(grouped_trajectories.keys()) + 1
  colors = cmap(np.linspace(0, 1, num_objects))  # Generate colors for each object
  colors[:, -1] = alpha  # Set alpha for transparency
  figure_dpi = 100

  for i in range(rgb.shape[0]):
    fig, ax = plt.subplots(
        1, 1,
        figsize=(image_width / figure_dpi, image_height / figure_dpi),
        dpi=figure_dpi,
        facecolor='black'
    )
    ax.axis('off')

    # Plot predictions
    ax.imshow(rgb[i])
    for obj_idx, obj_tracks in sorted(grouped_trajectories.items()):
      if not obj_idx:
        continue
      obj_tracks = obj_tracks.cpu().numpy()
      valid_pred = obj_tracks[:, i, 0] > 0
      valid_pred = np.logical_and(valid_pred, obj_tracks[:, i, 0] < rgb.shape[2] - 1)
      valid_pred = np.logical_and(valid_pred, obj_tracks[:, i, 1] > 0)
      valid_pred = np.logical_and(valid_pred, obj_tracks[:, i, 1] < rgb.shape[1] - 1)
      ax.scatter(
          obj_tracks[valid_pred, i, 0] - 0.5,
          obj_tracks[valid_pred, i, 1] - 0.5,
          s=3,
          c=[colors[obj_idx]]  # Use the color with alpha
      )
    if title:
      ax.set_title(
          title,
          color='white',  # White title
          fontsize=12,
          pad=4  # Adjust padding
      )
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.canvas.draw()

    # Convert the figure canvas to an image array
    try:
      width, height = fig.get_size_inches() * fig.get_dpi()
      img = (
          np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
          .reshape(int(height), int(width), 3)
      )
    except AttributeError as e:
      width, height = fig.canvas.get_width_height()
      img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
      img = img.reshape(height, width, 4)[..., :3]
      
    disp.append(np.copy(img))
    plt.close(fig)

  frames = [Image.fromarray(img) for img in disp]

  if file_name is not None:
    gif_path = Path(output_dir) / file_name
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=100,
        loop=0
    )

  return frames


def animate_motion_per_object(
    rgb: np.array,
    grouped_trajectories: Dict[int, np.array],
    output_dir: str = ".",
    file_name: Optional[str] = None,
    figure_size: Tuple[int, int] = (224, 224),
    alpha=1.0,
    title: Optional[str] = None
) -> Path:
  """Plot predicted trajectories over the input frames and save as GIF."""

  from PIL import Image

  image_width, image_height = figure_size
  num_objects = max(grouped_trajectories.keys()) + 1
  cmap = plt.cm.cool
  colors = cmap(np.linspace(0, 1, num_objects))  # Generate colors for each object
  colors[:, -1] = alpha  # Set alpha for transparency
  figure_dpi = 100
  obj_motions = []

  for obj_idx, obj_tracks in sorted(grouped_trajectories.items()):
    if not obj_idx:
      continue
    disp = []
    obj_color = colors[obj_idx]
    obj_tracks = obj_tracks.cpu().numpy()
    for i in range(rgb.shape[0]):
      fig, ax = plt.subplots(
          1, 1,
          figsize=(image_width / figure_dpi, image_height / figure_dpi),
          dpi=figure_dpi,
          facecolor='black'
      )
      ax.axis('off')
      # Plot predictions
      ax.imshow(rgb[i])
      valid_pred = obj_tracks[:, i, 0] > 0
      valid_pred = np.logical_and(valid_pred, obj_tracks[:, i, 0] < rgb.shape[2] - 1)
      valid_pred = np.logical_and(valid_pred, obj_tracks[:, i, 1] > 0)
      valid_pred = np.logical_and(valid_pred, obj_tracks[:, i, 1] < rgb.shape[1] - 1)
      ax.scatter(
          obj_tracks[valid_pred, i, 0] - 0.5,
          obj_tracks[valid_pred, i, 1] - 0.5,
          s=3,
          c=[obj_color]  # Use the color with alpha
      )
      plt.subplots_adjust(wspace=0, hspace=0)
      fig.canvas.draw()
      width, height = fig.get_size_inches() * fig.get_dpi()
      img = (
          np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
          .reshape(int(height), int(width), 3)
      )
      disp.append(np.copy(img))
      plt.close(fig)

    obj_frames = [Image.fromarray(img) for img in disp]
    if title:
      obj_frames = add_title_to_frames(
          frames=obj_frames,
          title=f"{title} #{obj_idx}",
          extra_space=64,
          font_thickness=1,
          font_scale=0.6
      )
    obj_motions.append(obj_frames)

  obj_motions = concatenate_animations(obj_motions, np.vstack)

  frames = [Image.fromarray(img) for img in obj_motions]
  if file_name is not None:
    gif_path = Path(output_dir) / file_name
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=100,
        loop=0
    )

  return frames


def animate_gt_rgb_with_trajectories_side_by_side(
    ground_truth_trajectories: torch.Tensor,
    segmentations: torch.tensor,
    frames: torch.Tensor
):
  initial_frame_segmentations = segmentations[0, ...].cpu()
  trajectories = ground_truth_trajectories

  num_segmentation_neighbours = 5
  _, point_to_index_mapping = group_points_by_segmentation(
      trajectories[:, 0, :],
      initial_frame_segmentations,
      num_segmentation_neighbours
  )
  grouped_trajectories = group_trajectories(
      trajectories=ground_truth_trajectories,
      index=point_to_index_mapping.items()
  )
  gt_rgb = rearrange(
      frames,
      "t c h w -> t h w c"
  ).numpy()
  gt_rgb = add_title_to_frames(
      gt_rgb,
      "gt (rgb)",
      extra_space=64,
      font_thickness=1,
      font_scale=0.6
  )
  gt_rgb_with_tracks = animate_tracks_on_rgb_grouped(
      [grouped_trajectories],
      reference_frames=rearrange(frames, "t c h w -> () t h w c")[:, 0, ...].numpy(),
      different_color_per_object=True,
      title="",
      frames=rearrange(frames, "t c h w -> () t h w c").numpy(),
      track_history=6
  )
  gt_rgb_with_tracks = add_title_to_frames(
      gt_rgb_with_tracks,
      "gt (rgb + tracks)",
      extra_space=64,
      font_thickness=1,
      font_scale=0.6
  )
  frames = concatenate_animations(
      animations=[gt_rgb, gt_rgb_with_tracks],
      stack=np.hstack
  )
  return frames


def animate_motion(
    ground_truth_trajectories: torch.Tensor,
    predicted_trajectories: torch.Tensor,
    frames: torch.Tensor,
    segmentations: torch.Tensor,
    title: Optional[str] = None,
    num_segmentation_neighbours: int = 5,
    point_to_index: Optional[torch.Tensor] = None,
) -> List[Image]:
  # ground_truth_trajectories: [n, t, 2]
  # predicted_trajectories:    [k, n, t, 2]
  # frames:                    [t, c, h, w]
  # segmentations:             [t, h, w]
  num_samples, *_ = predicted_trajectories.shape
  motion_per_sample: List[List[Image]] = []
  initial_frame_segmentations = segmentations[0, ...].cpu()
  # gt
  if point_to_index is None:
    _, point_to_index = group_points_by_segmentation(
        ground_truth_trajectories[:, 0, :],
        initial_frame_segmentations,
        num_segmentation_neighbours
    )
  grouped_trajectories = group_trajectories(
      trajectories=ground_truth_trajectories,
      index=point_to_index.items()
  )
  gt_motion_rgb = animate_motion_of_all_objects(
      rearrange(frames, "t c h w -> t h w c").cpu().numpy(),
      grouped_trajectories,
      title="gt",
      alpha=0.33,
  )
  gt_motion_black = animate_motion_of_all_objects(
      (
          torch.zeros_like(
              rearrange(frames, "t c h w -> t h w c")
          )
          .cpu()
          .numpy()
      ),
      grouped_trajectories,
      title="gt",
  )
  # samples
  for sample_idx in range(num_samples):
    trajectories = predicted_trajectories[sample_idx, ...].cpu()
    grouped_trajectories = group_trajectories(
        trajectories=trajectories,
        index=point_to_index.items()
    )
    motion_per_sample.append(
        animate_motion_of_all_objects(
            torch.zeros_like(
                rearrange(frames, "t c h w -> t h w c")
            ).cpu().numpy(),
            grouped_trajectories,
            title=f"sample #{sample_idx}",
        )
    )
  pred_motion = concatenate_animations(
      motion_per_sample,
      np.hstack
  )
  # compare samples with gt
  frames = concatenate_animations(
      [gt_motion_rgb, gt_motion_black, pred_motion],
      np.hstack
  )
  if title:
    frames = add_title_to_frames(
        frames,
        title=title
    )
  return frames


def animate_motion_single_example(
    ground_truth_trajectories: torch.Tensor,
    predicted_trajectories: torch.Tensor,
    frames: torch.Tensor,
    segmentations: torch.Tensor,
    title: Optional[str] = None,
    num_segmentation_neighbours: int = 5
) -> List[Image]:
  # ground_truth_trajectories: [n, t, 2]
  # predicted_trajectories:    [n, t, 2]
  # frames:                    [t, c, h, w]
  # segmentations:             [t, h, w]
  num_samples, *_ = predicted_trajectories.shape
  initial_frame_segmentations = segmentations[0, ...].cpu()
  # gt
  _, point_to_index_mapping = group_points_by_segmentation(
      ground_truth_trajectories[:, 0, :],
      initial_frame_segmentations,
      num_segmentation_neighbours
  )
  grouped_trajectories = group_trajectories(
      trajectories=ground_truth_trajectories,
      index=point_to_index_mapping.items()
  )

  gt_per_object = animate_motion_per_object(
      torch.zeros_like(
          rearrange(frames, "t c h w -> t h w c")
      ).cpu().numpy(),
      grouped_trajectories,
      title="gt (motion)"
  )
  gt_motion_black = animate_motion_of_all_objects(
      (
          torch.zeros_like(
              rearrange(frames, "t c h w -> t h w c")
          )
          .cpu()
          .numpy()
      ),
      grouped_trajectories,
      title=None,
  )
  gt_motion_black = add_title_to_frames(
      gt_motion_black,
      "gt (motion)",
      extra_space=64,
      font_thickness=1,
      font_scale=0.6
  )
  # samples
  trajectories = predicted_trajectories.cpu()
  _, point_to_index_mapping = group_points_by_segmentation(
      trajectories[:, 0, :],
      initial_frame_segmentations,
      num_segmentation_neighbours
  )
  grouped_trajectories = group_trajectories(
      trajectories=trajectories,
      index=point_to_index_mapping.items()
  )

  pred_motion = animate_motion_of_all_objects(
      torch.zeros_like(
          rearrange(frames, "t c h w -> t h w c")
      ).cpu().numpy(),
      grouped_trajectories,
      title=None
  )
  pred_motion = add_title_to_frames(
      pred_motion,
      "pred (motion)",
      extra_space=64,
      font_thickness=1,
      font_scale=0.6
  )

  pred_per_object = animate_motion_per_object(
      torch.zeros_like(
          rearrange(frames, "t c h w -> t h w c")
      ).cpu().numpy(),
      grouped_trajectories,
      title="pred (motion)"
  )

  # compare samples with gt
  all_objects = concatenate_animations(
      [gt_motion_black, pred_motion],
      np.hstack
  )
  all_per_object = concatenate_animations([gt_per_object, pred_per_object], np.hstack)

  out = concatenate_animations(
      [all_objects, all_per_object],
      np.vstack
  )

  return out


def animate_example(
    ground_truth_trajectories: torch.Tensor,
    predicted_trajectories: torch.Tensor,
    frames: torch.Tensor,
    segmentations: torch.Tensor,
    title: str = "",
    different_color_per_object: bool = False,
    animate_per_object_trajectories: bool = True
):
  # ground_truth_trajectories: [n, t, 2]
  # predicted_trajectories:    [k, n, t, 2]
  # frames:                    [t, c, h, w]
  # segmentations:             [t, h, w]
  initial_frame_segmentations = segmentations[0, ...].cpu()
  reference_frames = frames[0, ...].cpu()
  all_objects_gt = animate_trajectories(
      trajectories=ground_truth_trajectories,
      reference_frames=reference_frames,
      initial_frame_segmentations=initial_frame_segmentations,
      title="gt all",
      different_color_per_object=different_color_per_object
  )
  per_object_gt_result = animate_trajectories_splitted_by_segmentation(
      ground_truth_trajectories,
      reference_frames,
      initial_frame_segmentations,
      title="gt",
      different_color_per_object=different_color_per_object
  )
  point_to_index = per_object_gt_result["index"]
  all_objects_plot = {
      "rgb": [all_objects_gt["rgb"]],
      "black": [all_objects_gt["black"]]
  }
  num_samples, *_ = predicted_trajectories.shape
  cmap = matplotlib.colormaps.get("cool")
  per_objects_plot = {
      "rgb": [],
      "black": []
  }
  if animate_per_object_trajectories:
    for sample_idx in range(num_samples):
      pred_all_objects_plot = animate_trajectories(
          predicted_trajectories[sample_idx, ...].cpu(),
          reference_frames,
          initial_frame_segmentations,
          title="pred all",
          title_color=np.round(
              np.array(cmap((sample_idx)/num_samples)[:3]) * 255
          ).astype(np.int32).tolist(),
          different_color_per_object=different_color_per_object
      )
      all_objects_plot["rgb"].append(pred_all_objects_plot["rgb"])
      all_objects_plot["black"].append(pred_all_objects_plot["black"])

      pred = animate_trajectories_splitted_by_segmentation(
          predicted_trajectories[sample_idx, ...].cpu(),
          reference_frames,
          initial_frame_segmentations,
          point_to_index_mapping=point_to_index,
          title="pred",
          title_color=np.round(
              np.array(cmap((sample_idx)/num_samples)[:3]) * 255
          ).astype(np.int32).tolist(),
          different_color_per_object=different_color_per_object
      )
      per_objects_plot["rgb"].append(pred["rgb"])
      per_objects_plot["black"].append(pred["black"])
  else:
    for sample_idx in range(num_samples):
      pred_all_objects_plot = animate_trajectories(
          predicted_trajectories[sample_idx, ...].cpu(),
          reference_frames,
          initial_frame_segmentations,
          title="pred all",
          title_color=np.round(
              np.array(cmap((sample_idx)/num_samples)[:3]) * 255
          ).astype(np.int32).tolist(),
          different_color_per_object=different_color_per_object
      )
      all_objects_plot["rgb"].append(pred_all_objects_plot["rgb"])
      all_objects_plot["black"].append(pred_all_objects_plot["black"])

  all_objects_plot["rgb"] = concatenate_animations(
      all_objects_plot["rgb"],
      np.hstack
  )
  all_objects_plot["black"] = concatenate_animations(
      all_objects_plot["black"],
      np.hstack
  )
  if animate_per_object_trajectories:
    per_objects_plot["rgb"] = concatenate_animations(
        per_objects_plot["rgb"],
        np.hstack
    )
    per_objects_plot["black"] = concatenate_animations(
        per_objects_plot["black"],
        np.hstack
    )
    per_objects_plot["rgb"] = concatenate_animations(
        [per_object_gt_result["rgb"], per_objects_plot["rgb"]],
        np.hstack
    )
    per_objects_plot["black"] = concatenate_animations(
        [per_object_gt_result["black"], per_objects_plot["black"]],
        np.hstack
    )
    result_plot = {
        "rgb": concatenate_animations(
            [all_objects_plot["rgb"], per_objects_plot["rgb"]],
            np.vstack
        ),
        "black": concatenate_animations(
            [all_objects_plot["black"], per_objects_plot["black"]],
            np.vstack
        )
    }
  else:
    result_plot = all_objects_plot
  result_plot["rgb"] = add_title_to_frames(
      result_plot["rgb"],
      title=title
  )
  result_plot["black"] = add_title_to_frames(
      result_plot["black"],
      title=title
  )
  return result_plot

# RGB WARPING ANIMATION #


def create_gaussian_splat(
    center: Tuple[int, int],
    size: Tuple[int, int] = (11, 11),
    sigma: float = 2.0
) -> np.ndarray:
  """Create a 2D Gaussian kernel centered at a specific point."""
  x = np.linspace(0, size[0] - 1, size[0])
  y = np.linspace(0, size[1] - 1, size[1])
  x, y = np.meshgrid(x, y)
  # Offset coordinates relative to the center
  x = x - center[0]
  y = y - center[1]
  # Create 2D Gaussian distribution
  pos = np.dstack((x, y))
  rv = multivariate_normal([0, 0], [[sigma**2, 0], [0, sigma**2]])
  return rv.pdf(pos)


def get_trajectory_labels(
    trajectories: np.ndarray,
    segmentation_mask: np.ndarray
) -> np.ndarray:
  """
  Get object labels for each trajectory using nearest neighbor interpolation.

  Args:
      trajectories: Array of shape (n, t, 2) for n points and t frames
      segmentation_mask: Array of shape (h, w, 1) with object labels (0 is background)

  Returns:
      trajectory_labels: Array of shape (n,) with object label for each trajectory
  """
  _, point_to_index_mapping = group_points_by_segmentation(
      trajectories[:, 0, :],
      segmentation_mask,
      5
  )
  return np.array(list(point_to_index_mapping.values()))


def filter_trajectories(
    trajectories: np.ndarray,
    segmentation_mask: np.ndarray,
    object_idx: Optional[int] = 0
) -> Tuple[np.ndarray, np.ndarray]:
  """Filter trajectories based on segmentation mask."""
  trajectory_labels = get_trajectory_labels(trajectories, segmentation_mask)
  if object_idx is not None:
    foreground_mask = trajectory_labels == object_idx
  else:
    foreground_mask = trajectory_labels > 0
  filtered_trajectories = trajectories[foreground_mask]
  return filtered_trajectories, foreground_mask


def sample_colors_from_reference(
    ref_frame: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
  """Sample RGB colors from reference frame at given points."""
  colors = []
  h, w, c = ref_frame.shape
  for point in points:
    x, y = int(point[0]), int(point[1])
    x = np.clip(x, 0, w-1)
    y = np.clip(y, 0, h-1)
    color = ref_frame[y, x].astype(np.float32)
    colors.append(color)
  return np.array(colors)


def render_frame(
    points: np.ndarray,
    colors: np.ndarray,
    frame_shape: Tuple[int, int],
    splat_size: Tuple[int, int] = (5, 5),
    sigma: float = 0.75
) -> np.ndarray:
  """Render a single frame with Gaussian splats."""
  h, w = frame_shape[:2]
  color_accumulation = np.zeros((h, w, 3), dtype=np.float32)
  weight_accumulation = np.zeros((h, w), dtype=np.float32)
  kernel = create_gaussian_splat((splat_size[0], splat_size[1]), sigma=sigma)
  kernel = kernel / kernel.max()
  half_size = (splat_size[0] // 2, splat_size[1] // 2)

  for point, color in zip(points, colors):
    x, y = int(point[0]), int(point[1])
    x_start = max(0, x - half_size[0])
    x_end = min(w, x + half_size[0] + 1)
    y_start = max(0, y - half_size[1])
    y_end = min(h, y + half_size[1] + 1)
    k_x_start = max(0, half_size[0] - x)
    k_x_end = k_x_start + (x_end - x_start)
    k_y_start = max(0, half_size[1] - y)
    k_y_end = k_y_start + (y_end - y_start)
    if k_x_end <= k_x_start or k_y_end <= k_y_start:
      continue
    kernel_section = kernel[k_y_start:k_y_end, k_x_start:k_x_end]
    for c in range(3):
      color_accumulation[y_start:y_end, x_start:x_end, c] += kernel_section * color[c]
    weight_accumulation[y_start:y_end, x_start:x_end] += kernel_section

  weight_accumulation = np.maximum(weight_accumulation, 1e-10)
  for c in range(3):
    color_accumulation[:, :, c] /= weight_accumulation
  return np.clip(color_accumulation, 0, 255).astype(np.uint8)


def animate_gaussian_splats(
    ref_frame: np.ndarray,
    trajectories: np.ndarray,
    segmentation_mask: np.ndarray,
    object_idx: Optional[int] = None,
    splat_size: Tuple[int, int] = (5, 5),
    sigma: float = 0.75
) -> List[np.ndarray]:
  """Animate Gaussian splats following trajectories."""
  filtered_trajectories, _ = filter_trajectories(trajectories, segmentation_mask, object_idx)
  if len(filtered_trajectories) == 0:
    return []
  _, n_frames, _ = filtered_trajectories.shape
  h, w, _ = ref_frame.shape
  initial_points = filtered_trajectories[:, 0]
  colors = sample_colors_from_reference(ref_frame, initial_points)
  frames = []
  for t in range(n_frames):
    current_points = filtered_trajectories[:, t]
    frame = render_frame(current_points, colors, (h, w), splat_size, sigma)
    frames.append(frame)
  return frames


def animate_rgb_by_warping(
    ground_truth_trajectories: torch.Tensor,
    predicted_trajectories: torch.Tensor,
    frames: torch.Tensor,
    segmentations: torch.Tensor,
    title: str = "",
    splat_size: Tuple[int, int] = (5, 5),
    sigma: float = 0.75
):
  # ground_truth_trajectories: [n, t, 2]
  # predicted_trajectories:    [n, t, 2]
  # frames:                    [t, c, h, w]
  # segmentations:             [t, h, w]
  ref_frame = (
      rearrange(frames[0, ...], "c h w -> h w c")
      .detach()
      .cpu()
      .to(torch.uint8)
      .numpy()
  )
  initial_frame_seg_mask = segmentations[0]

  # Animate all
  all_gt_frames = animate_gaussian_splats(
      ref_frame,
      ground_truth_trajectories,
      segmentation_mask=initial_frame_seg_mask,
      splat_size=splat_size,
      sigma=sigma
  )
  all_gt_frames = add_title_to_frames(
      all_gt_frames,
      "gt (warp)",
      extra_space=64,
      font_thickness=1,
      font_scale=0.6
  )
  sample_frames = None
  per_object_frames = []

  all_pred_frames = animate_gaussian_splats(
      ref_frame,
      predicted_trajectories,
      segmentation_mask=initial_frame_seg_mask,
      splat_size=splat_size,
      sigma=sigma
  )
  all_pred_frames = add_title_to_frames(
      all_pred_frames,
      "pred (warp)",
      extra_space=64,
      font_thickness=1,
      font_scale=0.6
  )

  for object_idx in np.unique(initial_frame_seg_mask):
    if object_idx > 0:
      obj_gt_frames = animate_gaussian_splats(
          ref_frame,
          ground_truth_trajectories,
          segmentation_mask=initial_frame_seg_mask,
          object_idx=object_idx,
          splat_size=splat_size,
          sigma=sigma
      )
      obj_gt_frames = add_title_to_frames(
          obj_gt_frames,
          f"gt (warp) #{object_idx}",
          extra_space=64,
          font_thickness=1,
          font_scale=0.6
      )
      obj_pred_frames = animate_gaussian_splats(
          ref_frame,
          predicted_trajectories,
          segmentation_mask=initial_frame_seg_mask,
          object_idx=object_idx,
          splat_size=splat_size,
          sigma=sigma
      )
      obj_pred_frames = add_title_to_frames(
          obj_pred_frames,
          f"pred (warp) #{object_idx}",
          extra_space=64,
          font_thickness=1,
          font_scale=0.6
      )
      per_object_frames.append(
          concatenate_animations(
              [obj_gt_frames, obj_pred_frames],
              np.hstack
          )
      )
  all_frames = concatenate_animations(
      [all_gt_frames, all_pred_frames],
      np.hstack
  )
  per_object_frames = concatenate_animations(
      per_object_frames,
      np.vstack
  )
  sample_frames = concatenate_animations(
      [all_frames, per_object_frames],
      np.vstack
  )
  return sample_frames
