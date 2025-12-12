
from collections import namedtuple
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from einops import rearrange, repeat
from omegaconf import DictConfig

from grid_sampler import sample
from preprocessing import PreprocessingConfig, normalize_points
from utils.to_torch import to_torch

Example = namedtuple("Example", [
    "frames",
    "trajectories",
    "visibility",
    "segmentations",
    "features",
    "index"
])

Batch = namedtuple("Batch", [
    "frames",
    "trajectories",
    "segmentations",
    "visibility",
    "index"
])


def sample_moving_and_static_trajectories(
    example: Example,
    num_trajectories: int,
    rng: torch.Generator,
    moving_ratio: float = 0.9,
    displacement_var_threshold: float = 0,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
    tolerance: Optional[int] = None
):
  deltas = torch.diff(example.trajectories, dim=1)  # Shape: (n, t-1, 2)
  displacements = torch.norm(deltas, dim=-1)  # Shape: (n, t-1)
  displacement_var = torch.var(displacements, dim=-1)  # Shape: (n,)

  moving_mask = displacement_var > displacement_var_threshold
  static_mask = ~moving_mask

  total_num_points, *_ = deltas.shape
  total_moving_points = moving_mask.sum().item()
  total_static_points = total_num_points - total_moving_points

  num_moving_points = min(
      moving_mask.sum().item(),
      int(num_trajectories * moving_ratio)
  )
  num_static_points = num_trajectories - num_moving_points

  if not total_moving_points or not total_static_points:
    # Sample anything using sample forward
    return sample_forward_trajectories(
        example,
        num_trajectories,
        rng,
        max_width=max_width,
        max_height=max_height,
        tolerance=tolerance
    )

  else:
    # Sample moving
    moving_points_batch = sample_forward_trajectories(
        Example(
            frames=example.frames,
            trajectories=example.trajectories[moving_mask],
            visibility=example.visibility[moving_mask],
            segmentations=example.segmentations,
            features=example.features,
            index=example.index
        ),
        num_trajectories=num_moving_points,
        rng=rng,
        max_width=max_width,
        max_height=max_height,
        tolerance=tolerance
    )
    # Sample static
    static_points_batch = sample_forward_trajectories(
        Example(
            frames=example.frames,
            trajectories=example.trajectories[static_mask],
            visibility=example.visibility[static_mask],
            segmentations=example.segmentations,
            features=example.features,
            index=example.index
        ),
        num_trajectories=num_static_points,
        rng=rng,
        max_width=max_width,
        max_height=max_height,
        tolerance=tolerance
    )
    # Combine
    return Batch(
        example.frames,
        torch.cat(
            [moving_points_batch.trajectories, static_points_batch.trajectories],
            dim=0
        ),
        torch.cat(
            [moving_points_batch.trajectory_masks, static_points_batch.trajectory_masks],
            dim=0
        ),
        torch.cat(
            [moving_points_batch.visibility, static_points_batch.visibility],
            dim=0
        ),
        moving_points_batch.segmentations,
        example.features,
        moving_points_batch.index,
    )


def map_trajectories_to_objects(
    segmentation_mask: torch.Tensor,
    trajectories: torch.Tensor
) -> torch.Tensor:
  n, *_ = trajectories.shape
  h, w = segmentation_mask.shape
  # Normalize coordinates to be in the range [-1, 1] for grid_sample
  normalized_query_points = normalize_points(
      trajectories[:, 0, :],  # query points
      max_width=w,
      max_height=h
  )
  normalized_query_points = rearrange(
      normalized_query_points,
      "n d -> () n d"
  )
  mask = repeat(
      segmentation_mask.float(),
      "h w -> () n () h w",
      n=n
  )
  segment_ids = (
      sample(
          features=mask,
          coords=normalized_query_points,
          mode="nearest",
          align_corners=True
      )
      .long()
      .flatten()
  )
  return segment_ids


def mask_within_camera(
    trajectories: torch.Tensor,
    max_width: int,
    max_height: int,
    tolerance: int = 32,
    epsilon: float = 1e-6
):
  within_camera_mask = (
      (trajectories[:, :, 0] >= -tolerance - epsilon) &
      (trajectories[:, :, 0] <= max_width + tolerance + epsilon) &
      (trajectories[:, :, 1] >= -tolerance - epsilon) &
      (trajectories[:, :, 1] <= max_height + tolerance + epsilon)
  )
  return repeat(
      within_camera_mask.to(trajectories.device),
      "n t -> n t d",
      d=2
  ).contiguous()


def sample_equally_from_provided_objects(
    example: Example,
    num_trajectories: int,
    trajectory2object: torch.Tensor,
    rng: torch.Generator,
    allowed_objects: Union[List[int], torch.Tensor],
    max_height: int,
    max_width: int,
    tolerance: int = 32,
    epsilon: float = 1e-6,
    return_info: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[int, int]]]:
  """
  Sample trajectories ensuring the most equal distribution possible across allowed objects.
  Ensures no duplicate trajectories are sampled even when trajectories belong to multiple objects.

  Args:
      example: Example containing trajectories and other data
      num_trajectories: Total number of trajectories to sample
      trajectory2object: Tensor of shape (n) containing object IDs for each trajectory
      rng: Random number generator
      allowed_objects: List or tensor of object IDs that are allowed for sampling
      max_height, max_width: Maximum dimensions for masking
      tolerance: Tolerance parameter for masking
      epsilon: Small value to avoid numerical issues
      return_info: If True, returns additional information about number of trajectories sampled per object

  Returns:
      If return_info is False: Batch with sampled trajectories
      If return_info is True: Tuple of (Batch, object_count_dict)
  """
  # Convert allowed_objects to tensor if needed
  if not isinstance(allowed_objects, torch.Tensor):
    allowed_objects = torch.tensor(allowed_objects)

  # Ensure allowed_objects is not empty
  if len(allowed_objects) == 0:
    raise ValueError("No allowed objects provided")

  # Set to track already sampled trajectory indices
  already_sampled_indices = set()

  # Dictionary to store indices per object and counts
  object_indices = {}
  object_counts = {}

  # Count available trajectories per object (excluding duplicates across objects)
  available_trajectories_per_object = {}
  for obj_id in allowed_objects:
    obj_id_item = obj_id.item()
    # Get indices of trajectories for this object
    obj_mask = trajectory2object == obj_id
    obj_trajectory_indices = torch.nonzero(obj_mask, as_tuple=True)[0]
    available_trajectories_per_object[obj_id_item] = len(obj_trajectory_indices)

  # Sort objects by number of available trajectories (ascending)
  # This prioritizes objects with fewer trajectories to ensure they get their share
  sorted_objects = sorted(
      [(obj.item(), available_trajectories_per_object[obj.item()]) for obj in allowed_objects],
      key=lambda x: x[1]
  )

  # First pass: calculate how many trajectories we can sample per object
  remaining_trajectories = num_trajectories
  samples_per_object = {}

  # Start with minimum allocation (1 per object)
  for obj_id, _ in sorted_objects:
    samples_per_object[obj_id] = 1
    remaining_trajectories -= 1

  # Distribute remaining trajectories proportionally
  while remaining_trajectories > 0:
    for obj_id, avail_count in sorted_objects:
      # Skip if this object doesn't have any more available trajectories
      if samples_per_object[obj_id] >= avail_count:
        continue

      # Allocate one more trajectory to this object
      samples_per_object[obj_id] += 1
      remaining_trajectories -= 1

      # Break if no more trajectories to allocate
      if remaining_trajectories == 0:
        break

  # Now sample trajectories for each object
  total_sampled = 0
  for obj_id, _ in sorted_objects:
    samples_needed = samples_per_object[obj_id]
    if samples_needed == 0:
      continue

    # Get indices of trajectories for this object
    obj_mask = trajectory2object == obj_id
    obj_trajectory_indices = torch.nonzero(obj_mask, as_tuple=True)[0]

    # Filter out indices that have already been sampled
    valid_indices = [idx.item() for idx in obj_trajectory_indices if idx.item() not in already_sampled_indices]
    valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)

    # Check if we have enough trajectories
    if len(valid_indices_tensor) < samples_needed:
      # If we don't have enough trajectories, take what we can
      samples_to_take = len(valid_indices_tensor)
    else:
      samples_to_take = samples_needed

    # Sample trajectories for this object
    if samples_to_take > 0:
      perm = torch.randperm(len(valid_indices_tensor), generator=rng)
      sampled_indices = valid_indices_tensor[perm[:samples_to_take]]

      # Update our tracking sets
      for idx in sampled_indices.tolist():
        already_sampled_indices.add(idx)

      object_indices[obj_id] = sampled_indices
      object_counts[obj_id] = samples_to_take
      total_sampled += samples_to_take

  # Combine all sampled indices
  if len(object_indices) > 0:
    all_sampled_indices = torch.cat(list(object_indices.values()))
  else:
    # Handle the case where no trajectories were sampled
    raise ValueError("No trajectories could be sampled with the given constraints")

  # Double-check that we don't have duplicates
  assert len(all_sampled_indices) == len(torch.unique(all_sampled_indices)), "Duplicate indices detected"

  if len(all_sampled_indices) != num_trajectories:
    raise ValueError("No trajectories could be sampled with the given constraints")

  # Gather the sampled trajectories
  sampled_trajectories = example.trajectories[all_sampled_indices]
  sampled_visibility = example.visibility[all_sampled_indices]

  # Create mask for not denoising the query point
  masks = mask_within_camera(
      sampled_trajectories,
      max_width=max_width,
      max_height=max_height,
      tolerance=tolerance,
      epsilon=epsilon
  )

  # We are not denoising the query point
  masks[:, 0, :] = 0.0

  result = Batch(
      example.frames,
      sampled_trajectories,
      masks,
      sampled_visibility,
      example.segmentations,
      example.features,
      example.index
  )

  if not return_info:
    return result
  else:
    return result, object_counts


def sample_trajectories_uniformly_across_objects(
    example: Example,
    num_trajectories: int,
    rng: torch.Generator,
    max_height: int,
    max_width: int,
    ignore_background: bool = False,
    tolerance: int = 32,
    epsilon: float = 1e-6,
    trajectory2object: Optional[torch.Tensor] = None
):
  if trajectory2object is None:
    trajectory2object = map_trajectories_to_objects(
        segmentation_mask=example.segmentations[0, ...],  # reference frame segmentation mask
        trajectories=example.trajectories,
    )
  unique_objects = torch.unique(trajectory2object)
  num_objects = len(unique_objects)
  # Compute the number of points to sample per object
  samples_per_object = num_trajectories // num_objects
  extra_samples = num_trajectories % num_objects
  sampled_indices: List[torch.Tensor] = []
  # Sample uniformly across objects
  for object_id in unique_objects:
    if ignore_background and object_id == 0:
      continue
    # Get indices of the current object
    object_indices = torch.nonzero(trajectory2object == object_id, as_tuple=True)[0]
    # Determine the number of samples for this object
    num_object_samples = samples_per_object + (1 if extra_samples > 0 else 0)
    extra_samples -= 1
    # Sample indices uniformly at random (without replacement)
    if len(object_indices) > num_object_samples:
      selected_indices = object_indices[
          torch.multinomial(
              input=torch.ones(
                  len(object_indices),
                  device=object_indices.device
              ),  # uniform weights
              num_samples=num_object_samples,
              replacement=False,
              generator=rng
          )
      ]
    else:
      # If fewer points are available than required, take all points
      selected_indices = object_indices
    sampled_indices.append(selected_indices)
  # Concatenate all sampled indices
  sampled_indices = torch.cat(sampled_indices, dim=0)
  # If insufficient points were sampled, sample the remaining points uniformly from all trajectories
  missing_samples = num_trajectories - len(sampled_indices)
  while missing_samples > 0:
    remaining_indices = torch.arange(
        example.trajectories.size(0),
        device=example.trajectories.device
    )  # [n, ]
    remaining_indices = remaining_indices[
        ~torch.isin(remaining_indices, sampled_indices)
    ]  # Exclude already sampled indices
    if len(remaining_indices) > 0:
      # If there are unsampled points remain, resample uniformly from the these
      additional_indices = remaining_indices[
          torch.multinomial(
              input=torch.ones(
                  len(remaining_indices),
                  device=example.trajectories.device
              ),  # uniform weights
              num_samples=missing_samples,
              replacement=False
          )
      ]
    else:
      # If no unsampled points remain, resample uniformly from the full dataset
      additional_indices = torch.multinomial(
          input=torch.ones(
              example.trajectories.size(0),
              device=example.trajectories.device
          ),  # uniform weights
          num_samples=missing_samples,
          replacement=False
      )
    sampled_indices = torch.cat([sampled_indices, additional_indices], dim=0)
    missing_samples = num_trajectories - len(sampled_indices)
  # Gather the sampled trajectories
  sampled_trajectories = example.trajectories[sampled_indices]
  sampled_visibility = example.visibility[sampled_indices]
  # Create mask for not denoising the query point
  masks = mask_within_camera(
      sampled_trajectories,
      max_width=max_width,
      max_height=max_height,
      tolerance=tolerance,
      epsilon=epsilon
  )
  # We are not denoising the query point
  masks[:, 0, :] = 0.0
  return Batch(
      example.frames,
      sampled_trajectories,
      masks,
      sampled_visibility,
      example.segmentations,
      example.features,
      example.index
  )


def take_all_trajectories(example: Example):
  # frames:        [t c h w]
  # trajectories:  [n t 2]
  # visibility:    [n t]
  video = example.frames
  trajectories = example.trajectories
  visibility = example.visibility
  # Create mask for not denoising the query point
  masks = torch.ones_like(trajectories)
  masks[:, 0, :] = 0.0
  return Batch(
      video,
      trajectories,
      masks,
      visibility,
      example.segmentations,
      example.features,
      example.index
  )


def sample_forward_trajectories(
    example: Example,
    num_trajectories: int,
    rng: torch.Generator,
    max_width: int,
    max_height: int,
    tolerance: Optional[int] = None
) -> Batch:
  # frames:        [t c h w]
  # trajectories:  [n t 2]
  # visibility:    [n t]
  video = example.frames
  trajectories = example.trajectories
  visibility = example.visibility
  segmentations = example.segmentations
  features = example.features
  num_points, *_ = trajectories.shape
  # Sample trajectory indices
  trajectory_idx = torch.randint(
      low=0,
      high=num_points,
      size=(num_trajectories, ),
      generator=rng
  )  # [N]
  sampled_trajectories = trajectories[trajectory_idx, :, :]  # [N t 2]
  sampled_visibility = visibility[trajectory_idx, :]  # [N t]
  # Create mask for not denoising the query point
  if tolerance is not None:
    masks = mask_within_camera(
        sampled_trajectories,
        max_width=max_width,
        max_height=max_height,
        tolerance=tolerance,
    )
  else:
    masks = torch.ones_like(sampled_trajectories)
  # We are not denoising the query point
  masks[:, 0, :] = 0.0
  return Batch(
      video,
      sampled_trajectories,
      masks,
      sampled_visibility,
      segmentations,
      features,
      example.index
  )


def sample_uniformly_within_tolerance(
    example: Example,
    num_trajectories: int,
    rng: torch.Generator,
    tolerance: int,
    max_width: int,
    max_height: int,
    epsilon: float = 1e-6,
) -> Batch:
  # frames:        [t c h w]
  # trajectories:  [n t 2]
  # visibility:    [n t]
  video = example.frames
  trajectories = example.trajectories
  visibility = example.visibility
  segmentations = example.segmentations
  features = example.features
  num_points, *_ = trajectories.shape
  # Sample trajectory indices
  trajectory_idx = torch.randint(
      low=0,
      high=num_points,
      size=(num_trajectories, ),
      generator=rng
  )  # [N]
  sampled_trajectories = trajectories[trajectory_idx, :, :]  # [N t 2]
  sampled_visibility = visibility[trajectory_idx, :]  # [N t]
  # Create mask for not denoising the query point
  masks = mask_within_camera(
      sampled_trajectories,
      max_width=max_width,
      max_height=max_height,
      tolerance=tolerance,
      epsilon=epsilon
  )
  # We are not denoising the query point
  masks[:, 0, :] = 0.0
  return Batch(
      video,
      sampled_trajectories,
      masks,
      sampled_visibility,
      segmentations,
      features,
      example.index
  )


def sample_backward_trajectories(
    example: Example,
    num_trajectories: int,
    rng: torch.Generator,
) -> Batch:
  # frames:        [t c h w]
  # trajectories:  [n t 2]
  # visibility:    [n t]
  return sample_forward_trajectories(
      Example(
          frames=torch.flip(example.frames, dims=[0]),
          trajectories=torch.flip(example.trajectories, dims=[1]),
          visibility=torch.flip(example.visibility, dims=[1]),
          segmentations=torch.flip(example.segmentations, dims=[0]),
          features=torch.flip(example.features, dims=[0]),
          index=example.index
      ),
      num_trajectories,
      rng
  )


def sample_trajectories(
    example: Example,
    num_trajectories: int,
    rng: torch.Generator,
    backward_probability: float = 0.5,
) -> Batch:
  # frames:        [t c h w]
  # trajectories:  [n t 2]
  # visibility:    [n t]
  p = torch.rand(1, generator=rng).item()
  if p < backward_probability:
    return sample_backward_trajectories(example, num_trajectories, rng)
  else:
    return sample_forward_trajectories(example, num_trajectories, rng)


def get_initially_visible_mask(example: Example, resolution: int):
  mask_within_reference_frame = (
      (example.trajectories[:, 0, 0] >= 0) &
      (example.trajectories[:, 0, 0] <= resolution) &
      (example.trajectories[:, 0, 1] >= 0) &
      (example.trajectories[:, 0, 1] <= resolution)
  )
  mask_visible = example.visibility[:, 0]
  return (
      mask_within_reference_frame &
      mask_visible
  )


def select_only_initially_visible_trajectories(
    example: Example,
    resolution: int
) -> Example:
  # frames:        [t c h w]
  # trajectories:  [n t 2]
  # visibility:    [n t]
  # Choose visible tracks within the reference frame
  mask = get_initially_visible_mask(example, resolution)
  return Example(
      frames=example.frames,
      trajectories=example.trajectories[mask, ...],
      visibility=example.visibility[mask, ...],
      segmentations=example.segmentations,
      features=example.features,
      index=example.index,
  )


def get_valid_trajectories_mask(example: Example) -> Example:
  past_positions = example.trajectories[:, :-1, ...]  # Shape: (B, N, T-1, D)
  future_positions = example.trajectories[:, 1:, ...]  # Shape: (B, N, T-1, D)
  eps = 1e-5
  mirrored_future_positions = -future_positions
  invalid_mask = (
      torch.linalg.norm(past_positions - mirrored_future_positions, dim=-1)
      < torch.linalg.norm(past_positions - future_positions, dim=-1) + eps
  )
  invalid_mask = invalid_mask.sum(dim=-1) > 0  # Shape: (B, N)
  valid_mask = ~invalid_mask
  return valid_mask


def select_only_valid_trajectories(example: Example) -> Example:
  valid_mask = get_valid_trajectories_mask(example)  # Shape: (B, N)

  return Example(
      frames=example.frames,
      trajectories=example.trajectories[valid_mask, ...],
      visibility=example.visibility[valid_mask, ...],
      segmentations=example.segmentations,
      features=example.features,
      index=example.index,
  )


def shift_example(
    example: Example,
    starting_frame: int = 0
) -> Example:
  if starting_frame == 0:
    return example
  else:
    # frames:        [t c h w]
    # trajectories:  [n t 2]
    # visibility:    [n t]
    return Example(
        frames=example.frames[starting_frame:, ...],
        trajectories=example.trajectories[:, starting_frame:, ...],
        visibility=example.visibility[:, starting_frame:],
        segmentations=example.segmentations,
        features=example.features,
        index=example.index
    )


def map_to_example(dict: Dict[str, torch.Tensor]) -> Example:
  t, h, w, _ = dict["frames.npy"].shape
  if hasattr(dict, "frame_features.npz"):
    features = dict["frame_features.npz"]
  else:
    features = {
        "last_hidden_state": torch.zeros((256, 1024)),
        "pooler_output": torch.zeros((1, 1024))
    }
  if "_" in dict['__key__']:
    key = int(dict['__key__'].split("_")[-1])
  else:
    key = int(dict['__key__'])
  return Example(
      torch.from_numpy(dict["frames.npy"]),
      torch.from_numpy(dict["trajectories.npy"]),
      torch.from_numpy(dict["visibility.npy"]),
      torch.from_numpy(dict["segmentations.npy"].reshape((t, h, w))),
      to_torch(features),
      torch.tensor(key)
  )


def map_to_batch(x: Any) -> Batch:
  (
      frames,
      trajectories,
      segmentations,
      visibility,
      index
  ) = x
  return Batch(
      frames,
      trajectories,
      segmentations,
      visibility,
      index
  )


class SamplingType(str, Enum):
  UNIFORM_ACROSS_OBJECTS_FIXED_LENGTH = "uniform_across_objects_fixed_length"
  UNIFORM_FIXED_LENGTH = "uniform_fixed_length"
  BALANCED_MOVING_STATIC_FIXED_LENGTH = "balanced_moving_static_fixed_length"
  MIXED_FIXED_LENGTH = "mixed_fixed_length"
  UNIFORM_VARIABLE = "uniform_variable_length"
  NONE = "none"


def filter_valid_trajectories(sample: Batch):
  (
      _,
      trajectories,
      _,
      _,
      _
  ) = sample
  # Check if all (x, y) coordinates are within [-384, 384]
  return torch.all((trajectories >= -384) & (trajectories <= 384)).item()


class KubricDataPipeline:
  def __init__(
      self,
      preprocess_cfg: PreprocessingConfig,
      cfg: DictConfig,
      mode: Literal["training", "sampling"],
      batch_rng: torch.Generator,
      apply_preprocessing: bool = True,
      use_pseudo_trajectories: bool = False
  ):
    self.preprocess_cfg: PreprocessingConfig = preprocess_cfg
    self.cfg: DictConfig = cfg
    self.mode: Literal["training", "sampling"] = mode
    self.batch_rng = batch_rng
    self.apply_preprocessing = apply_preprocessing
    self.use_pseudo_trajectories = use_pseudo_trajectories

  def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # WebDataset does not like Batch NamedTuple ...
    # "frames",
    # "trajectories",
    # "trajectory_masks",
    # "visibility",
    # "segmentations",
    # "features"
    # "index"
    # WebDataset does not like nested structure
    import torch
    import torch.nn.functional as F

    def downsample(video: torch.Tensor) -> torch.Tensor:
      """
      Downsamples a video tensor of shape (t, h, w, c) to (t, h//2, w//2, c) using bilinear interpolation.

      Args:
          video (torch.Tensor): Input tensor of shape (t, h, w, c)

      Returns:
          torch.Tensor: Resized tensor of shape (t, h//2, w//2, c)
      """
      t, h, w, c = video.shape
      # Rearrange to (t, c, h, w) for interpolation
      video_nchw = video.permute(0, 3, 1, 2)

      # Interpolate
      video_resized = F.interpolate(video_nchw, size=(h // 2, w // 2), mode='bilinear', align_corners=False)

      # Back to (t, h//2, w//2, c)
      return video_resized.permute(0, 2, 3, 1)

    if self.use_pseudo_trajectories:
      return (
          downsample(torch.from_numpy(x['frames.npy'])),
          torch.from_numpy(x['pseudo_trajectories.npy']),
          torch.from_numpy(x['segmentations.npy']),
          torch.from_numpy(x['pseudo_visibility.npy']),
          x['__key__']
      )
    else:
      return (
          downsample(torch.from_numpy(x['frames.npy'])),
          torch.from_numpy(x['trajectories.npy']),
          torch.from_numpy(x['segmentations.npy']),
          torch.from_numpy(x['visibility.npy']),
          x['__key__']
      )
