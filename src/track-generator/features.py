from typing import NamedTuple, Optional

import torch


class ImageFeatures(NamedTuple):
  pooler_output: torch.Tensor
  last_hidden_state: torch.Tensor


class TextFeatures(NamedTuple):
  text_embeds: torch.Tensor
  last_hidden_state: torch.Tensor


class Features(NamedTuple):
  frame: Optional[ImageFeatures] = None
  text: Optional[TextFeatures] = None


class FeatureSpace(NamedTuple):
  frame: Optional[int] = None
  text: Optional[int] = None
