from typing import Optional, OrderedDict

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from dino import encode_reference_frames
from einops import rearrange, repeat
from features import Features
from torchdiffeq import odeint
from transformers import AutoImageProcessor, AutoModel
from vae import encode_tracks

from nn_latte import Latte


def pixel_center_grid(h, w, device='cpu'):
    w, h = torch.meshgrid(
        torch.arange(h, dtype=torch.float, device=device) + 0.5,
        torch.arange(w, dtype=torch.float, device=device) + 0.5,
        indexing='xy'
    )
    return torch.stack([w, h], dim=-1)  # Shape (h, w, 2)
