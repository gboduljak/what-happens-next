from typing import TypedDict

import torch
from transformers import AutoImageProcessor, AutoModel

DINO_PATH = "/scratch/shared/beegfs/gabrijel/hf/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"   # noqa: E501


class DINO(TypedDict):
    pooler_output: torch.Tensor
    last_hidden_state: torch.Tensor


@torch.inference_mode()
def extract_dino_features(
    frames: torch.Tensor,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: torch.device
) -> DINO:
    batched_inputs = processor(
        images=frames,
        return_tensors="pt"
    ).to(device)
    batched_output = model(**batched_inputs)
    return batched_output


@torch.no_grad()
def encode_reference_frames(
    reference_frames: torch.Tensor,
    processor: AutoImageProcessor,
    dino: AutoModel,
    device: torch.device
):
    # Encode reference frames
    inputs = processor(
        images=reference_frames,
        size={"height": 224, "width": 224},
        return_tensors="pt"
    )
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }
    return dino(**inputs)["last_hidden_state"][:, 1:, :]  # [b, n, d]
