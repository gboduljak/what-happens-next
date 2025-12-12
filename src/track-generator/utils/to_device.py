from typing import Any, Dict

import torch
from features import Features, ImageFeatures, TextFeatures


def to_device(x: torch.Tensor | Dict[Any, Any] | Features, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, Features):
        return Features(
            frame=(
                ImageFeatures(
                    pooler_output=to_device(x.frame.pooler_output, device),
                    last_hidden_state=to_device(
                        x.frame.last_hidden_state, device)
                )
                if x.frame is not None else None
            ),
            text=(
                TextFeatures(
                    text_embeds=to_device(x.text.text_embeds, device),
                    last_hidden_state=to_device(
                        x.text.last_hidden_state, device)
                )
                if x.text is not None else None
            )
        )
    else:
        return {
            key: to_device(value, device)
            for (key, value) in x.items()
        }
