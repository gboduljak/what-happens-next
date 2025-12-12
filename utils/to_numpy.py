from typing import Any, Dict

import numpy as np
import torch


def to_numpy(x: torch.Tensor | Dict[Any, Any]):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return {
            key: to_numpy(value)
            for (key, value) in x.items()
        }
