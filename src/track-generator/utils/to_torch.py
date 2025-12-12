from typing import Any, Dict

import numpy as np
import torch


def to_torch(x: np.ndarray | Dict[Any, Any]):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        return {
            key: to_torch(value)
            for (key, value) in x.items()
        }
