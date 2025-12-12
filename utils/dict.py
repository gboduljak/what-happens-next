
from typing import Dict

import numpy as np


def reconstruct_nested_dict(nested_dict: Dict[str, np.array]) -> Dict[str, Dict[str, np.array]]:
    return {
        outer_key: {inner_key: nested_dict[f"{outer_key}__{inner_key}"]
                    for inner_key in {k.split("__")[1] for k in nested_dict if k.startswith(outer_key)}}
        for outer_key in {k.split("__")[0] for k in nested_dict}
    }
