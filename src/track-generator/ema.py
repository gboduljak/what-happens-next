from typing import List, Tuple

import torch

from nn_latte import Latte


@torch.compile(
    fullgraph=True,
    mode="reduce-overhead"
)
@torch.no_grad
def update_ema(ema_params: List[torch.Tensor], model_params: List[torch.Tensor], decay: float = 0.9999):
    """
    Torch-compilable EMA update function.
    Assumes ema_params and model_params are pre-filtered lists of trainable parameters
    in the same order.
    """
    alpha = 1.0 - decay
    for ema_param, model_param in zip(ema_params, model_params):
        ema_param.mul_(decay).add_(model_param, alpha=alpha)


def get_trainable_params(model: Latte):
    named_params = [(name, p)
                    for name, p in model.named_parameters() if p.requires_grad]
    named_params.sort(key=lambda x: x[0])
    return [(name, p) for name, p in named_params]


def ensure_params_compatible(
    ema_named_params: List[Tuple[str, torch.Tensor]],
    model_named_params: List[Tuple[str, torch.Tensor]]
):
    """Assert that parameter lists are compatible for EMA update"""
    assert len(ema_named_params) == len(model_named_params), \
        f"Parameter count mismatch: EMA has {len(ema_named_params)}, model has {len(model_named_params)}"

    for i, ((ema_name, ema_p), (model_name, model_p)) in enumerate(zip(ema_named_params, model_named_params)):
        assert ema_name == model_name, \
            f"Parameter name mismatch at index {i}: EMA '{ema_name}' vs model '{model_name}'"
        assert ema_p.shape == model_p.shape, \
            f"Shape mismatch for '{ema_name}': EMA {ema_p.shape} vs model {model_p.shape}"
        assert ema_p.dtype == model_p.dtype, \
            f"Dtype mismatch for '{ema_name}': EMA {ema_p.dtype} vs model {model_p.dtype}"


def freeze(model: Latte):
    for p in model.parameters():
        p.requires_grad = False
