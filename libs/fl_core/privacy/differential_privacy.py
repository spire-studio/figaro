from __future__ import annotations

import logging
from typing import Dict

import torch


def _is_trainable_float(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.is_floating_point()
        and tensor.dim() > 0
        and "num_batches_tracked" not in name
        and "running_" not in name
    )


class DifferentialPrivacyManager:
    def __init__(self, clipping_norm: float = 1.0, noise_multiplier: float = 0.0):
        self.clipping_norm = float(clipping_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.logger = logging.getLogger("DifferentialPrivacyManager")

    def apply(self, update_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trainable = {
            name: tensor
            for name, tensor in update_dict.items()
            if _is_trainable_float(name, tensor)
        }
        if not trainable:
            return update_dict

        total_norm = torch.sqrt(
            sum(torch.sum(tensor.detach() ** 2) for tensor in trainable.values())
        )
        clip_factor = min(1.0, self.clipping_norm / (total_norm.item() + 1e-12))
        noise_std = self.noise_multiplier * self.clipping_norm

        protected = {}
        for name, tensor in update_dict.items():
            if not _is_trainable_float(name, tensor):
                protected[name] = tensor
                continue
            value = tensor * clip_factor
            if noise_std > 0:
                value = value + torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=tensor.shape,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
            protected[name] = value
        return protected
