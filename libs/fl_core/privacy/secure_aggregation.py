from __future__ import annotations

import logging
from typing import Dict, List

import torch


def _is_trainable_float(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.is_floating_point()
        and tensor.dim() > 0
        and "num_batches_tracked" not in name
        and "running_" not in name
    )


class SecureAggregationMasker:
    """
    Simulation-only additive masking.

    Masks are generated so their unweighted sum is zero. This is suitable for
    equal-weight FedAvg/simple average in local simulation; it is intentionally
    not a production secure aggregation protocol.
    """

    def __init__(self, mask_std: float = 1.0):
        self.mask_std = float(mask_std)
        self.logger = logging.getLogger("SecureAggregationMasker")

    def mask_models(self, updates: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        if len(updates) <= 1 or self.mask_std <= 0:
            return updates

        masked = [{name: tensor.clone() for name, tensor in update.items()} for update in updates]
        reference = updates[0]

        for name, tensor in reference.items():
            if not _is_trainable_float(name, tensor):
                continue
            running_sum = torch.zeros_like(tensor)
            for idx in range(len(updates) - 1):
                mask = torch.normal(
                    mean=0.0,
                    std=self.mask_std,
                    size=tensor.shape,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                masked[idx][name] = masked[idx][name] + mask
                running_sum = running_sum + mask
            masked[-1][name] = masked[-1][name] - running_sum

        return masked
