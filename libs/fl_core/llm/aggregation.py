from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class AdapterClientUpdate:
    """A client-side PEFT adapter update plus weighting metadata."""

    adapter_state: Mapping[str, torch.Tensor]
    num_examples: int
    num_tokens: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


def aggregate_adapter_state_dicts(
    updates: list[AdapterClientUpdate],
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted-average LoRA/PEFT adapter tensors by client example count."""
    if not updates:
        raise ValueError("updates must not be empty")

    reference_keys = tuple(updates[0].adapter_state.keys())
    if not reference_keys:
        raise ValueError("adapter_state must not be empty")

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    weights = _normalized_example_weights(updates)
    aggregated: dict[str, torch.Tensor] = {}

    for key in reference_keys:
        first_tensor = updates[0].adapter_state[key]
        if not torch.is_floating_point(first_tensor):
            raise TypeError(f"Adapter tensor {key!r} must be floating point")
        accumulator = torch.zeros_like(first_tensor, device=target_device)

        for update, weight in zip(updates, weights):
            _validate_update_tensor(update, key, first_tensor)
            accumulator += update.adapter_state[key].to(target_device) * weight
        aggregated[key] = accumulator.detach().cpu()

    return aggregated


def _normalized_example_weights(updates: list[AdapterClientUpdate]) -> list[float]:
    raw_weights = [max(0, int(update.num_examples)) for update in updates]
    total = sum(raw_weights)
    if total <= 0:
        return [1.0 / len(updates)] * len(updates)
    return [weight / total for weight in raw_weights]


def _validate_update_tensor(update: AdapterClientUpdate, key: str, reference: torch.Tensor) -> None:
    if key not in update.adapter_state:
        raise ValueError(f"Adapter update is missing key: {key}")
    tensor = update.adapter_state[key]
    if tensor.shape != reference.shape:
        raise ValueError(f"Adapter tensor {key!r} has mismatched shape: {tuple(tensor.shape)} != {tuple(reference.shape)}")
    if tensor.dtype != reference.dtype:
        raise ValueError(f"Adapter tensor {key!r} has mismatched dtype: {tensor.dtype} != {reference.dtype}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"Adapter tensor {key!r} must be floating point")

