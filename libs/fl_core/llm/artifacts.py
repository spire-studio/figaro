from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


ADAPTER_STATE_KEY = "adapter_state"
METADATA_KEY = "metadata"


def save_adapter_artifact(
    path: str | Path,
    adapter_state: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist an adapter state dict with lightweight metadata."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        ADAPTER_STATE_KEY: {key: value.detach().cpu() for key, value in adapter_state.items()},
        METADATA_KEY: dict(metadata or {}),
    }
    torch.save(payload, resolved)
    return resolved


def load_adapter_artifact(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load an adapter artifact saved by save_adapter_artifact."""
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Adapter artifact must contain a dictionary payload")
    adapter_state = payload.get(ADAPTER_STATE_KEY)
    metadata = payload.get(METADATA_KEY, {})
    if not isinstance(adapter_state, dict):
        raise ValueError("Adapter artifact is missing adapter_state")
    if not isinstance(metadata, dict):
        metadata = {}
    return adapter_state, metadata


def adapter_state_size_bytes(adapter_state: Mapping[str, torch.Tensor]) -> int:
    """Return the total tensor storage size for an adapter state dict."""
    return sum(tensor.numel() * tensor.element_size() for tensor in adapter_state.values())

