from __future__ import annotations

import hashlib
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


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest for an artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_artifact_record(
    path: str | Path,
    *,
    round_num: int,
    size_bytes: int,
    selected_clients: list[int],
    parent_path: str | None = None,
    parent_sha256: str | None = None,
) -> dict[str, Any]:
    """Build serializable lineage metadata for one global adapter artifact."""
    resolved = Path(path)
    return {
        "round": int(round_num),
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(size_bytes),
        "selected_clients": list(selected_clients),
        "parent_path": parent_path,
        "parent_sha256": parent_sha256,
    }
