"""
Helpers for inspecting the current federated learning stack capabilities.

Used by the Agent to ground its proposals in the actually supported
datasets, models, and aggregations.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_list_literal(text: str) -> List[Any]:
    """
    Extract a Python list literal from a string like: [ "a", "b" ].
    Returns [] on failure.
    """
    try:
        value = ast.literal_eval(text)
        if isinstance(value, list):
            return value
    except Exception:
        pass
    return []


def _extract_supported_datasets(project_root: Path) -> List[str]:
    path = project_root / "libs" / "fl_core" / "data" / "data_loader.py"
    if not path.exists():
        return []
    text = _read_text(path)
    match = re.search(r"supported_datasets\s*=\s*(\[[^\]]*\])", text, flags=re.DOTALL)
    if not match:
        return []
    items = _extract_list_literal(match.group(1))
    out: List[str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
    return out


def _extract_model_registry_keys(project_root: Path) -> List[str]:
    path = project_root / "libs" / "fl_core" / "models" / "model_manager.py"
    if not path.exists():
        return []
    text = _read_text(path)
    match = re.search(r"model_registry\s*=\s*\{(.*?)\}\s*\n", text, flags=re.DOTALL)
    if not match:
        return []
    block = match.group(1)
    keys = re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]\s*:", block)
    seen: set[str] = set()
    out: List[str] = []
    for key in keys:
        lower = key.lower()
        if lower not in seen:
            seen.add(lower)
            out.append(lower)
    return out


def _extract_aggregation_strategies(project_root: Path) -> List[str]:
    """
    Returns aggregation strategies from federated/aggregation.py without imports.
    """
    path = project_root / "libs" / "fl_core" / "federated" / "aggregation.py"
    if not path.exists():
        return []
    text = _read_text(path)

    match = re.search(r"_strategies\s*=\s*\{(.*?)\}", text, flags=re.DOTALL)
    if not match:
        return []
    aggregations = re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]\s*:", match.group(1))

    seen: set[str] = set()
    out: List[str] = []
    for item in aggregations:
        lower = item.lower()
        if lower not in seen:
            seen.add(lower)
            out.append(lower)
    return out


def get_platform_capabilities(project_root: Path | None = None) -> Dict[str, Any]:
    """
    Inspect the current codebase and return the supported configuration options
    for datasets, models, and aggregations.
    """
    root = project_root or Path(__file__).resolve().parents[4]

    datasets = _extract_supported_datasets(root)
    models = _extract_model_registry_keys(root)
    aggregations = _extract_aggregation_strategies(root)

    metrics = {
        "global_results": ["rounds", "global_loss", "global_accuracy"],
        "client_results": ["train_loss", "train_acc", "test_loss", "test_acc"],
    }

    return {
        "datasets": datasets,
        "distributions": ["iid", "non_iid"],
        "models": models,
        "aggregations": aggregations,
        "metrics": metrics,
    }


