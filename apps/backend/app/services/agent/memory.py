"""
Helpers for building and formatting experiment memory in bench mode.
"""

from __future__ import annotations

import copy
from typing import Any

from .objectives import AgentOptimizationObjective
from .state import ConfigChange, ExperimentRecord


def compute_config_diff(
    previous_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> list[ConfigChange]:
    """Compute a normalized diff between two config trees."""
    if previous_config is None:
        return [
            ConfigChange(
                path="$",
                change_type="initialize",
                old_value=None,
                new_value=copy.deepcopy(current_config),
            )
        ]

    changes: list[ConfigChange] = []
    _walk_config_diff(previous_config, current_config, path="", changes=changes)
    changes.sort(key=lambda item: item.path)
    return changes


def summarize_config_diff(changes: list[ConfigChange], *, limit: int = 6) -> str:
    """Convert a structured config diff into a short human-readable summary."""
    if not changes:
        return "No config changes."

    parts: list[str] = []
    for change in changes[:limit]:
        target = change.path or "$"
        if change.change_type == "initialize":
            parts.append("initialized baseline config")
        elif change.change_type == "added":
            parts.append(f"{target} added -> {change.new_value}")
        elif change.change_type == "removed":
            parts.append(f"{target} removed (was {change.old_value})")
        else:
            parts.append(f"{target}: {change.old_value} -> {change.new_value}")

    if len(changes) > limit:
        parts.append(f"... and {len(changes) - limit} more changes")
    return "; ".join(parts)


def build_history_context(
    history: list[ExperimentRecord],
    *,
    objective: AgentOptimizationObjective | None = None,
    best_metrics: dict[str, Any] | None = None,
    best_config: dict[str, Any] | None = None,
    max_recent_records: int = 10,
) -> str:
    """Build a compact summary of completed experiments."""
    if not history:
        return "No prior experiments."

    lines: list[str] = []
    for record in history[-max_recent_records:]:
        score_str = f"{record.score:.4f}" if record.score is not None else "-"
        lines.append(
            f"Experiment {record.name or record.iteration}: "
            f"score={score_str}, "
            f"summary={record.plan_summary or '-'}"
        )

    return "\n".join(lines)


def _walk_config_diff(
    previous: Any,
    current: Any,
    *,
    path: str,
    changes: list[ConfigChange],
) -> None:
    if isinstance(previous, dict) and isinstance(current, dict):
        keys = sorted(set(previous.keys()) | set(current.keys()))
        for key in keys:
            next_path = f"{path}.{key}" if path else key
            if key not in previous:
                changes.append(
                    ConfigChange(
                        path=next_path,
                        change_type="added",
                        old_value=None,
                        new_value=copy.deepcopy(current[key]),
                    )
                )
                continue
            if key not in current:
                changes.append(
                    ConfigChange(
                        path=next_path,
                        change_type="removed",
                        old_value=copy.deepcopy(previous[key]),
                        new_value=None,
                    )
                )
                continue
            _walk_config_diff(previous[key], current[key], path=next_path, changes=changes)
        return

    if previous != current:
        changes.append(
            ConfigChange(
                path=path or "$",
                change_type="updated",
                old_value=copy.deepcopy(previous),
                new_value=copy.deepcopy(current),
            )
        )
