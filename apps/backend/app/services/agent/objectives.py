"""
Objective helpers for the bench-mode experiment executor.

Bench mode does not optimize — it runs experiments and reports results.
The objective system is kept minimal for compatibility.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class AgentOptimizationObjective(str, Enum):
    AUTO = "auto"
    ACCURACY = "accuracy"


def normalize_objective(
    value: str | AgentOptimizationObjective | None,
) -> AgentOptimizationObjective:
    """Parse a requested objective value with a robust fallback."""
    if isinstance(value, AgentOptimizationObjective):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for candidate in AgentOptimizationObjective:
            if candidate.value == normalized:
                return candidate
    return AgentOptimizationObjective.AUTO


def resolve_objective(
    *,
    goal: str,
    requested_objective: str | AgentOptimizationObjective | None,
) -> AgentOptimizationObjective:
    """Resolve the effective objective — always accuracy in bench mode."""
    return AgentOptimizationObjective.ACCURACY


def objective_display_name(
    objective: str | AgentOptimizationObjective | None,
) -> str:
    return "accuracy"


def objective_metric_label(
    objective: str | AgentOptimizationObjective | None,
) -> str:
    return "global accuracy"


def is_better_score(
    *,
    objective: str | AgentOptimizationObjective | None,
    candidate_score: float | None,
    reference_score: float | None,
) -> bool:
    """Higher accuracy is always better in bench mode."""
    if candidate_score is None:
        return False
    if reference_score is None:
        return True
    return candidate_score > reference_score
