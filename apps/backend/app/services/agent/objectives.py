"""
Objective helpers for the bench-mode experiment executor.

Bench mode runs a planned batch of experiments and reports comparable scores.
The objective surface is intentionally small for API compatibility.
"""

from __future__ import annotations

from enum import Enum


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
    """Resolve the effective objective.

    LLM PEFT runs are scored by negative loss in the summary layer, but the
    public objective enum remains accuracy-only for the current Agent API.
    """
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
    """Higher comparable score is better."""
    if candidate_score is None:
        return False
    if reference_score is None:
        return True
    return candidate_score > reference_score
