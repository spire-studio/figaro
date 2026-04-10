"""
Agent state definitions for bench-mode FL experiment execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .objectives import AgentOptimizationObjective


@dataclass
class ConfigChange:
    """One normalized configuration change between two configs."""

    path: str
    change_type: str
    old_value: Any = None
    new_value: Any = None


@dataclass
class ExperimentPlan:
    """
    One planned experiment within a batch.

    In bench mode each plan corresponds to one experiment in the batch,
    identified by ``name``.
    """

    iteration: int
    iteration_goal: str
    name: str = ""
    plan_summary: str | None = None
    hypothesis: str | None = None
    rationale: list[str] = field(default_factory=list)
    config_patch: dict[str, Any] = field(default_factory=dict)
    config_diff: list[ConfigChange] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    """One completed experiment together with its results."""

    iteration: int
    run_id: str
    job_id: int
    config: dict[str, Any]
    metrics: dict[str, Any]
    name: str = ""
    iteration_goal: str | None = None
    plan_summary: str | None = None
    hypothesis: str | None = None
    rationale: list[str] = field(default_factory=list)
    config_patch: dict[str, Any] = field(default_factory=dict)
    config_diff: list[ConfigChange] = field(default_factory=list)
    score: float | None = None
    result_summary: str | None = None
    decision: str | None = None
    lessons_learned: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class AgentState:
    """
    High-level state for the bench-mode LangGraph pipeline.

    ``goal`` is the user-provided natural-language experiment request.
    ``experiments`` holds the parsed batch of experiment plans.
    ``experiment_results`` holds all completed experiment results.
    ``history`` is kept for API compatibility and mirrors ``experiment_results``.
    """

    goal: str
    max_iterations: int = 1  # bench mode runs a single pipeline pass

    system_mode: str = "simulation"
    model_name: str | None = None
    job_name: str | None = None
    objective: AgentOptimizationObjective = AgentOptimizationObjective.AUTO
    resolved_objective: AgentOptimizationObjective = AgentOptimizationObjective.ACCURACY

    # Batch experiment list produced by the parse node
    experiments: list[ExperimentPlan] = field(default_factory=list)
    # All completed experiment results
    experiment_results: list[ExperimentRecord] = field(default_factory=list)

    # Current run tracking (used during launch/collect)
    iteration: int = 0
    current_plan: ExperimentPlan | None = None
    current_config: dict[str, Any] = field(default_factory=dict)
    current_job_id: int | None = None
    current_job_name: str | None = None
    current_run_id: str | None = None
    current_run_status: str | None = None
    phase: str = "queued"
    error_message: str | None = None

    # API-compatible fields
    history: list[ExperimentRecord] = field(default_factory=list)
    terminated: bool = False
    summary: str | None = None
