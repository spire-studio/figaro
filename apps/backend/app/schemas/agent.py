"""
API schemas for the agent optimization endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.schemas.simulation import SimulationRunMetricsResponse
from app.services.agent.objectives import AgentOptimizationObjective


class AgentOptimizeRequest(BaseModel):
    """
    Request payload for starting an agent-driven optimization loop.
    """

    goal: str = Field(..., description="Natural-language experiment request.")
    max_iterations: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum number of experiments the agent may run.",
    )
    system_mode: str = Field(
        "simulation",
        description="System mode to use, currently only 'simulation' is supported.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional preferred base model name.",
    )
    job_name: Optional[str] = Field(
        default=None,
        description="Optional simulation job name for this optimization task.",
    )
    objective: AgentOptimizationObjective = Field(
        default=AgentOptimizationObjective.AUTO,
        description="Objective for the experiment batch. 'accuracy' maximizes accuracy; 'auto' defaults to accuracy.",
    )


class AgentModelsResponse(BaseModel):
    """Response payload for available agent model names."""

    models: list[str] = Field(default_factory=list)
    default_model: str = Field(..., description="Default model configured by backend.")


class AgentConfigChangeResponse(BaseModel):
    """One normalized config change between two iterations."""

    path: str
    change_type: str
    old_value: Any = None
    new_value: Any = None


class AgentCurrentPlanResponse(BaseModel):
    """Planner output for the iteration currently being prepared or executed."""

    iteration: int
    iteration_goal: str
    plan_summary: str | None = None
    hypothesis: str | None = None
    rationale: list[str] = Field(default_factory=list)
    config_patch: dict[str, Any] = Field(default_factory=dict)
    config_diff: list[AgentConfigChangeResponse] = Field(default_factory=list)


class AgentExperimentSummary(BaseModel):
    """
    One experiment executed by the agent.
    """

    iteration: int = 0
    run_id: str
    job_id: int
    iteration_goal: str | None = None
    plan_summary: str | None = None
    hypothesis: str | None = None
    rationale: list[str] = Field(default_factory=list)
    config_patch: dict[str, Any] = Field(default_factory=dict)
    config_diff: list[AgentConfigChangeResponse] = Field(default_factory=list)
    config: dict[str, Any]
    metrics: SimulationRunMetricsResponse
    score: float | None = None
    result_summary: str | None = None
    decision: str | None = None
    lessons_learned: list[str] = Field(default_factory=list)


class AgentOptimizeResponse(BaseModel):
    """
    Final summary returned after the agent completes its optimization loop.
    """

    goal: str
    job_name: str | None = None
    max_iterations: int
    objective: AgentOptimizationObjective = AgentOptimizationObjective.AUTO
    resolved_objective: AgentOptimizationObjective = AgentOptimizationObjective.ACCURACY
    iterations_executed: int
    best_config: dict[str, Any] | None = None
    best_metrics: SimulationRunMetricsResponse | None = None
    experiments: list[AgentExperimentSummary] = Field(default_factory=list)
    summary_text: str | None = None


class AgentCurrentExperimentResponse(BaseModel):
    """Live view of the experiment currently being planned or executed."""

    iteration: int
    phase: str | None = None
    job_id: int | None = None
    job_name: str | None = None
    run_id: str | None = None
    run_status: str | None = None
    config: dict[str, Any] | None = None
    metrics: SimulationRunMetricsResponse | None = None


class AgentOptimizeProgressResponse(BaseModel):
    """Progress payload for an asynchronous optimization task."""

    optimization_job_id: int | None = None
    task_id: str
    status: str
    goal: str
    job_name: str | None = None
    max_iterations: int
    model_name: str | None = None
    objective: AgentOptimizationObjective = AgentOptimizationObjective.AUTO
    resolved_objective: AgentOptimizationObjective = AgentOptimizationObjective.ACCURACY
    current_phase: str | None = None
    current_iteration: int = 0
    completed_iterations: int = 0
    current_plan: AgentCurrentPlanResponse | None = None
    current_experiment: AgentCurrentExperimentResponse | None = None
    best_config: dict[str, Any] | None = None
    best_metrics: SimulationRunMetricsResponse | None = None
    experiments: list[AgentExperimentSummary] = Field(default_factory=list)
    summary_text: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    finished_at: datetime | None = None


class AgentOptimizationJobSummaryResponse(BaseModel):
    """Summary row for one persisted agent optimization job."""

    optimization_job_id: int
    task_id: str
    job_name: str | None = None
    status: str
    goal: str
    model_name: str | None = None
    objective: AgentOptimizationObjective = AgentOptimizationObjective.AUTO
    resolved_objective: AgentOptimizationObjective = AgentOptimizationObjective.ACCURACY
    current_phase: str | None = None
    max_iterations: int
    current_iteration: int = 0
    completed_iterations: int = 0
    best_score: float | None = None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None


# -----------------------------------------------------------------------
#  Agent Experiment / Run response schemas (DB-backed tables)
# -----------------------------------------------------------------------


class AgentExperimentResponse(BaseModel):
    """Response payload for an agent experiment row."""

    id: int
    name: str
    description: str | None = None
    status: str
    config_json: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class AgentRunResponse(BaseModel):
    """Response payload for an agent experiment run row."""

    id: str
    experiment_id: int
    status: str
    config_json: dict[str, Any] = Field(default_factory=dict)
    metrics_json: dict[str, Any] = Field(default_factory=dict)
    started_at: str | None = None
    ended_at: str | None = None
    created_at: str


class AgentRunLogResponse(BaseModel):
    """Response payload for an agent experiment run log entry."""

    id: int
    level: str
    message: str
    created_at: str


class AgentRunMetricsResponse(BaseModel):
    """Response payload for agent run metrics."""

    run_id: str
    metrics: dict[str, Any] = Field(default_factory=dict)
