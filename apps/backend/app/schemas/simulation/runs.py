"""
Simulation run API schemas.

Defines response payloads for simulation run status, logs, result artifacts,
and normalized training metrics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.simulation import SimulationRunStatus


class SimulationRunResponse(BaseModel):
    """Response payload for simulation run resource."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: int
    status: SimulationRunStatus
    process_id: int | None
    command: str
    started_at: datetime | None
    ended_at: datetime | None
    exit_code: int | None
    error_message: str | None
    job_name: str | None = None
    job_description: str | None = None
    created_at: datetime
    updated_at: datetime


class SimulationRunLogResponse(BaseModel):
    """Response payload for one simulation run log entry."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    level: str
    message: str
    created_at: datetime


class SimulationRunResultResponse(BaseModel):
    """Response payload for one simulation run artifact record."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    artifact_type: str
    path: str
    metadata_json: dict[str, Any]
    created_at: datetime


class SimulationRunGlobalResults(BaseModel):
    """Global aggregation metrics in simulation mode."""

    rounds: list[int] = Field(default_factory=list)
    global_loss: list[float] = Field(default_factory=list)
    global_accuracy: list[float] = Field(default_factory=list)


class SimulationRunLlmResults(BaseModel):
    """LLM PEFT metric series."""

    rounds: list[int] = Field(default_factory=list)
    train_loss: list[float] = Field(default_factory=list)
    validation_loss: list[float] = Field(default_factory=list)
    perplexity: list[float] = Field(default_factory=list)
    token_throughput: list[float] = Field(default_factory=list)
    adapter_size_bytes: list[int] = Field(default_factory=list)


class SimulationRunClientSeries(BaseModel):
    """Per-client metric series in simulation mode."""

    rounds: list[int] = Field(default_factory=list)
    train_loss: list[float] = Field(default_factory=list)
    train_acc: list[float] = Field(default_factory=list)
    test_loss: list[float] = Field(default_factory=list)
    test_acc: list[float] = Field(default_factory=list)


class SimulationRunExperimentInfo(BaseModel):
    """Experiment configuration snapshot sections."""

    basic: dict[str, Any] = Field(default_factory=dict)
    federated: dict[str, Any] = Field(default_factory=dict)
    security: dict[str, Any] = Field(default_factory=dict)


class SimulationRunMetricsResponse(BaseModel):
    """Response payload for normalized simulation run metrics."""

    experiment_info: SimulationRunExperimentInfo = Field(default_factory=SimulationRunExperimentInfo)
    global_results: SimulationRunGlobalResults = Field(default_factory=SimulationRunGlobalResults)
    llm_results: SimulationRunLlmResults = Field(default_factory=SimulationRunLlmResults)
    llm_dataset: dict[str, Any] = Field(default_factory=dict)
    llm_evaluation: dict[str, Any] = Field(default_factory=dict)
    llm_runtime: dict[str, Any] = Field(default_factory=dict)
    llm_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    client_results: dict[str, SimulationRunClientSeries] = Field(default_factory=dict)
