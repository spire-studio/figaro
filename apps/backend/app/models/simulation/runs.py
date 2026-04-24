"""
Simulation run models.

Defines run lifecycle, persisted run logs, and run result payloads for
simulation executions.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, DateTime, Enum as SAEnum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

from app.core.constants import RunLimits
from app.models.base import utcnow


class SimulationRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationRun(SQLModel, table=True):
    __tablename__ = "simulation_runs"

    id: str = Field(
        default_factory=lambda: uuid4().hex[:8],
        sa_column=Column(String(RunLimits.ID_MAX_LENGTH), primary_key=True, nullable=False),
    )
    job_id: int = Field(
        sa_column=Column(Integer, ForeignKey("simulation_jobs.id"), nullable=False, index=True),
    )
    status: SimulationRunStatus = Field(
        default=SimulationRunStatus.QUEUED,
        sa_column=Column(SAEnum(SimulationRunStatus, name="simulation_run_status"), nullable=False, index=True),
    )
    process_id: int | None = Field(
        default=None,
        sa_column=Column(Integer, nullable=True, index=True),
    )
    command: str = Field(
        sa_column=Column(String(RunLimits.COMMAND_MAX_LENGTH), nullable=False),
    )
    started_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    ended_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    exit_code: int | None = Field(
        default=None,
        sa_column=Column(Integer, nullable=True),
    )
    error_message: str | None = Field(
        default=None,
        sa_column=Column(String(RunLimits.ERROR_MESSAGE_MAX_LENGTH), nullable=True),
    )
    config_json: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON().with_variant(JSONB, "postgresql"), nullable=False),
    )
    metrics_json: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON().with_variant(JSONB, "postgresql"), nullable=False),
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class SimulationRunLog(SQLModel, table=True):
    __tablename__ = "simulation_run_logs"

    id: int | None = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, nullable=False),
    )
    run_id: str = Field(
        sa_column=Column(String(RunLimits.ID_MAX_LENGTH), ForeignKey("simulation_runs.id"), nullable=False, index=True),
    )
    level: str = Field(
        default="INFO",
        sa_column=Column(String(RunLimits.LOG_LEVEL_MAX_LENGTH), nullable=False),
    )
    message: str = Field(
        sa_column=Column(String(RunLimits.LOG_MESSAGE_MAX_LENGTH), nullable=False),
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )


class SimulationRunResult(SQLModel, table=True):
    __tablename__ = "simulation_run_results"

    id: int | None = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, nullable=False),
    )
    run_id: str = Field(
        sa_column=Column(String(RunLimits.ID_MAX_LENGTH), ForeignKey("simulation_runs.id"), nullable=False, index=True),
    )
    artifact_type: str = Field(
        sa_column=Column(String(RunLimits.ARTIFACT_TYPE_MAX_LENGTH), nullable=False, index=True),
    )
    path: str = Field(
        sa_column=Column(String(RunLimits.ARTIFACT_PATH_MAX_LENGTH), nullable=False),
    )
    metadata_json: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON().with_variant(JSONB, "postgresql"), nullable=False),
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
