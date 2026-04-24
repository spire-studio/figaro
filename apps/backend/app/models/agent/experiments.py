"""
Agent experiment models.

Defines agent experiment metadata, run lifecycle, logs, and result payloads.
Independent of simulation tables to avoid polluting the Simulation page.
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

from app.core.constants import JobLimits, RunLimits
from app.models.base import utcnow


class AgentExperimentStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    DONE = "done"


class AgentExperiment(SQLModel, table=True):
    __tablename__ = "agent_experiments"

    id: int | None = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, nullable=False),
    )
    name: str = Field(
        sa_column=Column(String(JobLimits.NAME_MAX_LENGTH), nullable=False, unique=True, index=True),
    )
    description: str | None = Field(
        default=None,
        sa_column=Column(String(JobLimits.DESCRIPTION_MAX_LENGTH), nullable=True),
    )
    status: AgentExperimentStatus = Field(
        default=AgentExperimentStatus.DRAFT,
        sa_column=Column(SAEnum(AgentExperimentStatus, name="agent_experiment_status"), nullable=False, index=True),
    )
    config_json: dict[str, Any] = Field(
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


class AgentExperimentRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentExperimentRun(SQLModel, table=True):
    __tablename__ = "agent_experiment_runs"

    id: str = Field(
        default_factory=lambda: uuid4().hex[:8],
        sa_column=Column(String(RunLimits.ID_MAX_LENGTH), primary_key=True, nullable=False),
    )
    experiment_id: int = Field(
        sa_column=Column(Integer, ForeignKey("agent_experiments.id"), nullable=False, index=True),
    )
    status: AgentExperimentRunStatus = Field(
        default=AgentExperimentRunStatus.QUEUED,
        sa_column=Column(SAEnum(AgentExperimentRunStatus, name="agent_experiment_run_status"), nullable=False, index=True),
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


class AgentExperimentRunLog(SQLModel, table=True):
    __tablename__ = "agent_experiment_run_logs"

    id: int | None = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, nullable=False),
    )
    run_id: str = Field(
        sa_column=Column(String(RunLimits.ID_MAX_LENGTH), ForeignKey("agent_experiment_runs.id"), nullable=False, index=True),
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


class AgentExperimentRunResult(SQLModel, table=True):
    __tablename__ = "agent_experiment_run_results"

    id: int | None = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, nullable=False),
    )
    run_id: str = Field(
        sa_column=Column(String(RunLimits.ID_MAX_LENGTH), ForeignKey("agent_experiment_runs.id"), nullable=False, index=True),
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
