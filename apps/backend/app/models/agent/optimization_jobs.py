"""
Persistent agent optimization job models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Column, DateTime, Enum as SAEnum, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

from app.models.base import utcnow


class AgentOptimizationJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentOptimizationJob(SQLModel, table=True):
    __tablename__ = "agent_optimization_jobs"

    id: int | None = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, nullable=False),
    )
    task_id: str = Field(
        sa_column=Column(String(64), nullable=False, unique=True, index=True),
    )
    job_name: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True, unique=True, index=True),
    )
    goal: str = Field(
        sa_column=Column(Text, nullable=False),
    )
    system_mode: str = Field(
        default="simulation",
        sa_column=Column(String(32), nullable=False),
    )
    model_name: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
    )
    status: AgentOptimizationJobStatus = Field(
        default=AgentOptimizationJobStatus.QUEUED,
        sa_column=Column(SAEnum(AgentOptimizationJobStatus, name="agent_optimization_job_status"), nullable=False, index=True),
    )
    current_phase: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
    )
    max_iterations: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False),
    )
    current_iteration: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False),
    )
    completed_iterations: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False),
    )
    simulation_job_id: int | None = Field(
        default=None,
        sa_column=Column(Integer, nullable=True, index=True),
    )
    best_score: float | None = Field(
        default=None,
        sa_column=Column(Float, nullable=True),
    )
    snapshot_json: dict[str, Any] = Field(
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
    finished_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
