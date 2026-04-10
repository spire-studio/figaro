"""
Simulation job models.

Defines simulation job metadata, persisted configuration, and high-level job
lifecycle status.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Column, DateTime, Enum as SAEnum, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

from app.core.constants import JobLimits
from app.models.base import utcnow


class SimulationJobStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    ARCHIVED = "archived"
    DONE = "done"


class SimulationJob(SQLModel, table=True):
    __tablename__ = "simulation_jobs"

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
    status: SimulationJobStatus = Field(
        default=SimulationJobStatus.DRAFT,
        sa_column=Column(SAEnum(SimulationJobStatus, name="simulation_job_status"), nullable=False, index=True),
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
