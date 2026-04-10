"""
Distributed job models.

Defines job-level configuration and lifecycle state for distributed training.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Column, DateTime, Enum as SAEnum, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

from app.core.constants import DistributedLimits
from app.models.base import utcnow


class DistributedJobStatus(str, Enum):
    DRAFT = "draft"
    WAITING_CLIENTS = "waiting_clients"
    READY = "ready"
    RUNNING = "running"
    FINISHED = "finished"


class DistributedJob(SQLModel, table=True):
    __tablename__ = "distributed_jobs"

    id: int | None = Field(
        default=None,
        sa_column=Column(Integer, primary_key=True, nullable=False),
    )
    name: str = Field(
        sa_column=Column(String(DistributedLimits.NAME_MAX_LENGTH), nullable=False, unique=True, index=True),
    )
    description: str | None = Field(
        default=None,
        sa_column=Column(String(DistributedLimits.DESCRIPTION_MAX_LENGTH), nullable=True),
    )
    expected_clients: int = Field(
        default=1,
        sa_column=Column(Integer, nullable=False),
    )
    config_json: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON().with_variant(JSONB, "postgresql"), nullable=False),
    )
    malicious_client_ids_json: list[int] = Field(
        default_factory=list,
        sa_column=Column(JSON().with_variant(JSONB, "postgresql"), nullable=False),
    )
    status: DistributedJobStatus = Field(
        default=DistributedJobStatus.DRAFT,
        sa_column=Column(SAEnum(DistributedJobStatus, name="distributedjobstatus"), nullable=False, index=True),
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
