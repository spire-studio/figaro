"""
Distributed session models.

Defines server session lifecycle and network endpoint fields for a distributed
training round.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import uuid4

from sqlalchemy import Column, DateTime, Enum as SAEnum, ForeignKey, Integer, String
from sqlmodel import Field, SQLModel

from app.core.constants import DistributedLimits
from app.models.base import utcnow


class DistributedSessionStatus(str, Enum):
    WAITING_CLIENTS = "waiting_clients"
    READY_TO_START = "ready_to_start"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class DistributedSession(SQLModel, table=True):
    __tablename__ = "distributed_sessions"

    id: str = Field(
        default_factory=lambda: uuid4().hex[:8],
        sa_column=Column(String(DistributedLimits.ID_MAX_LENGTH), primary_key=True, nullable=False),
    )
    job_id: int = Field(
        sa_column=Column(Integer, ForeignKey("distributed_jobs.id"), nullable=False, index=True),
    )
    server_ip: str = Field(
        default="localhost",
        sa_column=Column(String(DistributedLimits.SERVER_IP_MAX_LENGTH), nullable=False),
    )
    server_port: int = Field(
        default=50052,
        sa_column=Column(Integer, nullable=False),
    )
    status: DistributedSessionStatus = Field(
        default=DistributedSessionStatus.WAITING_CLIENTS,
        sa_column=Column(SAEnum(DistributedSessionStatus, name="distributedsessionstatus"), nullable=False, index=True),
    )
    started_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    ended_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
