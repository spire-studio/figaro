"""
Distributed participant models.

Defines participant lifecycle status and persistence schema for nodes that
join a distributed session from the client side.
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

from app.core.constants import DistributedLimits
from app.models.base import utcnow


class DistributedParticipantStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    READY = "ready"
    RUNNING = "running"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    DISCONNECTED = "disconnected"


class DistributedParticipant(SQLModel, table=True):
    __tablename__ = "distributed_participants"

    id: str = Field(
        default_factory=lambda: uuid4().hex[:8],
        sa_column=Column(String(DistributedLimits.ID_MAX_LENGTH), primary_key=True, nullable=False),
    )
    session_id: str = Field(
        sa_column=Column(
            String(DistributedLimits.ID_MAX_LENGTH),
            ForeignKey("distributed_sessions.id"),
            nullable=False,
            index=True,
        ),
    )
    participant_name: str = Field(
        sa_column=Column(String(DistributedLimits.CLIENT_NAME_MAX_LENGTH), nullable=False, index=True),
    )
    assigned_participant_id: int | None = Field(
        default=None,
        sa_column=Column(Integer, nullable=True, index=True),
    )
    status: DistributedParticipantStatus = Field(
        default=DistributedParticipantStatus.PENDING_APPROVAL,
        sa_column=Column(
            SAEnum(DistributedParticipantStatus, name="distributedparticipantstatus"),
            nullable=False,
            index=True,
        ),
    )
    metadata_json: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON().with_variant(JSONB, "postgresql"), nullable=False),
    )
    requested_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    approved_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    ready_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    last_seen_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
