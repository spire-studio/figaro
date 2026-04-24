"""
Distributed session schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.distributed import DistributedSessionStatus


class DistributedSessionCreateRequest(BaseModel):
    """
    Request payload for creating or getting distributed session.
    """

    server_ip: str = "localhost"
    server_port: int = 50052


class DistributedSessionResponse(BaseModel):
    """
    Response payload for distributed session resource.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: int
    server_ip: str
    server_port: int
    status: DistributedSessionStatus
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DistributedSessionProgressResponse(BaseModel):
    """
    Response payload for distributed server progress.
    """

    session_id: str
    status: str
    pid: int | None
    started_at: datetime | None
    ended_at: datetime | None
    exit_code: int | None
    total_rounds: int
    last_round: int
    latest_global_loss: float | None
    latest_global_accuracy: float | None
    metrics_json: dict[str, Any] = Field(default_factory=dict)
    log_path: str | None = None
