"""
Distributed job schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.distributed import DistributedJobStatus


class DistributedJobCreateRequest(BaseModel):
    """
    Request payload for creating a distributed job.
    """

    name: str
    description: str | None = None


class DistributedJobConfigUpdateRequest(BaseModel):
    """
    Request payload for updating distributed job config.
    """

    config: dict[str, Any] = Field(default_factory=dict)


class DistributedJobResponse(BaseModel):
    """
    Response payload for distributed job resource.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    expected_clients: int
    config_json: dict[str, Any]
    status: DistributedJobStatus
    created_at: datetime
    updated_at: datetime
