"""
Distributed participant schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.distributed import DistributedParticipantStatus


class DistributedParticipantConnectRequest(BaseModel):
    """
    Request payload for participant connection request.
    """

    participant_name: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class DistributedParticipantResponse(BaseModel):
    """
    Response payload for distributed participant resource.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    participant_name: str
    assigned_participant_id: int | None
    status: DistributedParticipantStatus
    metadata_json: dict[str, Any]
    requested_at: datetime
    approved_at: datetime | None
    ready_at: datetime | None
    last_seen_at: datetime
    created_at: datetime
    updated_at: datetime


class DistributedParticipantRuntimeConfigResponse(BaseModel):
    """
    Response payload for approved participant runtime config.
    """

    participant_id: str
    session_id: str
    assigned_participant_id: int
    server_ip: str
    server_port: int
    config_json: dict[str, Any]


class DistributedLocalParticipantStartRequest(BaseModel):
    """
    Request payload for starting local participant runtime.
    """

    participant_id: str
    config_json: dict[str, Any] = Field(default_factory=dict)
