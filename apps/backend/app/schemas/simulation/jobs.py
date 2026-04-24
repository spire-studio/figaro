"""
Simulation job API schemas.

Defines request and response payloads for simulation job management and
configuration endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.simulation import SimulationJobStatus


class SimulationJobCreateRequest(BaseModel):
    """Request payload for creating a simulation job."""

    name: str
    description: str | None = None


class SimulationJobUpdateRequest(BaseModel):
    """Request payload for updating simulation job metadata."""

    name: str | None = None
    description: str | None = None
    status: SimulationJobStatus | None = None


class SimulationJobConfigUpdateRequest(BaseModel):
    """Request payload for replacing simulation job configuration."""

    config: dict[str, Any]


class SimulationJobCopyRequest(BaseModel):
    """Request payload for copying a simulation job."""

    name: str | None = None


class SimulationJobConfigResponse(BaseModel):
    """Response payload for simulation job configuration."""

    job_id: int
    config_json: dict[str, Any]
    updated_at: datetime


class SimulationJobResponse(BaseModel):
    """Response payload for simulation job resource."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    status: SimulationJobStatus
    config_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime
