"""
Distributed runtime schemas.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DistributedRuntimeStatusResponse(BaseModel):
    """
    Response payload for runtime process status.
    """

    role: str
    runtime_id: str
    status: str
    pid: int | None
    started_at: datetime | None
    ended_at: datetime | None
    exit_code: int | None
    config_path: str
    log_path: str | None
    results_path: str | None
    total_rounds: int
