"""Job endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status

from app.api.deps import AsyncSessionDep
from app.schemas import (
    Message,
    SimulationJobConfigResponse,
    SimulationJobConfigUpdateRequest,
    SimulationJobCopyRequest,
    SimulationJobCreateRequest,
    SimulationJobResponse,
    SimulationJobUpdateRequest,
    SimulationRunResponse,
)
from app.services import SimulationJobService, SimulationRunService

jobs_router = APIRouter()


@jobs_router.get(
    "/config/schema",
    response_model=dict[str, Any],
    summary="Get Job Config Schema",
)
async def get_config_schema() -> dict[str, Any]:
    """Return the topology configuration schema for job editors."""
    return SimulationJobService.get_config_schema()


@jobs_router.post(
    "",
    response_model=SimulationJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Job",
)
async def create_job(
    payload: SimulationJobCreateRequest,
    session: AsyncSessionDep,
) -> SimulationJobResponse:
    """Create a new job with an initial configuration."""
    service = SimulationJobService(session)
    job = await service.create_job(payload.name, payload.description)
    return SimulationJobResponse.model_validate(job)


@jobs_router.get(
    "",
    response_model=list[SimulationJobResponse],
    summary="List Jobs",
)
async def list_jobs(session: AsyncSessionDep) -> list[SimulationJobResponse]:
    """Return all jobs ordered by creation time."""
    service = SimulationJobService(session)
    jobs = await service.list_jobs()
    return [SimulationJobResponse.model_validate(job) for job in jobs]


@jobs_router.get(
    "/{job_id}",
    response_model=SimulationJobResponse,
    summary="Get Job",
)
async def get_job(job_id: int, session: AsyncSessionDep) -> SimulationJobResponse:
    """Return a single job by its identifier."""
    service = SimulationJobService(session)
    job = await service.get_job_or_raise(job_id)
    return SimulationJobResponse.model_validate(job)


@jobs_router.patch(
    "/{job_id}",
    response_model=SimulationJobResponse,
    summary="Update Job",
)
async def update_job(
    job_id: int,
    payload: SimulationJobUpdateRequest,
    session: AsyncSessionDep,
) -> SimulationJobResponse:
    """Update job metadata or status."""
    service = SimulationJobService(session)
    job = await service.update_job(
        job_id,
        name=payload.name,
        description=payload.description,
        status=payload.status,
    )
    return SimulationJobResponse.model_validate(job)


@jobs_router.delete(
    "/{job_id}",
    response_model=Message,
    summary="Delete Job",
)
async def delete_job(job_id: int, session: AsyncSessionDep) -> Message:
    """Delete a job when it has no active runs."""
    service = SimulationJobService(session)
    await service.delete_job(job_id)
    return Message(message="Job deleted")


@jobs_router.get(
    "/{job_id}/config",
    response_model=SimulationJobConfigResponse,
    summary="Get Job Config",
)
async def get_job_config(
    job_id: int,
    session: AsyncSessionDep,
) -> SimulationJobConfigResponse:
    """Return the current configuration for a job."""
    service = SimulationJobService(session)
    job = await service.get_job_or_raise(job_id)
    config_json = await service.get_job_config(job_id)
    return SimulationJobConfigResponse(
        job_id=job.id,
        config_json=config_json,
        updated_at=job.updated_at,
    )


@jobs_router.patch(
    "/{job_id}/config",
    response_model=SimulationJobConfigResponse,
    summary="Update Job Config",
)
async def update_job_config(
    job_id: int,
    payload: SimulationJobConfigUpdateRequest,
    session: AsyncSessionDep,
) -> SimulationJobConfigResponse:
    """Update the current configuration for a job."""
    service = SimulationJobService(session)
    job = await service.update_job_config(job_id, payload.config)
    return SimulationJobConfigResponse(
        job_id=job.id,
        config_json=job.config_json,
        updated_at=job.updated_at,
    )


@jobs_router.post(
    "/{job_id}/copy",
    response_model=SimulationJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Copy Job",
)
async def copy_job(
    job_id: int,
    payload: SimulationJobCopyRequest,
    session: AsyncSessionDep,
) -> SimulationJobResponse:
    """Create a new job by copying an existing one."""
    service = SimulationJobService(session)
    job = await service.copy_job(job_id, payload.name)
    return SimulationJobResponse.model_validate(job)


@jobs_router.post(
    "/{job_id}/runs",
    response_model=SimulationRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start Job Run",
)
async def run_job(job_id: int, session: AsyncSessionDep) -> SimulationRunResponse:
    """Start a new run for the given job."""
    run_service = SimulationRunService(session)
    run = await run_service.start_run(job_id)
    payload = SimulationRunResponse.model_validate(run).model_dump()
    job = await SimulationJobService(session).get_job_or_raise(job_id)
    payload["job_name"] = job.name
    payload["job_description"] = job.description
    return SimulationRunResponse.model_validate(payload)


__all__ = ["jobs_router"]
