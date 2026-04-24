"""Run endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import AsyncSessionDep
from app.models.simulation import SimulationRun
from app.schemas import (
    Message,
    SimulationRunLogResponse,
    SimulationRunMetricsResponse,
    SimulationRunResponse,
    SimulationRunResultResponse,
)
from app.services import SimulationJobService, SimulationRunService

runs_router = APIRouter()


async def _to_run_response(run: SimulationRun, session: AsyncSessionDep) -> SimulationRunResponse:
    """Attach job metadata to the run response payload."""
    payload = SimulationRunResponse.model_validate(run).model_dump()
    job = await SimulationJobService(session).get_job(run.job_id)
    payload["job_name"] = job.name if job else None
    payload["job_description"] = job.description if job else None
    return SimulationRunResponse.model_validate(payload)


@runs_router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=list[SimulationRunResponse],
    summary="List Runs",
)
async def list_runs(session: AsyncSessionDep, limit: int = 200) -> list[SimulationRunResponse]:
    """Return runs ordered by creation time."""
    service = SimulationRunService(session)
    runs = await service.list_runs(limit=limit)
    responses: list[SimulationRunResponse] = []
    for run in runs:
        responses.append(await _to_run_response(run, session))
    return responses


@runs_router.get(
    "/{run_id}",
    status_code=status.HTTP_200_OK,
    response_model=SimulationRunResponse,
    summary="Get Run",
)
async def get_run(run_id: str, session: AsyncSessionDep) -> SimulationRunResponse:
    """Return details for a single run."""
    service = SimulationRunService(session)
    run = await service.get_run(run_id)
    return await _to_run_response(run, session)


@runs_router.post(
    "/{run_id}/stop",
    status_code=status.HTTP_200_OK,
    response_model=SimulationRunResponse,
    summary="Stop Run",
)
async def stop_run(run_id: str, session: AsyncSessionDep) -> SimulationRunResponse:
    """Request termination for a running run."""
    service = SimulationRunService(session)
    run = await service.stop_run(run_id)
    return await _to_run_response(run, session)


@runs_router.post(
    "/{run_id}/rerun",
    status_code=status.HTTP_200_OK,
    response_model=SimulationRunResponse,
    summary="Rerun Run",
)
async def rerun_run(run_id: str, session: AsyncSessionDep) -> SimulationRunResponse:
    """Create a new run from an existing run configuration."""
    service = SimulationRunService(session)
    run = await service.rerun_run(run_id)
    return await _to_run_response(run, session)


@runs_router.delete(
    "/{run_id}",
    status_code=status.HTTP_200_OK,
    response_model=Message,
    summary="Delete Run",
)
async def delete_run(run_id: str, session: AsyncSessionDep) -> Message:
    """Delete a run when it is no longer active."""
    service = SimulationRunService(session)
    await service.delete_run(run_id)
    return Message(message="Run deleted")


@runs_router.get(
    "/{run_id}/logs",
    status_code=status.HTTP_200_OK,
    response_model=list[SimulationRunLogResponse],
    summary="List Run Logs",
)
async def list_run_logs(
    run_id: str,
    session: AsyncSessionDep,
    limit: int = 500,
) -> list[SimulationRunLogResponse]:
    """Return log entries for a run."""
    service = SimulationRunService(session)
    logs = await service.get_run_logs(run_id, limit=limit)
    return [SimulationRunLogResponse.model_validate(log) for log in logs]


@runs_router.get(
    "/{run_id}/results",
    status_code=status.HTTP_200_OK,
    response_model=list[SimulationRunResultResponse],
    summary="List Run Results",
)
async def list_run_results(
    run_id: str,
    session: AsyncSessionDep,
) -> list[SimulationRunResultResponse]:
    """Return recorded run result artifacts."""
    service = SimulationRunService(session)
    results = await service.get_run_results(run_id)
    return [SimulationRunResultResponse.model_validate(result) for result in results]


@runs_router.get(
    "/{run_id}/metrics",
    status_code=status.HTTP_200_OK,
    response_model=SimulationRunMetricsResponse,
    summary="Get Run Metrics",
)
async def get_run_metrics(
    run_id: str,
    session: AsyncSessionDep,
) -> SimulationRunMetricsResponse:
    """Return normalized training metrics for a run."""
    service = SimulationRunService(session)
    metrics = await service.get_run_metrics(run_id)
    return SimulationRunMetricsResponse.model_validate(metrics)


__all__ = ["runs_router"]
