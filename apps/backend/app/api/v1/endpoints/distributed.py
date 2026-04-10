"""Distributed orchestration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import AsyncSessionDep
from app.schemas import (
    DistributedParticipantConnectRequest,
    DistributedParticipantRuntimeConfigResponse,
    DistributedParticipantResponse,
    DistributedJobConfigUpdateRequest,
    DistributedJobCreateRequest,
    DistributedJobResponse,
    DistributedLocalParticipantStartRequest,
    DistributedRuntimeStatusResponse,
    DistributedSessionCreateRequest,
    DistributedSessionProgressResponse,
    DistributedSessionResponse,
)
from app.services import DistributedJobService, DistributedRuntimeService, DistributedSessionService

distributed_router = APIRouter()


@distributed_router.get(
    "/jobs",
    response_model=list[DistributedJobResponse],
    status_code=status.HTTP_200_OK,
    summary="List Distributed Jobs",
)
async def list_distributed_jobs(session: AsyncSessionDep) -> list[DistributedJobResponse]:
    job_service = DistributedJobService(session)
    jobs = await job_service.list_jobs()
    return [DistributedJobResponse.model_validate(item) for item in jobs]


@distributed_router.post(
    "/jobs",
    response_model=DistributedJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Distributed Job",
)
async def create_distributed_job(
    payload: DistributedJobCreateRequest,
    session: AsyncSessionDep,
) -> DistributedJobResponse:
    job_service = DistributedJobService(session)
    job = await job_service.create_job(
        name=payload.name,
        description=payload.description,
    )
    return DistributedJobResponse.model_validate(job)


@distributed_router.patch(
    "/jobs/{job_id}/config",
    response_model=DistributedJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Distributed Job Config",
)
async def update_distributed_job_config(
    job_id: int,
    payload: DistributedJobConfigUpdateRequest,
    session: AsyncSessionDep,
) -> DistributedJobResponse:
    job_service = DistributedJobService(session)
    job = await job_service.update_job_config(job_id=job_id, config_json=payload.config)
    return DistributedJobResponse.model_validate(job)


@distributed_router.get(
    "/jobs/{job_id}",
    response_model=DistributedJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Distributed Job",
)
async def get_distributed_job(job_id: int, session: AsyncSessionDep) -> DistributedJobResponse:
    job_service = DistributedJobService(session)
    job = await job_service.get_job_or_raise(job_id)
    return DistributedJobResponse.model_validate(job)


@distributed_router.get(
    "/jobs/{job_id}/session",
    response_model=DistributedSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Active Distributed Session For Job",
)
async def get_active_distributed_session_for_job(
    job_id: int,
    session: AsyncSessionDep,
) -> DistributedSessionResponse:
    session_service = DistributedSessionService(session)
    distributed_session = await session_service.get_active_session_for_job_or_raise(job_id)
    return DistributedSessionResponse.model_validate(distributed_session)


@distributed_router.post(
    "/jobs/{job_id}/session",
    response_model=DistributedSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Or Get Distributed Session",
)
async def create_distributed_session(
    job_id: int,
    payload: DistributedSessionCreateRequest,
    session: AsyncSessionDep,
) -> DistributedSessionResponse:
    session_service = DistributedSessionService(session)
    distributed_session = await session_service.create_or_get_session(
        job_id=job_id,
        server_ip=payload.server_ip,
        server_port=payload.server_port,
    )
    return DistributedSessionResponse.model_validate(distributed_session)


@distributed_router.get(
    "/sessions/{session_id}",
    response_model=DistributedSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Distributed Session",
)
async def get_distributed_session(session_id: str, session: AsyncSessionDep) -> DistributedSessionResponse:
    session_service = DistributedSessionService(session)
    distributed_session = await session_service.get_session(session_id)
    return DistributedSessionResponse.model_validate(distributed_session)


@distributed_router.post(
    "/sessions/{session_id}/connect",
    response_model=DistributedParticipantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Request Participant Connection",
)
async def request_participant_connection(
    session_id: str,
    payload: DistributedParticipantConnectRequest,
    session: AsyncSessionDep,
) -> DistributedParticipantResponse:
    session_service = DistributedSessionService(session)
    participant = await session_service.request_connect(
        session_id=session_id,
        participant_name=payload.participant_name,
        metadata_json=payload.metadata_json,
    )
    return DistributedParticipantResponse.model_validate(participant)


@distributed_router.get(
    "/sessions/{session_id}/participants",
    response_model=list[DistributedParticipantResponse],
    status_code=status.HTTP_200_OK,
    summary="List Session Participants",
)
async def list_session_participants(session_id: str, session: AsyncSessionDep) -> list[DistributedParticipantResponse]:
    session_service = DistributedSessionService(session)
    participants = await session_service.list_session_participants(session_id)
    return [DistributedParticipantResponse.model_validate(item) for item in participants]


@distributed_router.get(
    "/participants/{participant_id}",
    response_model=DistributedParticipantResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Distributed Participant",
)
async def get_distributed_participant(participant_id: str, session: AsyncSessionDep) -> DistributedParticipantResponse:
    session_service = DistributedSessionService(session)
    participant = await session_service.get_participant(participant_id, touch=True)
    return DistributedParticipantResponse.model_validate(participant)


@distributed_router.get(
    "/participants/{participant_id}/runtime-config",
    response_model=DistributedParticipantRuntimeConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Distributed Participant Runtime Config",
)
async def get_distributed_participant_runtime_config(
    participant_id: str,
    session: AsyncSessionDep,
) -> DistributedParticipantRuntimeConfigResponse:
    session_service = DistributedSessionService(session)
    payload = await session_service.get_participant_runtime_config(participant_id)
    return DistributedParticipantRuntimeConfigResponse.model_validate(payload)


@distributed_router.post(
    "/participants/{participant_id}/approve",
    response_model=DistributedParticipantResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve Distributed Participant",
)
async def approve_distributed_participant(participant_id: str, session: AsyncSessionDep) -> DistributedParticipantResponse:
    session_service = DistributedSessionService(session)
    participant = await session_service.approve_participant(participant_id)
    return DistributedParticipantResponse.model_validate(participant)


@distributed_router.post(
    "/participants/{participant_id}/reject",
    response_model=DistributedParticipantResponse,
    status_code=status.HTTP_200_OK,
    summary="Reject Distributed Participant",
)
async def reject_distributed_participant(participant_id: str, session: AsyncSessionDep) -> DistributedParticipantResponse:
    session_service = DistributedSessionService(session)
    participant = await session_service.reject_participant(participant_id)
    return DistributedParticipantResponse.model_validate(participant)


@distributed_router.post(
    "/participants/{participant_id}/ready",
    response_model=DistributedParticipantResponse,
    status_code=status.HTTP_200_OK,
    summary="Mark Distributed Participant Ready",
)
async def mark_distributed_participant_ready(participant_id: str, session: AsyncSessionDep) -> DistributedParticipantResponse:
    session_service = DistributedSessionService(session)
    participant = await session_service.mark_participant_ready(participant_id)
    return DistributedParticipantResponse.model_validate(participant)


@distributed_router.post(
    "/participants/{participant_id}/cancel",
    response_model=DistributedParticipantResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel Distributed Participant Request",
)
async def cancel_distributed_participant(participant_id: str, session: AsyncSessionDep) -> DistributedParticipantResponse:
    session_service = DistributedSessionService(session)
    participant = await session_service.cancel_participant(participant_id)
    return DistributedParticipantResponse.model_validate(participant)


@distributed_router.post(
    "/runtime/participant/start",
    response_model=DistributedRuntimeStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Start Local Distributed Participant Runtime",
)
async def start_local_distributed_participant_runtime(
    payload: DistributedLocalParticipantStartRequest,
) -> DistributedRuntimeStatusResponse:
    runtime = DistributedRuntimeService()
    state = await runtime.start_local_participant(
        participant_id=payload.participant_id,
        config_json=payload.config_json,
    )
    return DistributedRuntimeStatusResponse.model_validate(state)


@distributed_router.post(
    "/sessions/{session_id}/start",
    response_model=DistributedSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Start Distributed Session",
)
async def start_distributed_session(session_id: str, session: AsyncSessionDep) -> DistributedSessionResponse:
    session_service = DistributedSessionService(session)
    runtime = DistributedRuntimeService()

    async def _start_server_runtime(distributed_session, _job) -> None:
        config_json = await session_service.get_server_runtime_config(distributed_session.id)
        await runtime.start_server_session(
            session_id=distributed_session.id,
            config_json=config_json,
        )

    distributed_session = await session_service.start_session(
        session_id,
        on_before_mark_running=_start_server_runtime,
    )
    return DistributedSessionResponse.model_validate(distributed_session)


@distributed_router.get(
    "/sessions/{session_id}/progress",
    response_model=DistributedSessionProgressResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Distributed Session Training Progress",
)
async def get_distributed_session_progress(
    session_id: str,
    session: AsyncSessionDep,
) -> DistributedSessionProgressResponse:
    session_service = DistributedSessionService(session)
    await session_service.get_session_or_raise(session_id)
    runtime = DistributedRuntimeService()
    progress = runtime.get_session_progress(session_id=session_id)
    return DistributedSessionProgressResponse.model_validate(progress)


__all__ = ["distributed_router"]
