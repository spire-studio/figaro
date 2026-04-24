"""
Distributed session service.

Handles distributed sessions, participant lifecycle, and runtime config payloads.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import exceptions
from app.models.base import utcnow
from app.models.distributed import (
    DistributedJob,
    DistributedJobStatus,
    DistributedParticipant,
    DistributedParticipantStatus,
    DistributedSession,
    DistributedSessionStatus,
)
from app.repositories.distributed import (
    DistributedJobRepository,
    DistributedParticipantRepository,
    DistributedSessionRepository,
)
from app.services.distributed.config_service import DistributedConfigService
from app.services.distributed.participant_service import DistributedParticipantService


class DistributedSessionService:
    """
    Application service for distributed session orchestration.
    """

    REUSABLE_PARTICIPANT_STATUSES = {
        DistributedParticipantStatus.REJECTED,
        DistributedParticipantStatus.CANCELLED,
        DistributedParticipantStatus.DISCONNECTED,
    }


    def __init__(self, session: AsyncSession):
        self.session = session
        self.job_repository = DistributedJobRepository(session)
        self.session_repository = DistributedSessionRepository(session)
        self.participant_repository = DistributedParticipantRepository(session)
        self.config_service = DistributedConfigService()
        self.participant_service = DistributedParticipantService(self)


    async def create_or_get_session(
        self,
        *,
        job_id: int,
        server_ip: str,
        server_port: int,
    ) -> DistributedSession:
        """
        Create a new active session or return existing active one for job.
        """
        job = await self.get_job_or_raise(job_id)

        active = await self.session_repository.get_active_session_for_job(job_id)
        if active:
            return active

        distributed_session = await self.session_repository.create_session(
            job_id=job_id,
            server_ip=server_ip,
            server_port=server_port,
        )
        await self.job_repository.update_job_status(job, DistributedJobStatus.WAITING_CLIENTS)
        await self.session.commit()
        await self.session.refresh(distributed_session)
        return distributed_session


    async def get_active_session_for_job_or_raise(self, job_id: int) -> DistributedSession:
        """
        Get active session for a distributed job or raise not-found.
        """
        await self.get_job_or_raise(job_id)
        active = await self.session_repository.get_active_session_for_job(job_id)
        if not active:
            raise exceptions.ResourceNotFound("No active distributed session for this job")
        return active


    async def get_session_or_raise(self, session_id: str) -> DistributedSession:
        """
        Get distributed session by ID or raise not-found.
        """
        distributed_session = await self.session_repository.get_session(session_id)
        if not distributed_session:
            raise exceptions.ResourceNotFound("Distributed session not found")
        return distributed_session


    async def get_session(self, session_id: str) -> DistributedSession:
        """
        Get session with refreshed readiness status.
        """
        distributed_session = await self.get_session_or_raise(session_id)
        return await self._refresh_session_status(distributed_session)


    async def get_job_or_raise(self, job_id: int) -> DistributedJob:
        """
        Get distributed job by ID or raise not-found.
        """
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise exceptions.ResourceNotFound("Distributed job not found")
        return job


    async def get_server_runtime_config(self, session_id: str) -> dict[str, Any]:
        """
        Build runtime config for server node.
        """
        distributed_session = await self.get_session_or_raise(session_id)
        job = await self.get_job_or_raise(distributed_session.job_id)
        return self.config_service.build_server_runtime_config(
            job.config_json,
            server_ip=distributed_session.server_ip,
            server_port=distributed_session.server_port,
            expected_clients=job.expected_clients,
        )


    async def get_participant_runtime_config(self, participant_id: str) -> dict[str, Any]:
        """
        Build runtime config for a participant node.
        """
        participant = await self.get_participant_or_raise(participant_id)
        if participant.assigned_participant_id is None:
            raise exceptions.ResourceConflict("Participant must be approved before runtime config is available")

        distributed_session = await self.get_session_or_raise(participant.session_id)
        job = await self.get_job_or_raise(distributed_session.job_id)
        config_json = self.config_service.build_participant_runtime_config(
            job.config_json,
            server_ip=distributed_session.server_ip,
            server_port=distributed_session.server_port,
            expected_clients=job.expected_clients,
            assigned_participant_id=participant.assigned_participant_id,
        )
        return {
            "participant_id": participant.id,
            "session_id": distributed_session.id,
            "assigned_participant_id": participant.assigned_participant_id,
            "server_ip": distributed_session.server_ip,
            "server_port": distributed_session.server_port,
            "config_json": config_json,
        }


    async def start_session(
        self,
        session_id: str,
        *,
        on_before_mark_running: Callable[[DistributedSession, DistributedJob], Awaitable[None]] | None = None,
    ) -> DistributedSession:
        """
        Start distributed session after all required participants are ready.
        """
        distributed_session = await self.get_session_or_raise(session_id)
        job = await self.get_job_or_raise(distributed_session.job_id)
        participants = await self.participant_repository.list_participants(distributed_session.id)

        ready_participants = [
            item
            for item in participants
            if item.status == DistributedParticipantStatus.READY and item.assigned_participant_id is not None
        ]
        ready_slots = {item.assigned_participant_id for item in ready_participants}
        if len(ready_slots) < job.expected_clients:
            raise exceptions.ResourceConflict("Not all required participants are ready")

        if on_before_mark_running is not None:
            await on_before_mark_running(distributed_session, job)

        distributed_session.status = DistributedSessionStatus.RUNNING
        distributed_session.started_at = utcnow()
        await self.session_repository.update_session_status(distributed_session, DistributedSessionStatus.RUNNING)

        for item in ready_participants:
            item.status = DistributedParticipantStatus.RUNNING
            item.last_seen_at = utcnow()
            await self.participant_repository.update_participant(item)

        await self.job_repository.update_job_status(job, DistributedJobStatus.RUNNING)
        await self.session.commit()
        await self.session.refresh(distributed_session)
        return distributed_session


    async def get_participant_or_raise(self, participant_id: str) -> DistributedParticipant:
        """
        Get distributed participant by ID or raise not-found.
        """
        return await self.participant_service.get_participant_or_raise(participant_id)


    async def list_session_participants(self, session_id: str) -> list[DistributedParticipant]:
        """
        List participants for a distributed session.
        """
        return await self.participant_service.list_session_participants(session_id)


    async def request_connect(
        self,
        *,
        session_id: str,
        participant_name: str,
        metadata_json: dict[str, Any] | None = None,
    ) -> DistributedParticipant:
        """
        Create or refresh a participant connection request.
        """
        return await self.participant_service.request_connect(
            session_id=session_id,
            participant_name=participant_name,
            metadata_json=metadata_json,
        )


    async def approve_participant(self, participant_id: str) -> DistributedParticipant:
        """
        Approve participant and assign a slot if needed.
        """
        return await self.participant_service.approve_participant(participant_id)


    async def reject_participant(self, participant_id: str) -> DistributedParticipant:
        """
        Reject participant request when status allows it.
        """
        return await self.participant_service.reject_participant(participant_id)


    async def cancel_participant(self, participant_id: str) -> DistributedParticipant:
        """
        Cancel participant request when status allows it.
        """
        return await self.participant_service.cancel_participant(participant_id)


    async def mark_participant_ready(self, participant_id: str) -> DistributedParticipant:
        """
        Mark approved participant as ready.
        """
        return await self.participant_service.mark_participant_ready(participant_id)


    async def get_participant(self, participant_id: str, *, touch: bool = True) -> DistributedParticipant:
        """
        Get participant and optionally update heartbeat timestamp.
        """
        return await self.participant_service.get_participant(participant_id, touch=touch)


    async def _refresh_session_status(self, distributed_session: DistributedSession) -> DistributedSession:
        """
        Recompute and persist session status based on participant readiness.
        """
        if distributed_session.status in {
            DistributedSessionStatus.RUNNING,
            DistributedSessionStatus.FINISHED,
            DistributedSessionStatus.CANCELLED,
        }:
            return distributed_session

        job = await self.get_job_or_raise(distributed_session.job_id)
        participants = await self.participant_repository.list_participants(distributed_session.id)
        approved_or_ready_or_running = [
            item
            for item in participants
            if item.status
            in {
                DistributedParticipantStatus.APPROVED,
                DistributedParticipantStatus.READY,
                DistributedParticipantStatus.RUNNING,
            }
            and item.assigned_participant_id is not None
        ]
        ready_or_running = [
            item
            for item in approved_or_ready_or_running
            if item.status in {DistributedParticipantStatus.READY, DistributedParticipantStatus.RUNNING}
        ]

        unique_assigned = {item.assigned_participant_id for item in approved_or_ready_or_running}
        unique_ready = {item.assigned_participant_id for item in ready_or_running}

        target_status = (
            DistributedSessionStatus.READY_TO_START
            if len(unique_assigned) >= job.expected_clients and len(unique_ready) >= job.expected_clients
            else DistributedSessionStatus.WAITING_CLIENTS
        )
        if distributed_session.status != target_status:
            await self.session_repository.update_session_status(distributed_session, target_status)
            await self.session.commit()
            await self.session.refresh(distributed_session)
        return distributed_session

