"""
Distributed participant service.

Handles participant connection, approval, and readiness lifecycle.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from app.core import exceptions
from app.models.base import utcnow
from app.models.distributed import (
    DistributedParticipant,
    DistributedParticipantStatus,
    DistributedSessionStatus,
)

if TYPE_CHECKING:
    from app.services.distributed.session_service import DistributedSessionService


class DistributedParticipantService:
    """
    Sub-service for distributed participant lifecycle.
    """


    def __init__(self, session_service: "DistributedSessionService"):
        self.session_service = session_service


    async def get_participant_or_raise(self, participant_id: str) -> DistributedParticipant:
        """
        Get distributed participant by ID or raise not-found.
        """
        participant = await self.session_service.participant_repository.get_participant(participant_id)
        if not participant:
            raise exceptions.ResourceNotFound("Distributed participant not found")
        return participant


    async def list_session_participants(self, session_id: str) -> list[DistributedParticipant]:
        """
        List participants for a distributed session.
        """
        await self.session_service.get_session_or_raise(session_id)
        return await self.session_service.participant_repository.list_participants(session_id)


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
        distributed_session = await self.session_service.get_session_or_raise(session_id)
        if distributed_session.status in {
            DistributedSessionStatus.RUNNING,
            DistributedSessionStatus.FINISHED,
            DistributedSessionStatus.CANCELLED,
        }:
            raise exceptions.ResourceConflict("Session is not accepting new participants")

        latest = await self.session_service.participant_repository.get_latest_participant_by_name(
            session_id=session_id,
            participant_name=participant_name,
        )
        metadata = metadata_json or {}

        if latest and latest.status not in self.session_service.REUSABLE_PARTICIPANT_STATUSES:
            latest.metadata_json = metadata
            latest.last_seen_at = utcnow()
            await self.session_service.participant_repository.update_participant(latest)
            await self.session_service.session.commit()
            await self.session_service.session.refresh(latest)
            return latest

        participant = await self.session_service.participant_repository.create_participant(
            session_id=session_id,
            participant_name=participant_name,
            metadata_json=metadata,
        )
        await self.session_service.session.commit()
        await self.session_service.session.refresh(participant)
        await self.session_service._refresh_session_status(distributed_session)
        return participant


    async def approve_participant(self, participant_id: str) -> DistributedParticipant:
        """
        Approve participant and assign a slot if needed.
        """
        participant = await self.get_participant_or_raise(participant_id)
        distributed_session = await self.session_service.get_session_or_raise(participant.session_id)
        if distributed_session.status in {
            DistributedSessionStatus.RUNNING,
            DistributedSessionStatus.FINISHED,
            DistributedSessionStatus.CANCELLED,
        }:
            raise exceptions.ResourceConflict("Session status does not allow approving participants")

        if participant.status in {DistributedParticipantStatus.REJECTED, DistributedParticipantStatus.CANCELLED}:
            raise exceptions.ResourceConflict("Participant request can no longer be approved")

        job = await self.session_service.get_job_or_raise(distributed_session.job_id)
        participants = await self.session_service.participant_repository.list_participants(distributed_session.id)
        occupied_slots = {
            item.assigned_participant_id
            for item in participants
            if item.assigned_participant_id is not None
            and item.id != participant.id
            and item.status not in self.session_service.REUSABLE_PARTICIPANT_STATUSES
        }
        if participant.assigned_participant_id is None:
            assigned_slot = next(
                (slot for slot in range(job.expected_clients) if slot not in occupied_slots),
                None,
            )
            if assigned_slot is None:
                raise exceptions.ResourceConflict("No available participant slots to approve")
            participant.assigned_participant_id = assigned_slot

        participant.status = DistributedParticipantStatus.APPROVED
        participant.approved_at = utcnow()
        participant.last_seen_at = utcnow()
        await self.session_service.participant_repository.update_participant(participant)
        await self.session_service.session.commit()
        await self.session_service.session.refresh(participant)
        await self.session_service._refresh_session_status(distributed_session)
        return participant


    async def reject_participant(self, participant_id: str) -> DistributedParticipant:
        """
        Reject participant request when status allows it.
        """
        participant = await self.get_participant_or_raise(participant_id)
        distributed_session = await self.session_service.get_session_or_raise(participant.session_id)

        if participant.status in {DistributedParticipantStatus.RUNNING, DistributedParticipantStatus.READY}:
            raise exceptions.ResourceConflict("Cannot reject a ready or running participant")

        participant.status = DistributedParticipantStatus.REJECTED
        participant.last_seen_at = utcnow()
        await self.session_service.participant_repository.update_participant(participant)
        await self.session_service.session.commit()
        await self.session_service.session.refresh(participant)
        await self.session_service._refresh_session_status(distributed_session)
        return participant


    async def cancel_participant(self, participant_id: str) -> DistributedParticipant:
        """
        Cancel participant request when status allows it.
        """
        participant = await self.get_participant_or_raise(participant_id)
        distributed_session = await self.session_service.get_session_or_raise(participant.session_id)
        if participant.status == DistributedParticipantStatus.RUNNING:
            raise exceptions.ResourceConflict("Running participant cannot be cancelled")

        participant.status = DistributedParticipantStatus.CANCELLED
        participant.last_seen_at = utcnow()
        await self.session_service.participant_repository.update_participant(participant)
        await self.session_service.session.commit()
        await self.session_service.session.refresh(participant)
        await self.session_service._refresh_session_status(distributed_session)
        return participant


    async def mark_participant_ready(self, participant_id: str) -> DistributedParticipant:
        """
        Mark approved participant as ready.
        """
        participant = await self.get_participant_or_raise(participant_id)
        distributed_session = await self.session_service.get_session_or_raise(participant.session_id)
        if participant.status not in {DistributedParticipantStatus.APPROVED, DistributedParticipantStatus.READY}:
            raise exceptions.ResourceConflict("Participant must be approved before ready")

        participant.status = DistributedParticipantStatus.READY
        if participant.ready_at is None:
            participant.ready_at = utcnow()
        participant.last_seen_at = utcnow()
        await self.session_service.participant_repository.update_participant(participant)
        await self.session_service.session.commit()
        await self.session_service.session.refresh(participant)
        await self.session_service._refresh_session_status(distributed_session)
        return participant


    async def get_participant(self, participant_id: str, *, touch: bool = True) -> DistributedParticipant:
        """
        Get participant and optionally update heartbeat timestamp.
        """
        participant = await self.get_participant_or_raise(participant_id)
        if touch:
            participant.last_seen_at = utcnow()
            await self.session_service.participant_repository.update_participant(participant)
            await self.session_service.session.commit()
            await self.session_service.session.refresh(participant)
        return participant
