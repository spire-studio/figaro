"""
Repository layer for distributed participant-related models.
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.base import utcnow
from app.models.distributed import DistributedParticipant, DistributedParticipantStatus


class DistributedParticipantRepository:
    """CRUD helpers for distributed participant-related models."""

    def __init__(self, session: AsyncSession):
        self.session = session


    async def create_participant(
        self,
        *,
        session_id: str,
        participant_name: str,
        metadata_json: dict,
    ) -> DistributedParticipant:
        """
        Create a distributed participant record.
        """
        participant = DistributedParticipant(
            session_id=session_id,
            participant_name=participant_name,
            metadata_json=metadata_json,
            status=DistributedParticipantStatus.PENDING_APPROVAL,
        )
        self.session.add(participant)
        await self.session.flush()
        return participant


    async def get_participant(self, participant_id: str) -> DistributedParticipant | None:
        """
        Get a distributed participant by ID.
        """
        return await self.session.get(DistributedParticipant, participant_id)


    async def get_latest_participant_by_name(
        self,
        *,
        session_id: str,
        participant_name: str,
    ) -> DistributedParticipant | None:
        """
        Get the latest participant record by participant name in a session.
        """
        stmt = (
            select(DistributedParticipant)
            .where(
                DistributedParticipant.session_id == session_id,
                func.lower(DistributedParticipant.participant_name) == participant_name.lower(),
            )
            .order_by(DistributedParticipant.requested_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()


    async def list_participants(self, session_id: str) -> list[DistributedParticipant]:
        """
        List participants for a distributed session.
        """
        stmt = (
            select(DistributedParticipant)
            .where(DistributedParticipant.session_id == session_id)
            .order_by(DistributedParticipant.requested_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


    async def update_participant(self, participant: DistributedParticipant) -> DistributedParticipant:
        """
        Update a distributed participant record.
        """
        participant.updated_at = utcnow()
        self.session.add(participant)
        await self.session.flush()
        return participant
