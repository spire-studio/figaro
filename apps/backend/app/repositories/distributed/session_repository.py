"""
Repository layer for distributed session-related models.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.base import utcnow
from app.models.distributed import DistributedSession, DistributedSessionStatus


class DistributedSessionRepository:
    """CRUD helpers for distributed session-related models."""

    def __init__(self, session: AsyncSession):
        self.session = session


    async def create_session(
        self,
        *,
        job_id: int,
        server_ip: str,
        server_port: int,
    ) -> DistributedSession:
        """
        Create a distributed session.
        """
        distributed_session = DistributedSession(
            job_id=job_id,
            server_ip=server_ip,
            server_port=server_port,
        )
        self.session.add(distributed_session)
        await self.session.flush()
        return distributed_session


    async def get_session(self, session_id: str) -> DistributedSession | None:
        """
        Get a distributed session by ID.
        """
        return await self.session.get(DistributedSession, session_id)


    async def get_active_session_for_job(self, job_id: int) -> DistributedSession | None:
        """
        Get the latest active session for a distributed job.
        """
        stmt = (
            select(DistributedSession)
            .where(
                DistributedSession.job_id == job_id,
                DistributedSession.status.in_(
                    (
                        DistributedSessionStatus.WAITING_CLIENTS,
                        DistributedSessionStatus.READY_TO_START,
                        DistributedSessionStatus.RUNNING,
                    )
                ),
            )
            .order_by(DistributedSession.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()


    async def update_session_status(self, distributed_session: DistributedSession, status) -> DistributedSession:
        """
        Update status for a distributed session.
        """
        distributed_session.status = status
        distributed_session.updated_at = utcnow()
        self.session.add(distributed_session)
        await self.session.flush()
        return distributed_session
