"""
Repository layer for distributed job-related models.
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.base import utcnow
from app.models.distributed import DistributedJob


class DistributedJobRepository:
    """CRUD helpers for distributed job-related models."""

    def __init__(self, session: AsyncSession):
        self.session = session


    async def create_job(
        self,
        *,
        name: str,
        description: str | None,
        expected_clients: int,
        config_json: dict,
    ) -> DistributedJob:
        """
        Create a distributed job.
        """
        job = DistributedJob(
            name=name,
            description=description,
            expected_clients=expected_clients,
            config_json=config_json,
        )
        self.session.add(job)
        await self.session.flush()
        return job


    async def get_job(self, job_id: int) -> DistributedJob | None:
        """
        Get a distributed job by ID.
        """
        return await self.session.get(DistributedJob, job_id)


    async def get_job_by_name(self, name: str) -> DistributedJob | None:
        """
        Get a distributed job by name.
        """
        stmt = select(DistributedJob).where(func.lower(DistributedJob.name) == name.lower())
        result = await self.session.execute(stmt)
        return result.scalars().first()


    async def list_jobs(self) -> list[DistributedJob]:
        """
        List distributed jobs ordered by creation time.
        """
        stmt = select(DistributedJob).order_by(DistributedJob.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


    async def update_job_status(self, job: DistributedJob, status) -> DistributedJob:
        """
        Update status for a distributed job.
        """
        job.status = status
        job.updated_at = utcnow()
        self.session.add(job)
        await self.session.flush()
        return job


    async def update_job_config(
        self,
        job: DistributedJob,
        *,
        config_json: dict,
        expected_clients: int,
    ) -> DistributedJob:
        """
        Update runtime config and expected client count for a distributed job.
        """
        job.config_json = config_json
        job.expected_clients = expected_clients
        job.updated_at = utcnow()
        self.session.add(job)
        await self.session.flush()
        return job
