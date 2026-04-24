"""
Repository layer for simulation job-related models.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.simulation import (
    SimulationJob,
    SimulationJobStatus,
    SimulationRun,
    SimulationRunLog,
    SimulationRunResult,
)
from app.models.base import utcnow


class SimulationJobRepository:
    """CURD helpers for simulation job-related models."""

    def __init__(self, session: AsyncSession):
        self.session = session


    async def create_job(
        self,
        name: str,
        description: str | None,
        config_json: Dict[str, Any]
    ) -> SimulationJob:
        """
        Create a new simulation job.
        """
        job = SimulationJob(
            name=name,
            description=description,
            status=SimulationJobStatus.READY,
            config_json=config_json,
        )
        self.session.add(job)
        await self.session.flush()
        return job


    async def get_job(self, job_id: int) -> SimulationJob | None:
        """
        Get a simulation job by ID.
        """
        return await self.session.get(SimulationJob, job_id)


    async def get_job_by_name(self, name: str) -> SimulationJob | None:
        """
        Get a simulation job by name.
        """
        stmt = select(SimulationJob).where(func.lower(SimulationJob.name) == name.lower())
        result = await self.session.execute(stmt)
        return result.scalars().first()


    async def list_jobs(self) -> list[SimulationJob]:
        """
        List all simulation jobs.
        """
        stmt = select(SimulationJob).order_by(SimulationJob.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


    async def update_job(
        self,
        job: SimulationJob,
        *,
        name: str | None = None,
        description: str | None = None,
        status: SimulationJobStatus | None = None,
        config_json: Dict[str, Any] | None = None,
    ) -> SimulationJob:
        """
        Update a simulation job's metadata and/or configuration.
        """
        if name is not None:
            job.name = name
        if description is not None:
            job.description = description
        if status is not None:
            job.status = status
        if config_json is not None:
            job.config_json = config_json
        job.updated_at = utcnow()
        self.session.add(job)
        await self.session.flush()
        return job


    async def delete_job(self, job_id: int) -> None:
        """
        Delete a simulation job and all associated runs, logs, and results.
        """
        run_ids_stmt = select(SimulationRun.id).where(SimulationRun.job_id == job_id)
        run_ids_result = await self.session.execute(run_ids_stmt)
        run_ids = list(run_ids_result.scalars().all())

        if run_ids:
            await self.session.execute(delete(SimulationRunLog).where(SimulationRunLog.run_id.in_(run_ids)))
            await self.session.execute(delete(SimulationRunResult).where(SimulationRunResult.run_id.in_(run_ids)))

        await self.session.execute(delete(SimulationRun).where(SimulationRun.job_id == job_id))
        await self.session.execute(delete(SimulationJob).where(SimulationJob.id == job_id))
