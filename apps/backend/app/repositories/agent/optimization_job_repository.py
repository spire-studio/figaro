"""
Repository layer for persistent agent optimization jobs.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.agent import AgentOptimizationJob, AgentOptimizationJobStatus
from app.models.base import utcnow


class AgentOptimizationJobRepository:
    """CRUD helpers for persistent agent optimization jobs."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_job(
        self,
        *,
        task_id: str,
        job_name: str | None,
        goal: str,
        system_mode: str,
        model_name: str | None,
        status: AgentOptimizationJobStatus,
        max_iterations: int,
        snapshot_json: dict[str, Any],
    ) -> AgentOptimizationJob:
        job = AgentOptimizationJob(
            task_id=task_id,
            job_name=job_name,
            goal=goal,
            system_mode=system_mode,
            model_name=model_name,
            status=status,
            max_iterations=max_iterations,
            snapshot_json=snapshot_json,
            current_phase=str(snapshot_json.get("current_phase")) if snapshot_json.get("current_phase") is not None else None,
            current_iteration=int(snapshot_json.get("current_iteration") or 0),
            completed_iterations=int(snapshot_json.get("completed_iterations") or 0),
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_job(self, optimization_job_id: int) -> AgentOptimizationJob | None:
        return await self.session.get(AgentOptimizationJob, optimization_job_id)

    async def get_job_by_task_id(self, task_id: str) -> AgentOptimizationJob | None:
        stmt = select(AgentOptimizationJob).where(AgentOptimizationJob.task_id == task_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_job_by_name(self, job_name: str) -> AgentOptimizationJob | None:
        stmt = select(AgentOptimizationJob).where(func.lower(AgentOptimizationJob.job_name) == job_name.lower())
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_jobs(self) -> list[AgentOptimizationJob]:
        stmt = select(AgentOptimizationJob).order_by(AgentOptimizationJob.updated_at.desc(), AgentOptimizationJob.id.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_job(
        self,
        job: AgentOptimizationJob,
        *,
        job_name: str | None = None,
        status: AgentOptimizationJobStatus | None = None,
        current_phase: str | None = None,
        current_iteration: int | None = None,
        completed_iterations: int | None = None,
        simulation_job_id: int | None = None,
        best_score: float | None = None,
        snapshot_json: dict[str, Any] | None = None,
        error_message: str | None = None,
        finished_at: Any = ...,
    ) -> AgentOptimizationJob:
        if job_name is not None:
            job.job_name = job_name
        if status is not None:
            job.status = status
        if current_phase is not None:
            job.current_phase = current_phase
        if current_iteration is not None:
            job.current_iteration = current_iteration
        if completed_iterations is not None:
            job.completed_iterations = completed_iterations
        if simulation_job_id is not None:
            job.simulation_job_id = simulation_job_id
        if best_score is not None or job.best_score is None:
            job.best_score = best_score
        if snapshot_json is not None:
            job.snapshot_json = snapshot_json
        if error_message is not None and snapshot_json is not None:
            payload = dict(job.snapshot_json)
            payload["error_message"] = error_message
            job.snapshot_json = payload
        if finished_at is not ...:
            job.finished_at = finished_at
        job.updated_at = utcnow()
        self.session.add(job)
        await self.session.flush()
        return job
