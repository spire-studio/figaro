"""
Persistence service for agent optimization jobs.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import exceptions
from app.models.agent import AgentOptimizationJob, AgentOptimizationJobStatus
from app.repositories.agent import AgentOptimizationJobRepository

from .summary import get_last_global_accuracy


class AgentOptimizationHistoryService:
    """Persist and retrieve agent optimization jobs and their snapshots."""

    JOB_NAME_CONFLICT_MESSAGE = "Agent optimization job name already exists"

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = AgentOptimizationJobRepository(session)

    async def create_job(
        self,
        *,
        task_id: str,
        job_name: str | None,
        goal: str,
        system_mode: str,
        model_name: str | None,
        max_iterations: int,
        snapshot: dict[str, Any],
        status: AgentOptimizationJobStatus | str = AgentOptimizationJobStatus.QUEUED,
    ) -> AgentOptimizationJob:
        if job_name:
            await self._ensure_unique_job_name(job_name)

        resolved_status = status
        if isinstance(status, str):
            resolved_status = AgentOptimizationJobStatus(status.lower())

        try:
            job = await self.repository.create_job(
                task_id=task_id,
                job_name=job_name,
                goal=goal,
                system_mode=system_mode,
                model_name=model_name,
                status=resolved_status,
                max_iterations=max_iterations,
                snapshot_json=self._to_json_safe(snapshot),
            )
            await self.session.commit()
            await self.session.refresh(job)
            return job
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE) from exc

    async def list_jobs(self) -> list[AgentOptimizationJob]:
        return await self.repository.list_jobs()

    async def get_job_or_raise(self, optimization_job_id: int) -> AgentOptimizationJob:
        job = await self.repository.get_job(optimization_job_id)
        if job is None:
            raise exceptions.ResourceNotFound("Agent optimization job not found")
        return job

    async def get_job_by_task_id(self, task_id: str) -> AgentOptimizationJob | None:
        return await self.repository.get_job_by_task_id(task_id)

    async def update_job_snapshot(
        self,
        *,
        task_id: str,
        snapshot: dict[str, Any],
    ) -> AgentOptimizationJob | None:
        job = await self.repository.get_job_by_task_id(task_id)
        if job is None:
            return None

        if job_name := snapshot.get("job_name"):
            if job.job_name and job.job_name.lower() != str(job_name).lower():
                await self._ensure_unique_job_name(str(job_name), exclude_job_id=job.id)

        status = AgentOptimizationJobStatus(str(snapshot.get("status", AgentOptimizationJobStatus.QUEUED.value)))
        finished_at = snapshot.get("finished_at") if snapshot.get("finished_at") is not None else ...
        best_metrics = snapshot.get("best_metrics")
        best_score = get_last_global_accuracy(best_metrics) if isinstance(best_metrics, dict) else None

        try:
            await self.repository.update_job(
                job,
                job_name=str(job_name) if job_name is not None else None,
                status=status,
                current_phase=str(snapshot.get("current_phase")) if snapshot.get("current_phase") is not None else None,
                current_iteration=int(snapshot.get("current_iteration") or 0),
                completed_iterations=int(snapshot.get("completed_iterations") or 0),
                simulation_job_id=self._extract_simulation_job_id(snapshot),
                best_score=best_score,
                snapshot_json=self._to_json_safe(snapshot),
                finished_at=finished_at,
            )
            await self.session.commit()
            await self.session.refresh(job)
            return job
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE) from exc

    async def mark_job_failed(
        self,
        *,
        task_id: str,
        snapshot: dict[str, Any],
    ) -> AgentOptimizationJob | None:
        return await self.update_job_snapshot(task_id=task_id, snapshot=snapshot)

    async def _ensure_unique_job_name(self, job_name: str, *, exclude_job_id: int | None = None) -> None:
        existing = await self.repository.get_job_by_name(job_name)
        if existing and (exclude_job_id is None or existing.id != exclude_job_id):
            current_status = existing.status.value if hasattr(existing.status, "value") else str(existing.status)
            if current_status.lower() == "pending_review":
                await self.session.delete(existing)
                await self.session.flush()
            else:
                raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE)
    @staticmethod
    def _extract_simulation_job_id(snapshot: dict[str, Any]) -> int | None:
        current_experiment = snapshot.get("current_experiment")
        if isinstance(current_experiment, dict):
            job_id = current_experiment.get("job_id")
            if isinstance(job_id, int):
                return job_id
        experiments = snapshot.get("experiments")
        if isinstance(experiments, list):
            for item in reversed(experiments):
                if isinstance(item, dict) and isinstance(item.get("job_id"), int):
                    return item["job_id"]
        return None

    @classmethod
    def _to_json_safe(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): cls._to_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._to_json_safe(item) for item in value]
        return copy.deepcopy(value)
