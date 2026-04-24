"""
Repository layer for simulation run-related models.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.simulation import (
    SimulationRun,
    SimulationRunLog,
    SimulationRunResult,
    SimulationRunStatus,
)
from app.models.base import utcnow


class SimulationRunRepository:
    """CURD helpers for simulation run-related models."""

    def __init__(self, session: AsyncSession):
        self.session = session


    async def create_run(self, run: SimulationRun) -> SimulationRun:
        """
        Create a new simulation run.
        """
        self.session.add(run)
        await self.session.flush()
        return run


    async def get_run(self, run_id: str) -> SimulationRun | None:
        """
        Get a simulation run by ID.
        """
        return await self.session.get(SimulationRun, run_id)


    async def list_runs(self, limit: int = 200) -> list[SimulationRun]:
        """
        List simulation runs ordered by creation time.
        """
        stmt = select(SimulationRun).order_by(SimulationRun.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


    async def list_runs_for_job(self, job_id: int) -> list[SimulationRun]:
        """
        List all simulation runs for a specific job.
        """
        stmt = select(SimulationRun).where(SimulationRun.job_id == job_id).order_by(SimulationRun.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


    async def update_run_status(
        self,
        run: SimulationRun,
        *,
        status: SimulationRunStatus,
        process_id: int | None = None,
        exit_code: int | None = None,
        error_message: str | None = None,
        mark_started: bool = False,
        mark_ended: bool = False,
    ) -> SimulationRun:
        """
        Update a simulation run's status and runtime metadata.
        """
        run.status = status
        run.updated_at = utcnow()

        if process_id is not None:
            run.process_id = process_id
        if exit_code is not None:
            run.exit_code = exit_code
        if error_message is not None:
            run.error_message = error_message
        if mark_started:
            run.started_at = utcnow()
        if mark_ended:
            run.ended_at = utcnow()

        self.session.add(run)
        await self.session.flush()
        return run


    async def update_run_metrics(self, run: SimulationRun, metrics_json: Dict[str, Any]) -> SimulationRun:
        """
        Update metrics payload for a simulation run.
        """
        run.metrics_json = metrics_json
        run.updated_at = utcnow()
        self.session.add(run)
        await self.session.flush()
        return run


    async def add_log(self, run_id: str, message: str, level: str = "INFO") -> SimulationRunLog:
        """
        Append a log entry for a simulation run.
        """
        log = SimulationRunLog(run_id=run_id, level=level, message=message)
        self.session.add(log)
        await self.session.flush()
        return log


    async def list_logs(self, run_id: str, limit: int = 500) -> list[SimulationRunLog]:
        """
        List log entries for a simulation run in ascending order.
        """
        stmt = (
            select(SimulationRunLog)
            .where(SimulationRunLog.run_id == run_id)
            .order_by(SimulationRunLog.created_at.asc(), SimulationRunLog.id.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


    async def add_result(
        self,
        run_id: str,
        artifact_type: str,
        path: str,
        metadata_json: Dict[str, Any] | None = None,
    ) -> SimulationRunResult:
        """
        Create a result artifact record for a simulation run.
        """
        result = SimulationRunResult(
            run_id=run_id,
            artifact_type=artifact_type,
            path=path,
            metadata_json=metadata_json or {},
        )
        self.session.add(result)
        await self.session.flush()
        return result


    async def list_results(self, run_id: str) -> list[SimulationRunResult]:
        """
        List result artifacts for a simulation run.
        """
        stmt = select(SimulationRunResult).where(SimulationRunResult.run_id == run_id).order_by(
            SimulationRunResult.created_at.asc()
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


    async def delete_run(self, run_id: str) -> None:
        """
        Delete a simulation run and all associated logs and results.
        """
        await self.session.execute(delete(SimulationRunLog).where(SimulationRunLog.run_id == run_id))
        await self.session.execute(delete(SimulationRunResult).where(SimulationRunResult.run_id == run_id))
        await self.session.execute(delete(SimulationRun).where(SimulationRun.id == run_id))
