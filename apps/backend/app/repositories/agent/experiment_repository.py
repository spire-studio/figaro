"""
Repository layer for agent experiment-related models.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.agent import (
    AgentExperiment,
    AgentExperimentRun,
    AgentExperimentRunLog,
    AgentExperimentRunResult,
    AgentExperimentRunStatus,
    AgentExperimentStatus,
)
from app.models.base import utcnow


class AgentExperimentRepository:
    """CRUD helpers for agent experiment-related models."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ------------------------------------------------------------------
    # Experiment (mirrors SimulationJobRepository)
    # ------------------------------------------------------------------

    async def create_experiment(
        self,
        name: str,
        description: str | None,
        config_json: Dict[str, Any],
    ) -> AgentExperiment:
        """Create a new agent experiment."""
        experiment = AgentExperiment(
            name=name,
            description=description,
            status=AgentExperimentStatus.READY,
            config_json=config_json,
        )
        self.session.add(experiment)
        await self.session.flush()
        return experiment

    async def get_experiment(self, experiment_id: int) -> AgentExperiment | None:
        """Get an agent experiment by ID."""
        return await self.session.get(AgentExperiment, experiment_id)

    async def get_experiment_by_name(self, name: str) -> AgentExperiment | None:
        """Get an agent experiment by name."""
        stmt = select(AgentExperiment).where(func.lower(AgentExperiment.name) == name.lower())
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_experiments(self) -> list[AgentExperiment]:
        """List all agent experiments."""
        stmt = select(AgentExperiment).order_by(AgentExperiment.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_experiment(
        self,
        experiment: AgentExperiment,
        *,
        name: str | None = None,
        description: str | None = None,
        status: AgentExperimentStatus | None = None,
        config_json: Dict[str, Any] | None = None,
    ) -> AgentExperiment:
        """Update an agent experiment's metadata and/or configuration."""
        if name is not None:
            experiment.name = name
        if description is not None:
            experiment.description = description
        if status is not None:
            experiment.status = status
        if config_json is not None:
            experiment.config_json = config_json
        experiment.updated_at = utcnow()
        self.session.add(experiment)
        await self.session.flush()
        return experiment

    async def delete_experiment(self, experiment_id: int) -> None:
        """Delete an agent experiment and all associated runs, logs, and results."""
        run_ids_stmt = select(AgentExperimentRun.id).where(AgentExperimentRun.experiment_id == experiment_id)
        run_ids_result = await self.session.execute(run_ids_stmt)
        run_ids = list(run_ids_result.scalars().all())

        if run_ids:
            await self.session.execute(delete(AgentExperimentRunLog).where(AgentExperimentRunLog.run_id.in_(run_ids)))
            await self.session.execute(delete(AgentExperimentRunResult).where(AgentExperimentRunResult.run_id.in_(run_ids)))

        await self.session.execute(delete(AgentExperimentRun).where(AgentExperimentRun.experiment_id == experiment_id))
        await self.session.execute(delete(AgentExperiment).where(AgentExperiment.id == experiment_id))

    # ------------------------------------------------------------------
    # Run (mirrors SimulationRunRepository)
    # ------------------------------------------------------------------

    async def create_run(self, run: AgentExperimentRun) -> AgentExperimentRun:
        """Create a new agent experiment run."""
        self.session.add(run)
        await self.session.flush()
        return run

    async def get_run(self, run_id: str) -> AgentExperimentRun | None:
        """Get an agent experiment run by ID."""
        return await self.session.get(AgentExperimentRun, run_id)

    async def list_runs(self, limit: int = 200) -> list[AgentExperimentRun]:
        """List agent experiment runs ordered by creation time."""
        stmt = select(AgentExperimentRun).order_by(AgentExperimentRun.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_runs_for_experiment(self, experiment_id: int) -> list[AgentExperimentRun]:
        """List all agent experiment runs for a specific experiment."""
        stmt = (
            select(AgentExperimentRun)
            .where(AgentExperimentRun.experiment_id == experiment_id)
            .order_by(AgentExperimentRun.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_run_status(
        self,
        run: AgentExperimentRun,
        *,
        status: AgentExperimentRunStatus,
        process_id: int | None = None,
        exit_code: int | None = None,
        error_message: str | None = None,
        mark_started: bool = False,
        mark_ended: bool = False,
    ) -> AgentExperimentRun:
        """Update an agent experiment run's status and runtime metadata."""
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

    async def update_run_metrics(self, run: AgentExperimentRun, metrics_json: Dict[str, Any]) -> AgentExperimentRun:
        """Update metrics payload for an agent experiment run."""
        run.metrics_json = metrics_json
        run.updated_at = utcnow()
        self.session.add(run)
        await self.session.flush()
        return run

    async def add_log(self, run_id: str, message: str, level: str = "INFO") -> AgentExperimentRunLog:
        """Append a log entry for an agent experiment run."""
        log = AgentExperimentRunLog(run_id=run_id, level=level, message=message)
        self.session.add(log)
        await self.session.flush()
        return log

    async def list_logs(self, run_id: str, limit: int = 500) -> list[AgentExperimentRunLog]:
        """List log entries for an agent experiment run in ascending order."""
        stmt = (
            select(AgentExperimentRunLog)
            .where(AgentExperimentRunLog.run_id == run_id)
            .order_by(AgentExperimentRunLog.created_at.asc(), AgentExperimentRunLog.id.asc())
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
    ) -> AgentExperimentRunResult:
        """Create a result artifact record for an agent experiment run."""
        result = AgentExperimentRunResult(
            run_id=run_id,
            artifact_type=artifact_type,
            path=path,
            metadata_json=metadata_json or {},
        )
        self.session.add(result)
        await self.session.flush()
        return result

    async def list_results(self, run_id: str) -> list[AgentExperimentRunResult]:
        """List result artifacts for an agent experiment run."""
        stmt = (
            select(AgentExperimentRunResult)
            .where(AgentExperimentRunResult.run_id == run_id)
            .order_by(AgentExperimentRunResult.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete_run(self, run_id: str) -> None:
        """Delete an agent experiment run and all associated logs and results."""
        await self.session.execute(delete(AgentExperimentRunLog).where(AgentExperimentRunLog.run_id == run_id))
        await self.session.execute(delete(AgentExperimentRunResult).where(AgentExperimentRunResult.run_id == run_id))
        await self.session.execute(delete(AgentExperimentRun).where(AgentExperimentRun.id == run_id))
