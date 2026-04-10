"""
In-memory runtime service for asynchronous bench-mode experiment tasks.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import uuid
from typing import Any

from app.core import exceptions
from app.core.db import AsyncSessionLocal
from app.models.base import utcnow
from app.services.llm import LLMService

from .graph import FederatedAgentGraphBuilder
from .history_service import AgentOptimizationHistoryService
from .objectives import AgentOptimizationObjective, resolve_objective
from .state import AgentState, ConfigChange, ExperimentPlan, ExperimentRecord

logger = logging.getLogger("app")


class AgentRuntimeService:
    """Manage background experiment tasks and their live progress snapshots."""

    _tasks: dict[str, asyncio.Task] = {}
    _snapshots: dict[str, dict[str, Any]] = {}
    _lock = asyncio.Lock()

    async def start_optimization(
        self,
        *,
        goal: str,
        max_iterations: int,
        system_mode: str,
        model_name: str | None,
        job_name: str | None,
        objective: AgentOptimizationObjective,
    ) -> dict[str, Any]:
        """
        Create a background experiment task and return its initial snapshot.
        """
        task_id = uuid.uuid4().hex
        now = utcnow()
        resolved_objective = resolve_objective(
            goal=goal, requested_objective=objective
        )
        snapshot = {
            "optimization_job_id": None,
            "task_id": task_id,
            "status": "queued",
            "goal": goal,
            "job_name": job_name,
            "max_iterations": max_iterations,
            "model_name": model_name,
            "objective": objective.value,
            "resolved_objective": resolved_objective.value,
            "current_phase": "queued",
            "current_iteration": 0,
            "completed_iterations": 0,
            "current_plan": None,
            "current_experiment": None,
            "best_config": None,
            "best_metrics": None,
            "experiments": [],
            "summary_text": None,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
            "finished_at": None,
        }

        persisted_job = await self._create_persisted_job(
            task_id=task_id,
            goal=goal,
            max_iterations=max_iterations,
            system_mode=system_mode,
            model_name=model_name,
            job_name=job_name,
            snapshot=snapshot,
        )
        snapshot["optimization_job_id"] = persisted_job["optimization_job_id"]
        snapshot["created_at"] = persisted_job["created_at"]
        snapshot["updated_at"] = persisted_job["updated_at"]

        async with self._lock:
            self._snapshots[task_id] = snapshot
            task = asyncio.create_task(
                self._run_task(
                    task_id=task_id,
                    goal=goal,
                    max_iterations=max_iterations,
                    system_mode=system_mode,
                    model_name=model_name,
                    job_name=job_name,
                    objective=objective,
                    resolved_objective=resolved_objective,
                )
            )
            task.add_done_callback(lambda finished_task, current_task_id=task_id: self._on_task_done(current_task_id, finished_task))
            self._tasks[task_id] = task

        return self._clone_snapshot(snapshot)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Return the latest snapshot for a task."""
        async with self._lock:
            snapshot = self._snapshots.get(task_id)
            if snapshot is None:
                raise exceptions.ResourceNotFound("Agent experiment task not found")
            return self._clone_snapshot(snapshot)

    async def _run_task(
        self,
        *,
        task_id: str,
        goal: str,
        max_iterations: int,
        system_mode: str,
        model_name: str | None,
        job_name: str | None,
        objective: AgentOptimizationObjective,
        resolved_objective: AgentOptimizationObjective,
    ) -> None:
        """Execute one experiment task in the background."""
        logger.info("agent_task_started task_id=%s", task_id)

        try:
            initial_state = AgentState(
                goal=goal,
                max_iterations=max_iterations,
                system_mode=system_mode,
                model_name=model_name,
                job_name=job_name,
                objective=objective,
                resolved_objective=resolved_objective,
                phase="parsing",
            )
            await self._update_from_state(task_id, initial_state, status="running")
            async with AsyncSessionLocal() as session:
                llm_service = LLMService()
                builder = FederatedAgentGraphBuilder(
                    llm_service=llm_service,
                    session=session,
                    progress_callback=lambda state: self._update_from_state(task_id, state, status="running"),
                )
                graph = builder.build().compile()
                raw_state = await graph.ainvoke(initial_state)  # type: ignore[arg-type]

            if isinstance(raw_state, AgentState):
                final_state = raw_state
            else:
                final_state = AgentState(**raw_state)  # type: ignore[arg-type]

            final_state.phase = "completed"
            await self._update_from_state(task_id, final_state, status="completed")
        except BaseException as exc:
            logger.exception("agent_task_failed task_id=%s error=%s", task_id, exc)
            snapshot_to_persist = None
            async with self._lock:
                snapshot = self._snapshots.get(task_id)
                if snapshot is not None:
                    snapshot.update(
                        {
                            "status": "failed",
                            "job_name": snapshot.get("job_name") or job_name,
                            "current_phase": "failed",
                            "error_message": (
                                "Agent experiment task was cancelled."
                                if isinstance(exc, asyncio.CancelledError)
                                else str(exc)
                            ),
                            "updated_at": utcnow(),
                            "finished_at": utcnow(),
                        }
                    )
                    snapshot_to_persist = self._clone_snapshot(snapshot)
            if snapshot_to_persist is not None:
                await self._persist_snapshot(snapshot_to_persist)
        finally:
            async with self._lock:
                self._tasks.pop(task_id, None)

    def _on_task_done(self, task_id: str, task: asyncio.Task) -> None:
        """Surface task termination issues."""
        try:
            if task.cancelled():
                logger.error("agent_task_cancelled task_id=%s", task_id)
                asyncio.create_task(
                    self._mark_task_failed_if_stale(
                        task_id=task_id,
                        message="Agent experiment task was cancelled before progress was recorded.",
                    )
                )
                return
            exception = task.exception()
            if exception is not None:
                logger.exception("agent_task_done_with_exception task_id=%s error=%s", task_id, exception)
                asyncio.create_task(
                    self._mark_task_failed_if_stale(
                        task_id=task_id,
                        message=str(exception),
                    )
                )
        except Exception as exc:
            logger.exception("agent_task_done_callback_failed task_id=%s error=%s", task_id, exc)

    async def _mark_task_failed_if_stale(self, *, task_id: str, message: str) -> None:
        """Convert a stale queued snapshot into an explicit failure."""
        snapshot_to_persist = None
        async with self._lock:
            snapshot = self._snapshots.get(task_id)
            if snapshot is None:
                return
            if snapshot.get("status") != "queued":
                return
            snapshot.update(
                {
                    "status": "failed",
                    "current_phase": "failed",
                    "error_message": message,
                    "updated_at": utcnow(),
                    "finished_at": utcnow(),
                }
            )
            snapshot_to_persist = self._clone_snapshot(snapshot)

        if snapshot_to_persist is not None:
            await self._persist_snapshot(snapshot_to_persist)

    async def _update_from_state(self, task_id: str, state: AgentState, *, status: str) -> None:
        """Project AgentState into a serializable task snapshot."""
        latest_record = state.history[-1] if state.history else None
        current_experiment = self._build_current_experiment(state, latest_record)

        snapshot_to_persist = None
        async with self._lock:
            snapshot = self._snapshots.get(task_id)
            if snapshot is None:
                return

            snapshot.update(
                {
                    "status": status,
                    "goal": state.goal,
                    "job_name": state.current_job_name or state.job_name,
                    "max_iterations": state.max_iterations,
                    "model_name": state.model_name,
                    "objective": state.objective.value,
                    "resolved_objective": state.resolved_objective.value,
                    "current_phase": state.phase,
                    "current_iteration": state.iteration,
                    "completed_iterations": len(state.history),
                    "current_plan": self._serialize_plan(state.current_plan),
                    "current_experiment": current_experiment,
                    "best_config": None,
                    "best_metrics": None,
                    "experiments": [self._serialize_record(record) for record in state.history],
                    "summary_text": state.summary,
                    "error_message": state.error_message,
                    "updated_at": utcnow(),
                    "finished_at": utcnow() if status in {"completed", "failed"} else None,
                }
            )
            snapshot_to_persist = self._clone_snapshot(snapshot)

        if snapshot_to_persist is not None:
            await self._persist_snapshot(snapshot_to_persist)

    @staticmethod
    def _build_current_experiment(
        state: AgentState,
        latest_record: ExperimentRecord | None,
    ) -> dict[str, Any] | None:
        """Build the live current experiment projection for the frontend."""
        has_current_context = bool(state.current_config) or state.current_job_id is not None or state.current_run_id is not None
        if not has_current_context and state.iteration <= 0:
            return None

        metrics = None
        if latest_record and latest_record.run_id == state.current_run_id:
            metrics = copy.deepcopy(latest_record.metrics)

        return {
            "iteration": state.iteration,
            "phase": state.phase,
            "job_id": state.current_job_id,
            "job_name": state.current_job_name or state.job_name,
            "run_id": state.current_run_id,
            "run_status": state.current_run_status,
            "config": copy.deepcopy(state.current_config) if state.current_config else None,
            "metrics": metrics,
        }

    @staticmethod
    def _serialize_record(record: ExperimentRecord) -> dict[str, Any]:
        """Convert an experiment record into a snapshot-friendly dict."""
        return {
            "iteration": record.iteration,
            "run_id": record.run_id,
            "job_id": record.job_id,
            "config": copy.deepcopy(record.config),
            "metrics": copy.deepcopy(record.metrics),
            "name": record.name,
            "iteration_goal": record.iteration_goal,
            "plan_summary": record.plan_summary,
            "hypothesis": record.hypothesis,
            "rationale": copy.deepcopy(record.rationale),
            "config_patch": {
                k: v
                for k, v in copy.deepcopy(record.config_patch).items()
                if not k.startswith("_")
            },
            "config_diff": [AgentRuntimeService._serialize_config_change(change) for change in record.config_diff],
            "score": record.score,
            "result_summary": record.result_summary,
            "decision": record.decision,
            "lessons_learned": copy.deepcopy(record.lessons_learned),
            "notes": record.notes,
        }

    @staticmethod
    def _serialize_plan(plan: ExperimentPlan | None) -> dict[str, Any] | None:
        """Convert the current plan into a snapshot-friendly dict."""
        if plan is None:
            return None
        return {
            "iteration": plan.iteration,
            "iteration_goal": plan.iteration_goal,
            "name": plan.name,
            "plan_summary": plan.plan_summary,
            "hypothesis": plan.hypothesis,
            "rationale": copy.deepcopy(plan.rationale),
            "config_patch": {
                k: v
                for k, v in copy.deepcopy(plan.config_patch).items()
                if not k.startswith("_")
            },
            "config_diff": [AgentRuntimeService._serialize_config_change(change) for change in plan.config_diff],
        }

    @staticmethod
    def _serialize_config_change(change: ConfigChange) -> dict[str, Any]:
        """Convert one config change into a snapshot-friendly dict."""
        return {
            "path": change.path,
            "change_type": change.change_type,
            "old_value": copy.deepcopy(change.old_value),
            "new_value": copy.deepcopy(change.new_value),
        }

    @staticmethod
    def _clone_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        """Return a defensive copy of a task snapshot."""
        return copy.deepcopy(snapshot)

    async def _create_persisted_job(
        self,
        *,
        task_id: str,
        goal: str,
        max_iterations: int,
        system_mode: str,
        model_name: str | None,
        job_name: str | None,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Create the persistent optimization job row before the background task starts."""
        async with AsyncSessionLocal() as session:
            service = AgentOptimizationHistoryService(session)
            job = await service.create_job(
                task_id=task_id,
                job_name=job_name,
                goal=goal,
                system_mode=system_mode,
                model_name=model_name,
                max_iterations=max_iterations,
                snapshot=snapshot,
            )
            return {
                "optimization_job_id": job.id,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            }

    async def _persist_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Persist the latest snapshot to the optimization job history store."""
        async with AsyncSessionLocal() as session:
            service = AgentOptimizationHistoryService(session)
            await service.update_job_snapshot(
                task_id=str(snapshot["task_id"]),
                snapshot=snapshot,
            )
