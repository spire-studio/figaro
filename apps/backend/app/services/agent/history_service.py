"""
Persistence service for agent optimization jobs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import exceptions
from app.models.agent import AgentConfigVersion, AgentOptimizationJob, AgentOptimizationJobStatus
from app.repositories.agent import AgentConfigVersionRepository, AgentOptimizationJobRepository

from .memory import compute_config_diff
from .summary import get_agent_run_score


class AgentOptimizationHistoryService:
    """Persist and retrieve agent optimization jobs and their snapshots."""

    JOB_NAME_CONFLICT_MESSAGE = "Agent optimization job name already exists"

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = AgentOptimizationJobRepository(session)
        self.config_version_repository = AgentConfigVersionRepository(session)

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
            safe_snapshot = self._to_json_safe(snapshot)
            job = await self.repository.create_job(
                task_id=task_id,
                job_name=job_name,
                goal=goal,
                system_mode=system_mode,
                model_name=model_name,
                status=resolved_status,
                max_iterations=max_iterations,
                snapshot_json=safe_snapshot,
            )
            await self._sync_config_versions(job, safe_snapshot)
            await self.session.commit()
            await self.session.refresh(job)
            return job
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE) from exc

    async def list_jobs(
        self,
        *,
        status: AgentOptimizationJobStatus | str | None = None,
        q: str | None = None,
        model_name: str | None = None,
        objective: str | None = None,
        best_score_min: float | None = None,
        best_score_max: float | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        config_filters: dict[str, Any] | None = None,
    ) -> list[AgentOptimizationJob]:
        resolved_status = self._resolve_status(status)
        jobs = await self.repository.list_jobs(
            status=resolved_status,
            q=q,
            model_name=model_name,
            best_score_min=best_score_min,
            best_score_max=best_score_max,
            created_from=created_from,
            created_to=created_to,
        )
        return [
            job
            for job in jobs
            if self._matches_snapshot_filters(
                job.snapshot_json if isinstance(job.snapshot_json, dict) else {},
                objective=objective,
                config_filters=config_filters or {},
            )
        ]

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
        best_score = get_agent_run_score(best_metrics) if isinstance(best_metrics, dict) else None

        try:
            safe_snapshot = self._to_json_safe(snapshot)
            await self.repository.update_job(
                job,
                job_name=str(job_name) if job_name is not None else None,
                status=status,
                current_phase=str(snapshot.get("current_phase")) if snapshot.get("current_phase") is not None else None,
                current_iteration=int(snapshot.get("current_iteration") or 0),
                completed_iterations=int(snapshot.get("completed_iterations") or 0),
                simulation_job_id=self._extract_simulation_job_id(snapshot),
                best_score=best_score,
                snapshot_json=safe_snapshot,
                finished_at=finished_at,
            )
            await self._sync_config_versions(job, safe_snapshot)
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

    async def list_config_versions(self, optimization_job_id: int) -> list[AgentConfigVersion]:
        job = await self.get_job_or_raise(optimization_job_id)
        versions = await self.config_version_repository.list_versions(optimization_job_id, source="experiment")
        if versions:
            return versions
        snapshot = job.snapshot_json if isinstance(job.snapshot_json, dict) else {}
        if not self._build_config_version_candidates(snapshot):
            return []
        await self._sync_config_versions(job, snapshot)
        await self.session.commit()
        return await self.config_version_repository.list_versions(optimization_job_id, source="experiment")

    async def get_config_version_or_raise(self, optimization_job_id: int, version_id: int) -> AgentConfigVersion:
        await self.get_job_or_raise(optimization_job_id)
        version = await self.config_version_repository.get_version(version_id)
        if version is None or version.optimization_job_id != optimization_job_id:
            raise exceptions.ResourceNotFound("Agent config version not found")
        return version

    async def diff_config_versions(
        self,
        *,
        optimization_job_id: int,
        from_version_id: int | None,
        to_version_id: int,
    ) -> list[dict[str, Any]]:
        to_version = await self.get_config_version_or_raise(optimization_job_id, to_version_id)
        previous_config = None
        if from_version_id is not None:
            from_version = await self.get_config_version_or_raise(optimization_job_id, from_version_id)
            previous_config = from_version.config_json if isinstance(from_version.config_json, dict) else {}
        changes = compute_config_diff(previous_config, to_version.config_json if isinstance(to_version.config_json, dict) else {})
        return [self._serialize_config_change(change) for change in changes]

    async def _sync_config_versions(self, job: AgentOptimizationJob, snapshot: dict[str, Any]) -> None:
        if job.id is None:
            return
        for candidate in self._build_config_version_candidates(snapshot):
            existing = await self.config_version_repository.find_version(
                optimization_job_id=job.id,
                run_id=candidate["run_id"],
                iteration=candidate["iteration"],
                source=candidate["source"],
                config_hash=candidate["config_hash"],
            )
            if existing is not None:
                continue
            await self.config_version_repository.create_version(
                optimization_job_id=job.id,
                run_id=candidate["run_id"],
                iteration=candidate["iteration"],
                source=candidate["source"],
                label=candidate["label"],
                config_hash=candidate["config_hash"],
                config_json=candidate["config_json"],
                diff_json=candidate["diff_json"],
            )

    @classmethod
    def _build_config_version_candidates(cls, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        previous_config: dict[str, Any] | None = None

        experiments = snapshot.get("experiments")
        if isinstance(experiments, list):
            for index, item in enumerate(experiments, start=1):
                if not isinstance(item, dict):
                    continue
                config = item.get("config")
                if not isinstance(config, dict):
                    continue
                candidates.append(
                    cls._build_config_version_candidate(
                        config=config,
                        previous_config=previous_config,
                        iteration=int(item.get("iteration") or index),
                        source="experiment",
                        run_id=str(item.get("run_id")) if item.get("run_id") is not None else None,
                        label=str(item.get("name") or item.get("plan_summary") or f"Experiment {index}"),
                        diff_json=item.get("config_diff"),
                    )
                )
                previous_config = config

        return candidates

    @classmethod
    def _build_config_version_candidate(
        cls,
        *,
        config: dict[str, Any],
        previous_config: dict[str, Any] | None,
        iteration: int,
        source: str,
        run_id: str | None,
        label: str,
        diff_json: Any,
    ) -> dict[str, Any]:
        safe_config = cls._to_json_safe(config)
        if isinstance(diff_json, list):
            safe_diff = cls._to_json_safe(diff_json)
        else:
            safe_diff = [
                cls._serialize_config_change(change)
                for change in compute_config_diff(previous_config, safe_config)
            ]
        return {
            "run_id": run_id,
            "iteration": iteration,
            "source": source,
            "label": label[:255],
            "config_hash": cls._config_hash(safe_config),
            "config_json": safe_config,
            "diff_json": safe_diff,
        }

    @staticmethod
    def _best_snapshot_experiment(snapshot: dict[str, Any]) -> dict[str, Any] | None:
        experiments = snapshot.get("experiments")
        if not isinstance(experiments, list):
            return None
        best_record = None
        best_score = None
        for item in experiments:
            if not isinstance(item, dict):
                continue
            score = item.get("score")
            if score is None:
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            if best_score is None or numeric_score > best_score:
                best_record = item
                best_score = numeric_score
        return best_record

    @staticmethod
    def _serialize_config_change(change: Any) -> dict[str, Any]:
        if isinstance(change, dict):
            return {
                "path": str(change.get("path", "$")),
                "change_type": str(change.get("change_type", "updated")),
                "old_value": copy.deepcopy(change.get("old_value")),
                "new_value": copy.deepcopy(change.get("new_value")),
            }
        return {
            "path": change.path,
            "change_type": change.change_type,
            "old_value": copy.deepcopy(change.old_value),
            "new_value": copy.deepcopy(change.new_value),
        }

    @staticmethod
    def _config_hash(config: dict[str, Any]) -> str:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _resolve_status(cls, status: AgentOptimizationJobStatus | str | None) -> AgentOptimizationJobStatus | None:
        if status is None:
            return None
        if isinstance(status, AgentOptimizationJobStatus):
            return status
        status_text = str(status).strip().lower()
        if not status_text or status_text == "all":
            return None
        return AgentOptimizationJobStatus(status_text)

    @classmethod
    def _matches_snapshot_filters(
        cls,
        snapshot: dict[str, Any],
        *,
        objective: str | None,
        config_filters: dict[str, Any],
    ) -> bool:
        if objective and str(snapshot.get("objective", "")).lower() != objective.lower():
            return False
        for path, expected in config_filters.items():
            if expected is None or str(expected).strip() == "":
                continue
            if not cls._snapshot_has_config_value(snapshot, path, expected):
                return False
        return True

    @classmethod
    def _snapshot_has_config_value(cls, snapshot: dict[str, Any], path: str, expected: Any) -> bool:
        for config in cls._iter_snapshot_configs(snapshot):
            value = cls._get_value_by_path(config, path)
            if value is not None and cls._values_equal(value, expected):
                return True
        return False

    @classmethod
    def _iter_snapshot_configs(cls, snapshot: dict[str, Any]):
        for key in ("best_config", "config_constraints"):
            value = snapshot.get(key)
            if isinstance(value, dict):
                yield value
        current_experiment = snapshot.get("current_experiment")
        if isinstance(current_experiment, dict) and isinstance(current_experiment.get("config"), dict):
            yield current_experiment["config"]
        for collection_key in ("experiments", "draft_experiments"):
            collection = snapshot.get(collection_key)
            if not isinstance(collection, list):
                continue
            for item in collection:
                if not isinstance(item, dict):
                    continue
                for config_key in ("config", "config_patch"):
                    config = item.get(config_key)
                    if isinstance(config, dict):
                        yield config

    @staticmethod
    def _get_value_by_path(obj: dict[str, Any], path: str) -> Any:
        current: Any = obj
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _values_equal(actual: Any, expected: Any) -> bool:
        if isinstance(actual, (int, float)) and isinstance(expected, str):
            try:
                return float(actual) == float(expected)
            except ValueError:
                return False
        return str(actual).lower() == str(expected).lower()

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
