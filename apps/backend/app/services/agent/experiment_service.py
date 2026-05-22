"""
Service layer for agent experiment management.

Mirrors SimulationJobService and SimulationRunService but operates on
agent-specific tables (agent_experiments, agent_experiment_runs, etc.)
so agent workflow data never pollutes the Simulation page.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import exceptions
from app.core.db import AsyncSessionLocal
from app.models.agent import (
    AgentExperiment,
    AgentExperimentRun,
    AgentExperimentRunLog,
    AgentExperimentRunResult,
    AgentExperimentRunStatus,
    AgentExperimentStatus,
)
from app.repositories.agent import AgentExperimentRepository
from app.services.llm_resources import augment_config_schema_with_llm_resources
from app.services.simulation.compatibility import canonicalize_runtime_config, validate_runtime_config_or_raise

from app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared constants (mirrored from simulation run_service / run_metrics_service)
# ---------------------------------------------------------------------------

RESULT_PATH_PATTERN = re.compile(r"\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230[:\uff1a]\s*(.+)$")
LOG_LEVEL_PREFIX_PATTERN = re.compile(
    r"^\s*(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b[\s:\-]",
    re.IGNORECASE,
)
LOG_LEVEL_INLINE_PATTERN = re.compile(
    r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b",
    re.IGNORECASE,
)
RUN_CONFIG_ARTIFACT = "run_config"
TRAINING_RESULT_ARTIFACT = "training_result"
RESULT_FILE_ENV_KEY = "FIGARO_RESULTS_FILE"
DISABLE_FILE_LOG_ENV_KEY = "FIGARO_DISABLE_FILE_LOG"
TERMINAL_RUN_STATUSES = {
    AgentExperimentRunStatus.SUCCEEDED,
    AgentExperimentRunStatus.FAILED,
    AgentExperimentRunStatus.CANCELLED,
}


# ===================================================================
# AgentExperimentService  (mirrors SimulationJobService)
# ===================================================================

class AgentExperimentService:
    """Application service for agent experiment operations."""

    EXPERIMENT_NAME_CONFLICT_MESSAGE = "Experiment name already exists"
    _SCHEMA_META_KEYS = {"role", "depends_on", "hidden", "ui"}

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = AgentExperimentRepository(session)

    # -- create / update / list / get / copy / delete --------------------

    async def create_experiment(self, name: str, description: str | None) -> AgentExperiment:
        """Create an agent experiment with default normalized config."""
        await self._ensure_unique_name(name)
        try:
            default_config = self.normalize_simulation_config({})
            experiment = await self.repo.create_experiment(
                name=name,
                description=description,
                config_json=default_config,
            )
            await self.session.commit()
            await self.session.refresh(experiment)
            return experiment
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.EXPERIMENT_NAME_CONFLICT_MESSAGE) from exc

    async def update_experiment(
        self,
        experiment_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
        status: AgentExperimentStatus | None = None,
    ) -> AgentExperiment:
        """Update agent experiment metadata."""
        experiment = await self.repo.get_experiment(experiment_id)
        if not experiment:
            raise exceptions.JobNotFound("Experiment not found")

        if name is not None and name != experiment.name:
            await self._ensure_unique_name(name, exclude_id=experiment_id)

        try:
            await self.repo.update_experiment(
                experiment,
                name=name,
                description=description,
                status=status,
            )
            await self.session.commit()
            await self.session.refresh(experiment)
            return experiment
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.EXPERIMENT_NAME_CONFLICT_MESSAGE) from exc

    async def update_experiment_config(self, experiment_id: int, config: dict[str, Any]) -> AgentExperiment:
        """Update agent experiment config after normalization."""
        experiment = await self.repo.get_experiment(experiment_id)
        if not experiment:
            raise exceptions.JobNotFound("Experiment not found")

        normalized = self.normalize_simulation_config(config)
        await self.repo.update_experiment(
            experiment,
            config_json=normalized,
            status=AgentExperimentStatus.READY,
        )
        await self.session.commit()
        await self.session.refresh(experiment)
        return experiment

    async def list_experiments(self) -> list[AgentExperiment]:
        return await self.repo.list_experiments()

    async def get_experiment(self, experiment_id: int) -> AgentExperiment | None:
        return await self.repo.get_experiment(experiment_id)

    async def get_experiment_or_raise(self, experiment_id: int) -> AgentExperiment:
        experiment = await self.repo.get_experiment(experiment_id)
        if not experiment:
            raise exceptions.JobNotFound("Experiment not found")
        return experiment

    async def copy_experiment(self, source_id: int, name: str | None = None) -> AgentExperiment:
        source = await self.repo.get_experiment(source_id)
        if not source:
            raise exceptions.JobNotFound("Source experiment not found")

        cloned_name = name or f"{source.name} (copy)"
        await self._ensure_unique_name(cloned_name)

        try:
            cloned = await self.repo.create_experiment(
                name=cloned_name,
                description=source.description,
                config_json=deepcopy(source.config_json if isinstance(source.config_json, dict) else {}),
            )
            await self.session.commit()
            await self.session.refresh(cloned)
            return cloned
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.EXPERIMENT_NAME_CONFLICT_MESSAGE) from exc

    async def get_experiment_config(self, experiment_id: int) -> dict[str, Any]:
        experiment = await self.get_experiment_or_raise(experiment_id)
        if not isinstance(experiment.config_json, dict):
            raise exceptions.JobConfigNotFound("Experiment config not found")
        return experiment.config_json

    async def delete_experiment(self, experiment_id: int) -> None:
        experiment = await self.repo.get_experiment(experiment_id)
        if not experiment:
            raise exceptions.JobNotFound("Experiment not found")

        runs = await self.repo.list_runs_for_experiment(experiment_id)
        if any(r.status in {AgentExperimentRunStatus.QUEUED, AgentExperimentRunStatus.RUNNING} for r in runs):
            raise exceptions.ActiveRunsConflict("Cannot delete experiment with queued or running runs")

        await self.repo.delete_experiment(experiment_id)
        await self.session.commit()

        for run in runs:
            self._cleanup_run_local_files(run.id)

    # -- config schema / normalization (delegates to same logic) ---------

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        schema_path = cls._config_schema_path()
        if not schema_path.exists():
            raise exceptions.ConfigSchemaNotFound("Config schema not found")
        loaded = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise exceptions.InternalServiceError("Config schema must be a YAML object.")
        return augment_config_schema_with_llm_resources(loaded, cls._project_root())

    @classmethod
    def normalize_simulation_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Public config normalization (same logic as SimulationJobService)."""
        if not isinstance(config, dict):
            raise exceptions.BadRequestError("Config must be a JSON object")
        return cls._normalize_simulation_config(config)

    # -- private helpers (identical to SimulationJobService) --------------

    async def _ensure_unique_name(self, name: str, *, exclude_id: int | None = None) -> None:
        existing = await self.repo.get_experiment_by_name(name)
        if existing and (exclude_id is None or existing.id != exclude_id):
            raise exceptions.JobAlreadyExists(self.EXPERIMENT_NAME_CONFLICT_MESSAGE)

    @classmethod
    def _normalize_simulation_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        schema = cls.get_config_schema()
        defaults = cls._init_defaults_from_schema(schema)
        merged = cls._deep_merge(defaults, config)
        cls._synchronize_num_clients_fields(merged)

        system = merged.get("system")
        if not isinstance(system, dict):
            system = {}
            merged["system"] = system
        system["mode"] = "simulation"
        system["node_role"] = "server"

        canonicalize_runtime_config(merged)
        cls._validate_config_node(merged, schema, path_prefix="")
        validate_runtime_config_or_raise(merged)
        return merged

    @classmethod
    def _synchronize_num_clients_fields(cls, merged: dict[str, Any]) -> None:
        dataset = merged.get("dataset")
        if not isinstance(dataset, dict):
            dataset = {}
            merged["dataset"] = dataset
        federated = merged.get("federated")
        if not isinstance(federated, dict):
            federated = {}
            merged["federated"] = federated

        federated_clients = cls._as_positive_int(federated.get("num_clients"))
        dataset_clients = cls._as_positive_int(dataset.get("num_clients"))
        canonical = federated_clients if federated_clients is not None else dataset_clients
        if canonical is None:
            return
        federated["num_clients"] = canonical
        dataset["num_clients"] = canonical

    @classmethod
    def _validate_config_node(cls, config_node: dict[str, Any], schema_node: dict[str, Any], *, path_prefix: str) -> None:
        allowed_keys = {
            key for key, value in schema_node.items()
            if key not in cls._SCHEMA_META_KEYS and isinstance(value, dict)
        }
        for key in config_node:
            if key not in allowed_keys:
                full_path = f"{path_prefix}.{key}" if path_prefix else key
                raise exceptions.BadRequestError(f"Unknown config field: {full_path}")

        for key, raw_definition in schema_node.items():
            if key in cls._SCHEMA_META_KEYS or not isinstance(raw_definition, dict):
                continue
            if key not in config_node:
                continue
            full_path = f"{path_prefix}.{key}" if path_prefix else key
            value = config_node[key]
            if cls._is_field_definition(raw_definition):
                cls._validate_field(value, raw_definition, full_path)
                continue
            if not isinstance(value, dict):
                raise exceptions.BadRequestError(f"{full_path} must be an object")
            cls._validate_config_node(value, raw_definition, path_prefix=full_path)

    @classmethod
    def _validate_field(cls, value: Any, definition: dict[str, Any], path: str) -> None:
        field_type = definition.get("type")
        if field_type == "text":
            if not isinstance(value, str):
                raise exceptions.BadRequestError(f"{path} must be a string")
        elif field_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise exceptions.BadRequestError(f"{path} must be a number")
        elif field_type == "bool":
            if not isinstance(value, bool):
                raise exceptions.BadRequestError(f"{path} must be a boolean")
        elif field_type == "list_int":
            if not isinstance(value, list):
                raise exceptions.BadRequestError(f"{path} must be a list of integers")
            for item in value:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise exceptions.BadRequestError(f"{path} must be a list of integers")
        elif field_type == "select":
            options = definition.get("options")
            if not isinstance(options, list) or value not in options:
                raise exceptions.BadRequestError(f"{path} has invalid option")

    def _cleanup_run_local_files(self, run_id: str) -> None:
        candidates = [
            self._project_root() / "configs" / "agent_experiment_runs" / f"{run_id}.json",
            self._project_root() / "results" / f"live_results_{run_id}.json",
        ]
        for path in candidates:
            try:
                if path.exists() and path.is_file():
                    path.unlink()
            except OSError:
                continue

    @classmethod
    def _config_schema_path(cls) -> Path:
        return cls._project_root() / "config_schema.yaml"

    @classmethod
    def _init_defaults_from_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in schema.items():
            if key in cls._SCHEMA_META_KEYS:
                continue
            if not isinstance(value, dict):
                continue
            if cls._is_field_definition(value):
                output[key] = cls._default_for_field(value)
            else:
                output[key] = cls._init_defaults_from_schema(value)
        return output

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(base)
        for key, value in override.items():
            base_value = merged.get(key)
            if isinstance(base_value, dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge(base_value, value)
            else:
                merged[key] = deepcopy(value)
        return merged

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[5]

    @staticmethod
    def _is_field_definition(node: dict[str, Any]) -> bool:
        return isinstance(node.get("type"), str)

    @staticmethod
    def _default_for_field(node: dict[str, Any]) -> Any:
        if "default" in node:
            return deepcopy(node["default"])
        field_type = node.get("type")
        if field_type == "bool":
            return False
        if field_type == "number":
            return 0
        if field_type == "list_int":
            return []
        if field_type == "select":
            options = node.get("options")
            if isinstance(options, list) and options:
                return deepcopy(options[0])
            return ""
        return ""

    @staticmethod
    def _as_positive_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, float) and float(value).is_integer() and value > 0:
            return int(value)
        return None

# ===================================================================
# AgentExperimentRunService  (mirrors SimulationRunService)
# ===================================================================

class AgentExperimentRunService:
    """Application service for agent experiment run execution and tracking."""

    _processes: dict[str, asyncio.subprocess.Process] = {}
    _tasks: dict[str, asyncio.Task] = {}

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = AgentExperimentRepository(session)

    # -- public API ------------------------------------------------------

    async def start_run(self, experiment_id: int) -> AgentExperimentRun:
        """Start a run for an experiment using its persisted config."""
        experiment = await self.repo.get_experiment(experiment_id)
        if not experiment:
            raise exceptions.JobNotFound("Experiment not found")
        if not isinstance(experiment.config_json, dict) or not experiment.config_json:
            raise exceptions.JobConfigNotFound("Experiment config not found")
        return await self._start_run_with_config(experiment_id=experiment_id, config_json=experiment.config_json)

    async def rerun_run(self, run_id: str) -> AgentExperimentRun:
        """Create a new run from a previous run's config snapshot."""
        source = await self.repo.get_run(run_id)
        if not source:
            raise exceptions.RunNotFound("Run not found")
        experiment = await self.repo.get_experiment(source.experiment_id)
        if not experiment:
            raise exceptions.JobNotFound("Experiment not found")
        config_json = await self._load_config_from_run(run_id)
        if config_json is None:
            raise exceptions.JobConfigNotFound("Run config not found")
        return await self._start_run_with_config(experiment_id=source.experiment_id, config_json=config_json)

    async def stop_run(self, run_id: str) -> AgentExperimentRun:
        """Request cancellation for queued/running run."""
        run = await self.repo.get_run(run_id)
        if not run:
            raise exceptions.RunNotFound("Run not found")
        if run.status in TERMINAL_RUN_STATUSES:
            return run

        process = self._processes.get(run_id)
        if process and process.returncode is None:
            process.terminate()
            await self.repo.add_log(run_id, "run termination requested", level="WARNING")
            await self.repo.update_run_status(
                run,
                status=AgentExperimentRunStatus.CANCELLED,
                mark_ended=True,
                exit_code=-15,
            )
            await self.session.commit()
            await self.session.refresh(run)
            return run

        if run.status in {AgentExperimentRunStatus.QUEUED, AgentExperimentRunStatus.RUNNING}:
            await self.repo.add_log(run_id, "run cancellation requested without active process handle", level="WARNING")
            await self.repo.update_run_status(
                run,
                status=AgentExperimentRunStatus.CANCELLED,
                mark_ended=True,
                exit_code=-15,
            )
            await self.session.commit()
            await self.session.refresh(run)
        return run

    async def get_run(self, run_id: str) -> AgentExperimentRun:
        run = await self.repo.get_run(run_id)
        if not run:
            raise exceptions.RunNotFound("Run not found")
        return run

    async def list_runs(self, limit: int = 200) -> list[AgentExperimentRun]:
        return await self.repo.list_runs(limit=limit)

    async def get_run_logs(self, run_id: str, limit: int = 500) -> list[AgentExperimentRunLog]:
        await self.get_run(run_id)
        return await self.repo.list_logs(run_id, limit=limit)

    async def get_run_results(self, run_id: str) -> list[AgentExperimentRunResult]:
        await self.get_run(run_id)
        return await self.repo.list_results(run_id)

    async def get_run_metrics(self, run_id: str) -> dict[str, Any]:
        """Get normalized metrics payload for run."""
        run = await self.repo.get_run(run_id)
        if not run:
            raise exceptions.RunNotFound("Run not found")
        # Try live result file
        live_result_path = self._results_dir() / self._build_live_results_filename(run_id)
        loaded = self._load_metrics_file(live_result_path)
        if loaded is not None:
            await self._persist_run_metrics_if_changed(run, loaded)
            return loaded

        # Try result artifacts
        results = await self.repo.list_results(run_id)
        training_results = [r for r in results if r.artifact_type == TRAINING_RESULT_ARTIFACT]
        for artifact in reversed(training_results):
            artifact_path = self._resolve_result_artifact_path(artifact.path)
            loaded = self._load_metrics_file(artifact_path)
            if loaded is not None:
                await self._persist_run_metrics_if_changed(run, loaded)
                return loaded

        if isinstance(run.metrics_json, dict) and run.metrics_json:
            return self._normalize_metrics_payload(run.metrics_json)

        return self._empty_metrics_payload()

    async def _persist_run_metrics_if_changed(self, run: AgentExperimentRun, metrics: dict[str, Any]) -> None:
        current = run.metrics_json if isinstance(run.metrics_json, dict) else {}
        if self._stable_json(current) == self._stable_json(metrics):
            return

        await self.repo.update_run_metrics(run, metrics)
        await self.session.commit()
        await self.session.refresh(run)

    async def delete_run(self, run_id: str) -> None:
        run = await self.get_run(run_id)
        if run.status in {AgentExperimentRunStatus.QUEUED, AgentExperimentRunStatus.RUNNING}:
            raise exceptions.ActiveRunConflict("Cannot delete queued or running run")
        await self.repo.delete_run(run_id)
        await self.session.commit()
        self._cleanup_run_local_files(run_id)
        self._processes.pop(run_id, None)
        self._tasks.pop(run_id, None)

    # -- subprocess lifecycle --------------------------------------------

    async def _start_run_with_config(self, *, experiment_id: int, config_json: dict[str, Any]) -> AgentExperimentRun:
        """Create DB run, spawn subprocess, and initialize runtime tracking."""
        run = AgentExperimentRun(
            experiment_id=experiment_id,
            status=AgentExperimentRunStatus.QUEUED,
            command="",
            config_json=deepcopy(config_json),
            metrics_json=self._empty_metrics_payload(),
        )
        await self.repo.create_run(run)

        config_path = self._write_run_config(run.id, config_json)
        await self.repo.add_result(
            run_id=run.id,
            artifact_type=RUN_CONFIG_ARTIFACT,
            path=str(config_path),
            metadata_json={"source": "run_config"},
        )

        command = self._build_train_command(config_path)
        run.command = " ".join(command)

        live_results_path = self._results_dir() / self._build_live_results_filename(run.id)
        child_env = self._build_run_environment(live_results_path)
        process = await self._spawn_subprocess(command, env=child_env)

        await self.repo.update_run_status(
            run,
            status=AgentExperimentRunStatus.RUNNING,
            process_id=process.pid,
            mark_started=True,
        )
        await self.repo.add_log(run.id, f"run started with pid={process.pid}")
        await self.session.commit()
        await self.session.refresh(run)

        self._processes[run.id] = process
        self._tasks[run.id] = asyncio.create_task(self._watch_process(run.id, process))
        return run

    async def _watch_process(self, run_id: str, process: asyncio.subprocess.Process) -> None:
        """Consume subprocess streams and persist final run status/artifacts."""
        result_path: str | None = None
        log_path = self._run_log_path(run_id)

        async def consume(stream: asyncio.StreamReader, level: str, prefix: str) -> None:
            nonlocal result_path
            with log_path.open("a", encoding="utf-8") as file_obj:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    message = line.decode("utf-8", errors="replace").rstrip()
                    if not message:
                        continue
                    file_obj.write(f"[{prefix}] {message}\\n")
                    file_obj.flush()

                    match = RESULT_PATH_PATTERN.search(message)
                    if match:
                        result_path = match.group(1).strip()

                    async with AsyncSessionLocal() as session:
                        repo = AgentExperimentRepository(session)
                        inferred_level = self._infer_log_level(level, message)
                        await repo.add_log(run_id, message, level=inferred_level)
                        await session.commit()

        stdout = process.stdout or asyncio.StreamReader()
        stderr = process.stderr or asyncio.StreamReader()

        await asyncio.gather(
            consume(stdout, "INFO", "stdout"),
            consume(stderr, "ERROR", "stderr"),
        )
        exit_code = await process.wait()

        async with AsyncSessionLocal() as session:
            repo = AgentExperimentRepository(session)
            run = await repo.get_run(run_id)
            if run:
                status = self._derive_final_status(run.status, exit_code)
                await repo.update_run_status(
                    run,
                    status=status,
                    exit_code=exit_code,
                    mark_ended=run.ended_at is None,
                )
                await repo.add_log(run_id, f"run finished with exit_code={exit_code}")

                metrics_persisted = False
                if result_path:
                    await repo.add_result(
                        run_id=run_id,
                        artifact_type=TRAINING_RESULT_ARTIFACT,
                        path=result_path,
                        metadata_json={"source": "process_output"},
                    )
                    resolved = self._resolve_result_artifact_path(result_path)
                    normalized = self._load_metrics_file(resolved)
                    if normalized is not None:
                        await repo.update_run_metrics(run, normalized)
                        metrics_persisted = True

                if not metrics_persisted:
                    guessed = self._guess_live_result_file_for_run(run)
                    if guessed is not None:
                        await repo.add_result(
                            run_id=run_id,
                            artifact_type=TRAINING_RESULT_ARTIFACT,
                            path=str(guessed),
                            metadata_json={"source": "live_results_fallback"},
                        )
                        normalized = self._load_metrics_file(guessed)
                        if normalized is not None:
                            await repo.update_run_metrics(run, normalized)
            await session.commit()

        self._processes.pop(run_id, None)
        self._tasks.pop(run_id, None)

    # -- path helpers ----------------------------------------------------

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[5]

    def _job_config_dir(self) -> Path:
        path = self._project_root() / "configs" / "agent_experiment_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _results_dir(self) -> Path:
        path = self._project_root() / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _runtime_log_dir(self) -> Path:
        path = self._project_root() / "logs" / "agent_experiment_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_log_path(self, run_id: str) -> Path:
        return self._runtime_log_dir() / f"{run_id}_server.log"

    def _build_live_results_filename(self, run_id: str) -> str:
        return f"live_results_{run_id}.json"

    def _run_config_path(self, run_id: str) -> Path:
        return self._job_config_dir() / f"{run_id}.json"

    def _write_run_config(self, run_id: str, config_json: dict[str, Any]) -> Path:
        path = self._run_config_path(run_id)
        path.write_text(json.dumps(config_json, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _build_run_environment(live_results_path: Path) -> dict[str, str]:
        child_env = os.environ.copy()
        child_env[RESULT_FILE_ENV_KEY] = live_results_path.name
        child_env[DISABLE_FILE_LOG_ENV_KEY] = "1"
        return child_env

    async def _spawn_subprocess(self, command: list[str], env: dict[str, str] | None = None) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self._project_root()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def _load_config_from_run(self, run_id: str) -> dict[str, Any] | None:
        run = await self.repo.get_run(run_id)
        if run and isinstance(run.config_json, dict) and run.config_json:
            return run.config_json

        results = await self.repo.list_results(run_id)
        snapshot = next((r for r in results if r.artifact_type == RUN_CONFIG_ARTIFACT), None)
        if not snapshot:
            return None

        path = Path(snapshot.path)
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(loaded, dict):
            return None
        return loaded

    @staticmethod
    def _derive_final_status(current_status: AgentExperimentRunStatus, exit_code: int) -> AgentExperimentRunStatus:
        if current_status == AgentExperimentRunStatus.CANCELLED:
            return AgentExperimentRunStatus.CANCELLED
        if exit_code in {-15, 143}:
            return AgentExperimentRunStatus.CANCELLED
        return AgentExperimentRunStatus.SUCCEEDED if exit_code == 0 else AgentExperimentRunStatus.FAILED

    def _cleanup_run_local_files(self, run_id: str) -> None:
        for path in [
            self._run_config_path(run_id),
            self._results_dir() / self._build_live_results_filename(run_id),
            self._run_log_path(run_id),
        ]:
            try:
                if path.exists() and path.is_file():
                    path.unlink()
            except OSError:
                continue

    # -- metrics helpers (mirrored from SimulationRunMetricsService) ------

    @staticmethod
    def _empty_metrics_payload() -> dict[str, Any]:
        return {
            "experiment_info": {
                "basic": {},
                "federated": {},
                "security": {},
            },
            "global_results": {
                "rounds": [],
                "global_loss": [],
                "global_accuracy": [],
            },
            "llm_results": {
                "rounds": [],
                "train_loss": [],
                "validation_loss": [],
                "perplexity": [],
                "token_throughput": [],
                "adapter_size_bytes": [],
            },
            "llm_dataset": {},
            "llm_evaluation": {},
            "llm_runtime": {},
            "llm_artifacts": [],
            "client_results": {},
        }

    def _resolve_result_artifact_path(self, artifact_path: str) -> Path:
        path = Path(artifact_path)
        if path.is_absolute():
            return path
        return self._project_root() / path

    def _load_metrics_file(self, file_path: Path) -> dict[str, Any] | None:
        if not file_path.exists() or not file_path.is_file():
            return None
        try:
            loaded = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(loaded, dict):
            return None
        return self._normalize_metrics_payload(loaded)

    def _guess_live_result_file_for_run(self, run: AgentExperimentRun) -> Path | None:
        directory = self._results_dir()
        if not directory.exists():
            return None
        expected = directory / self._build_live_results_filename(run.id)
        if expected.exists() and expected.is_file():
            return expected

        started_at = run.started_at or run.created_at
        if started_at is None:
            return None
        start_ts = started_at.timestamp()
        candidates: list[Path] = []
        for path in directory.glob("live_results_*.json"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= start_ts - 5:
                candidates.append(path)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _normalize_metrics_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._empty_metrics_payload()

        experiment_info = payload.get("experiment_info")
        if isinstance(experiment_info, dict):
            for section in ("basic", "federated", "security"):
                section_payload = experiment_info.get(section)
                if isinstance(section_payload, dict):
                    normalized["experiment_info"][section] = section_payload

        global_results = payload.get("global_results")
        if isinstance(global_results, dict):
            normalized["global_results"] = {
                "rounds": self._to_int_list(global_results.get("rounds")),
                "global_loss": self._to_float_list(global_results.get("global_loss")),
                "global_accuracy": self._to_float_list(global_results.get("global_accuracy")),
            }

        llm_results = payload.get("llm_results")
        if isinstance(llm_results, dict):
            normalized["llm_results"] = {
                "rounds": self._to_int_list(llm_results.get("rounds")),
                "train_loss": self._to_float_list(llm_results.get("train_loss")),
                "validation_loss": self._to_float_list(llm_results.get("validation_loss")),
                "perplexity": self._to_float_list(llm_results.get("perplexity")),
                "token_throughput": self._to_float_list(llm_results.get("token_throughput")),
                "adapter_size_bytes": self._to_int_list(llm_results.get("adapter_size_bytes")),
            }

        llm_dataset = payload.get("llm_dataset")
        if isinstance(llm_dataset, dict):
            normalized["llm_dataset"] = llm_dataset

        llm_evaluation = payload.get("llm_evaluation")
        if isinstance(llm_evaluation, dict):
            normalized["llm_evaluation"] = llm_evaluation

        llm_runtime = payload.get("llm_runtime")
        if isinstance(llm_runtime, dict):
            normalized["llm_runtime"] = llm_runtime

        llm_artifacts = payload.get("llm_artifacts")
        if isinstance(llm_artifacts, list):
            normalized["llm_artifacts"] = [
                artifact for artifact in llm_artifacts if isinstance(artifact, dict)
            ]

        client_results = payload.get("client_results")
        if isinstance(client_results, dict):
            normalized_clients: dict[str, dict[str, list[float]]] = {}
            for client_name, client_data in client_results.items():
                if not isinstance(client_name, str) or not isinstance(client_data, dict):
                    continue
                normalized_clients[client_name] = {
                    "train_loss": self._to_float_list(client_data.get("train_loss")),
                    "train_acc": self._to_float_list(client_data.get("train_acc")),
                    "test_loss": self._to_float_list(client_data.get("test_loss")),
                    "test_acc": self._to_float_list(client_data.get("test_acc")),
                }
            normalized["client_results"] = normalized_clients

        return normalized

    @staticmethod
    def _to_float_list(value: Any) -> list[float]:
        if not isinstance(value, list):
            return []
        output: list[float] = []
        for item in value:
            try:
                output.append(float(item))
            except (TypeError, ValueError):
                continue
        return output

    @staticmethod
    def _to_int_list(value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        output: list[int] = []
        for item in value:
            try:
                output.append(int(item))
            except (TypeError, ValueError):
                continue
        return output

    @staticmethod
    def _stable_json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    # -- log level inference ---------------------------------------------

    @staticmethod
    def _normalize_log_level(level: str) -> str:
        normalized = level.upper()
        if normalized == "WARN":
            return "WARNING"
        return normalized

    @classmethod
    def _infer_log_level(cls, stream_level: str, message: str) -> str:
        prefix_match = LOG_LEVEL_PREFIX_PATTERN.match(message)
        if prefix_match:
            return cls._normalize_log_level(prefix_match.group(1))
        inline_match = LOG_LEVEL_INLINE_PATTERN.search(message)
        if inline_match and " - " in message:
            return cls._normalize_log_level(inline_match.group(1))
        # Ignore stream_level — many Python tools write plain INFO output to
        # stderr. Only promote to ERROR when the content actually looks like one.
        lowered = message.lower()
        if "traceback" in lowered or "exception" in lowered or "error:" in lowered:
            return "ERROR"
        return "INFO"

    @staticmethod
    def _build_train_command(config_path: Path) -> list[str]:
        return [
            sys.executable,
            "-u",
            "apps/backend/runners/experiment_runner.py",
            "--mode",
            "simulation",
            "--config",
            str(config_path),
        ]
