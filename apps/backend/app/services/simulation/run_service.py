"""
Service layer for simulation run lifecycle management.
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

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import exceptions
from app.core.db import AsyncSessionLocal
from app.models.simulation import SimulationRun, SimulationRunLog, SimulationRunResult, SimulationRunStatus
from app.repositories.simulation import SimulationJobRepository, SimulationRunRepository
from app.services.simulation.run_metrics_service import SimulationRunMetricsService

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
    SimulationRunStatus.SUCCEEDED,
    SimulationRunStatus.FAILED,
    SimulationRunStatus.CANCELLED,
}


class SimulationRunService:
    """Application service for simulation run execution and tracking."""

    _processes: dict[str, asyncio.subprocess.Process] = {}
    _tasks: dict[str, asyncio.Task] = {}


    def __init__(self, session: AsyncSession):
        self.session = session
        self.job_repository = SimulationJobRepository(session)
        self.run_repository = SimulationRunRepository(session)
        self.metrics_service = SimulationRunMetricsService(session)


    async def start_run(self, job_id: int) -> SimulationRun:
        """
        Start a simulation run for a job using its persisted config.
        """
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise exceptions.JobNotFound("Job not found")

        if not isinstance(job.config_json, dict) or not job.config_json:
            raise exceptions.JobConfigNotFound("Job config not found")

        return await self._start_run_with_config(job_id=job_id, config_json=job.config_json)


    async def rerun_run(self, run_id: str) -> SimulationRun:
        """
        Create a new run from a previous run's config snapshot.
        """
        source = await self.run_repository.get_run(run_id)
        if not source:
            raise exceptions.RunNotFound("Run not found")

        job = await self.job_repository.get_job(source.job_id)
        if not job:
            raise exceptions.JobNotFound("Job not found")

        config_json = await self._load_config_from_run(run_id)
        if config_json is None:
            raise exceptions.JobConfigNotFound("Run config not found")

        return await self._start_run_with_config(job_id=source.job_id, config_json=config_json)


    async def stop_run(self, run_id: str) -> SimulationRun:
        """
        Request cancellation for queued/running run.
        """
        run = await self.run_repository.get_run(run_id)
        if not run:
            raise exceptions.RunNotFound("Run not found")

        if run.status in TERMINAL_RUN_STATUSES:
            return run

        process = self._processes.get(run_id)
        if process and process.returncode is None:
            process.terminate()
            await self.run_repository.add_log(run_id, "run termination requested", level="WARNING")
            await self.run_repository.update_run_status(
                run,
                status=SimulationRunStatus.CANCELLED,
                mark_ended=True,
                exit_code=-15,
            )
            await self.session.commit()
            await self.session.refresh(run)
            return run

        if run.status in {SimulationRunStatus.QUEUED, SimulationRunStatus.RUNNING}:
            await self.run_repository.add_log(
                run_id,
                "run cancellation requested without active process handle",
                level="WARNING",
            )
            await self.run_repository.update_run_status(
                run,
                status=SimulationRunStatus.CANCELLED,
                mark_ended=True,
                exit_code=-15,
            )
            await self.session.commit()
            await self.session.refresh(run)

        return run


    async def get_run(self, run_id: str) -> SimulationRun:
        """
        Get run by ID or raise not-found.
        """
        run = await self.run_repository.get_run(run_id)
        if not run:
            raise exceptions.RunNotFound("Run not found")
        return run


    async def list_runs(self, limit: int = 200) -> list[SimulationRun]:
        """
        List simulation runs ordered by creation time.
        """
        return await self.run_repository.list_runs(limit=limit)


    async def get_run_logs(self, run_id: str, limit: int = 500) -> list[SimulationRunLog]:
        """
        Get logs for one run.
        """
        await self.get_run(run_id)
        return await self.run_repository.list_logs(run_id, limit=limit)


    async def get_run_results(self, run_id: str) -> list[SimulationRunResult]:
        """
        Get result artifacts for one run.
        """
        await self.get_run(run_id)
        return await self.run_repository.list_results(run_id)


    async def get_run_metrics(self, run_id: str) -> dict[str, Any]:
        """
        Get normalized metrics payload for run.
        """
        return await self.metrics_service.get_run_metrics(run_id)


    async def delete_run(self, run_id: str) -> None:
        """
        Delete run when not active and cleanup local artifacts.
        """
        run = await self.get_run(run_id)
        if run.status in {SimulationRunStatus.QUEUED, SimulationRunStatus.RUNNING}:
            raise exceptions.ActiveRunConflict("Cannot delete queued or running run")

        await self.run_repository.delete_run(run_id)
        await self.session.commit()
        self._cleanup_run_local_files(run_id)
        self._processes.pop(run_id, None)
        self._tasks.pop(run_id, None)


    async def _start_run_with_config(self, *, job_id: int, config_json: dict[str, Any]) -> SimulationRun:
        """
        Create DB run, spawn subprocess, and initialize runtime tracking.
        """
        run = SimulationRun(
            job_id=job_id,
            status=SimulationRunStatus.QUEUED,
            command="",
            config_json=deepcopy(config_json),
            metrics_json=self.metrics_service.empty_metrics_payload(),
        )
        await self.run_repository.create_run(run)

        config_path = self._write_run_config(run.id, config_json)
        await self.run_repository.add_result(
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

        await self.run_repository.update_run_status(
            run,
            status=SimulationRunStatus.RUNNING,
            process_id=process.pid,
            mark_started=True,
        )
        await self.run_repository.add_log(run.id, f"run started with pid={process.pid}")
        await self.session.commit()
        await self.session.refresh(run)

        self._processes[run.id] = process
        self._tasks[run.id] = asyncio.create_task(self._watch_process(run.id, process))
        return run


    async def _watch_process(self, run_id: str, process: asyncio.subprocess.Process) -> None:
        """
        Consume subprocess streams and persist final run status/artifacts.
        """
        result_path: str | None = None
        log_path = self._simulation_log_path(run_id)

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
                        repository = SimulationRunRepository(session)
                        inferred_level = self._infer_log_level(level, message)
                        await repository.add_log(run_id, message, level=inferred_level)
                        await session.commit()

        stdout = process.stdout or asyncio.StreamReader()
        stderr = process.stderr or asyncio.StreamReader()

        await asyncio.gather(
            consume(stdout, "INFO", "stdout"),
            consume(stderr, "ERROR", "stderr"),
        )
        exit_code = await process.wait()

        async with AsyncSessionLocal() as session:
            repository = SimulationRunRepository(session)
            run = await repository.get_run(run_id)
            if run:
                status = self._derive_final_status(run.status, exit_code)
                await repository.update_run_status(
                    run,
                    status=status,
                    exit_code=exit_code,
                    mark_ended=run.ended_at is None,
                )
                await repository.add_log(run_id, f"run finished with exit_code={exit_code}")

                metrics_persisted = False
                if result_path:
                    await repository.add_result(
                        run_id=run_id,
                        artifact_type=TRAINING_RESULT_ARTIFACT,
                        path=result_path,
                        metadata_json={"source": "process_output"},
                    )
                    normalized = self.metrics_service.load_metrics_file(
                        self.metrics_service.resolve_result_artifact_path(result_path)
                    )
                    if normalized is not None:
                        await repository.update_run_metrics(run, normalized)
                        metrics_persisted = True

                if not metrics_persisted:
                    guessed = self.metrics_service.guess_live_result_file_for_run(run)
                    if guessed is not None:
                        await repository.add_result(
                            run_id=run_id,
                            artifact_type=TRAINING_RESULT_ARTIFACT,
                            path=str(guessed),
                            metadata_json={"source": "live_results_fallback"},
                        )
                        normalized = self.metrics_service.load_metrics_file(guessed)
                        if normalized is not None:
                            await repository.update_run_metrics(run, normalized)
            await session.commit()

        self._processes.pop(run_id, None)
        self._tasks.pop(run_id, None)


    @staticmethod
    def _project_root() -> Path:
        """
        Return project root directory.
        """
        return Path(__file__).resolve().parents[5]


    def _job_config_dir(self) -> Path:
        """
        Return simulation run config directory.
        """
        path = self._project_root() / "configs" / "simulation_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path


    def _results_dir(self) -> Path:
        """
        Return simulation run results directory.
        """
        path = self._project_root() / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path


    def _runtime_log_dir(self) -> Path:
        """
        Return simulation runtime log directory.
        """
        path = self._project_root() / "logs" / "simulation_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path


    def _simulation_log_path(self, run_id: str, client_id: int | None = None) -> Path:
        """
        Build simulation log path for server or client role.
        """
        suffix = "server" if client_id is None else f"client_{client_id}"
        return self._runtime_log_dir() / f"{run_id}_{suffix}.log"


    def _build_live_results_filename(self, run_id: str) -> str:
        """
        Build live result filename for one run ID.
        """
        return f"live_results_{run_id}.json"


    def _run_config_path(self, run_id: str) -> Path:
        """
        Build local run config path.
        """
        return self._job_config_dir() / f"{run_id}.json"


    def _write_run_config(self, run_id: str, config_json: dict[str, Any]) -> Path:
        """
        Write run config payload to JSON file.
        """
        path = self._run_config_path(run_id)
        path.write_text(json.dumps(config_json, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


    @staticmethod
    def _build_run_environment(live_results_path: Path) -> dict[str, str]:
        """
        Build child process environment for simulation run.
        """
        child_env = os.environ.copy()
        child_env[RESULT_FILE_ENV_KEY] = live_results_path.name
        child_env[DISABLE_FILE_LOG_ENV_KEY] = "1"
        return child_env


    async def _spawn_subprocess(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
    ) -> asyncio.subprocess.Process:
        """
        Spawn simulation subprocess in project root.
        """
        return await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self._project_root()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )


    async def _load_config_from_run(self, run_id: str) -> dict[str, Any] | None:
        """
        Load config payload from run row or run config artifact file.
        """
        run = await self.run_repository.get_run(run_id)
        if run and isinstance(run.config_json, dict) and run.config_json:
            return run.config_json

        results = await self.run_repository.list_results(run_id)
        snapshot = next((result for result in results if result.artifact_type == RUN_CONFIG_ARTIFACT), None)
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
    def _derive_final_status(current_status: SimulationRunStatus, exit_code: int) -> SimulationRunStatus:
        """
        Derive terminal status from current status and process exit code.
        """
        if current_status == SimulationRunStatus.CANCELLED:
            return SimulationRunStatus.CANCELLED
        if exit_code in {-15, 143}:
            return SimulationRunStatus.CANCELLED
        return SimulationRunStatus.SUCCEEDED if exit_code == 0 else SimulationRunStatus.FAILED


    @staticmethod
    def _delete_file_if_exists(path: Path) -> None:
        """
        Delete file when it exists and is a regular file.
        """
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            return


    def _cleanup_run_local_files(self, run_id: str) -> None:
        """
        Cleanup local config/results/log files for a run ID.
        """
        self._delete_file_if_exists(self._run_config_path(run_id))
        self._delete_file_if_exists(self._results_dir() / self._build_live_results_filename(run_id))
        self._delete_file_if_exists(self._simulation_log_path(run_id))


    @staticmethod
    def _normalize_log_level(level: str) -> str:
        """
        Normalize textual log level.
        """
        normalized = level.upper()
        if normalized == "WARN":
            return "WARNING"
        return normalized


    @classmethod
    def _infer_log_level(cls, stream_level: str, message: str) -> str:
        """
        Infer structured log level from stream label and message content.
        """
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
        """
        Build simulation experiment runner command.
        """
        return [
            sys.executable,
            "-u",
            "apps/backend/runners/experiment_runner.py",
            "--mode",
            "simulation",
            "--config",
            str(config_path),
        ]
