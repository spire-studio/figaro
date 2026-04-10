"""Runtime service for distributed server/participant processes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core import exceptions
from app.core.db import AsyncSessionLocal
from app.models.base import utcnow
from app.models.distributed import DistributedJobStatus, DistributedParticipantStatus, DistributedSessionStatus
from app.repositories.distributed import (
    DistributedJobRepository,
    DistributedParticipantRepository,
    DistributedSessionRepository,
)

RESULT_FILE_ENV_KEY = "PHOENIX_RESULTS_FILE"
DISABLE_FILE_LOG_ENV_KEY = "PHOENIX_DISABLE_FILE_LOG"


class DistributedRuntimeService:
    """Manage distributed runtime subprocesses and runtime state snapshots."""

    _processes: dict[str, asyncio.subprocess.Process] = {}
    _tasks: dict[str, asyncio.Task] = {}
    _states: dict[str, dict[str, Any]] = {}


    async def start_server_session(self, *, session_id: str, config_json: dict[str, Any]) -> dict[str, Any]:
        """
        Start local distributed server runtime for a session.
        """
        key = self._runtime_key("server", session_id)
        existing = self._processes.get(key)
        if existing and existing.returncode is None:
            current = self._states.get(key)
            if current:
                return self._clone_state(current)

        total_rounds = self._to_int(
            ((config_json.get("federated") or {}).get("num_rounds")) if isinstance(config_json, dict) else 0,
            default=0,
        )
        config_path = self._runtime_config_dir() / f"server_{session_id}.json"
        log_path = self._runtime_log_dir() / self._runtime_log_filename(role="server", runtime_id=session_id)
        results_filename = f"distributed_session_{session_id}.json"
        results_path = self._results_dir() / results_filename
        self._write_config(config_path, config_json)

        env = os.environ.copy()
        env[RESULT_FILE_ENV_KEY] = results_filename
        env[DISABLE_FILE_LOG_ENV_KEY] = "1"
        command = self._build_command(config_path, role="server")
        process = await self._spawn_subprocess(command, env=env)

        state = self._build_state(
            role="server",
            runtime_id=session_id,
            config_path=config_path,
            log_path=log_path,
            results_path=results_path,
            total_rounds=total_rounds,
        )
        state["pid"] = process.pid
        state["started_at"] = utcnow()
        state["status"] = "running"

        self._states[key] = state
        self._processes[key] = process
        self._tasks[key] = asyncio.create_task(self._watch_runtime(key, process))
        return self._clone_state(state)


    async def start_local_participant(
        self,
        *,
        participant_id: str,
        config_json: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Start local distributed participant runtime.
        """
        key = self._runtime_key("client", participant_id)
        existing = self._processes.get(key)
        if existing and existing.returncode is None:
            current = self._states.get(key)
            if current:
                return self._clone_state(current)

        config_path = self._runtime_config_dir() / f"client_{participant_id}.json"
        log_path = self._runtime_log_dir() / self._runtime_log_filename(role="client", runtime_id=participant_id)
        self._write_config(config_path, config_json)

        command = self._build_command(config_path, role="client")
        env = os.environ.copy()
        env[DISABLE_FILE_LOG_ENV_KEY] = "1"
        process = await self._spawn_subprocess(command, env=env)

        state = self._build_state(
            role="client",
            runtime_id=participant_id,
            config_path=config_path,
            log_path=log_path,
            results_path=None,
            total_rounds=0,
        )
        state["pid"] = process.pid
        state["started_at"] = utcnow()
        state["status"] = "running"

        self._states[key] = state
        self._processes[key] = process
        self._tasks[key] = asyncio.create_task(self._watch_runtime(key, process))
        return self._clone_state(state)


    def get_session_progress(self, *, session_id: str) -> dict[str, Any]:
        """
        Get aggregated progress and metrics for a running server session.
        """
        key = self._runtime_key("server", session_id)
        state = self._states.get(key)
        if not state:
            return {
                "session_id": session_id,
                "status": "idle",
                "pid": None,
                "started_at": None,
                "ended_at": None,
                "exit_code": None,
                "total_rounds": 0,
                "last_round": 0,
                "latest_global_loss": None,
                "latest_global_accuracy": None,
                "metrics_json": self._empty_metrics_payload(),
                "log_path": None,
            }

        metrics = self._load_metrics(Path(state["results_path"]) if state.get("results_path") else None)
        global_results = metrics.get("global_results") if isinstance(metrics, dict) else {}
        rounds = global_results.get("rounds") if isinstance(global_results, dict) else []
        losses = global_results.get("global_loss") if isinstance(global_results, dict) else []
        accuracies = global_results.get("global_accuracy") if isinstance(global_results, dict) else []

        last_round = self._to_int(rounds[-1], default=0) if isinstance(rounds, list) and rounds else 0

        latest_global_loss = None
        if isinstance(losses, list) and losses:
            try:
                latest_global_loss = float(losses[-1])
            except (TypeError, ValueError):
                latest_global_loss = None

        latest_global_accuracy = None
        if isinstance(accuracies, list) and accuracies:
            try:
                latest_global_accuracy = float(accuracies[-1])
            except (TypeError, ValueError):
                latest_global_accuracy = None

        return {
            "session_id": session_id,
            "status": state["status"],
            "pid": state["pid"],
            "started_at": state["started_at"],
            "ended_at": state["ended_at"],
            "exit_code": state["exit_code"],
            "total_rounds": self._to_int(state.get("total_rounds"), default=0),
            "last_round": last_round,
            "latest_global_loss": latest_global_loss,
            "latest_global_accuracy": latest_global_accuracy,
            "metrics_json": metrics,
            "log_path": state.get("log_path"),
        }


    def get_local_participant_runtime(self, *, participant_id: str) -> dict[str, Any]:
        """
        Get runtime state for a local participant process.
        """
        key = self._runtime_key("client", participant_id)
        state = self._states.get(key)
        if not state:
            raise exceptions.ResourceNotFound("Local participant runtime not found")
        return self._clone_state(state)


    async def _watch_runtime(self, key: str, process: asyncio.subprocess.Process) -> None:
        """
        Monitor process streams and finalize in-memory runtime state.
        """
        state = self._states.get(key)
        if not state:
            return

        log_path = Path(state["log_path"])
        await asyncio.gather(
            self._consume_stream(process.stdout, log_path=log_path, prefix="stdout"),
            self._consume_stream(process.stderr, log_path=log_path, prefix="stderr"),
        )
        exit_code = await process.wait()

        state["exit_code"] = exit_code
        state["ended_at"] = utcnow()
        state["status"] = "finished" if exit_code == 0 else "failed"
        self._processes.pop(key, None)
        self._tasks.pop(key, None)

        if state["role"] == "server":
            await self._mark_server_session_terminal(state["runtime_id"], exit_code=exit_code)


    async def _mark_server_session_terminal(self, session_id: str, *, exit_code: int) -> None:
        """
        Persist terminal state transitions for session/job/participants.
        """
        async with AsyncSessionLocal() as session:
            job_repository = DistributedJobRepository(session)
            participant_repository = DistributedParticipantRepository(session)
            session_repository = DistributedSessionRepository(session)

            distributed_session = await session_repository.get_session(session_id)
            if not distributed_session:
                return

            if exit_code == 0:
                target_status = DistributedSessionStatus.FINISHED
                target_job_status = DistributedJobStatus.FINISHED
            else:
                target_status = DistributedSessionStatus.CANCELLED
                target_job_status = DistributedJobStatus.WAITING_CLIENTS

            await session_repository.update_session_status(distributed_session, target_status)
            distributed_session.ended_at = utcnow()

            job = await job_repository.get_job(distributed_session.job_id)
            if job:
                await job_repository.update_job_status(job, target_job_status)

            participants = await participant_repository.list_participants(distributed_session.id)
            for participant in participants:
                if participant.status == DistributedParticipantStatus.RUNNING:
                    participant.status = DistributedParticipantStatus.DISCONNECTED
                    participant.last_seen_at = utcnow()
                    await participant_repository.update_participant(participant)

            await session.commit()


    @staticmethod
    def _project_root() -> Path:
        """
        Return project root directory.
        """
        return Path(__file__).resolve().parents[5]


    @staticmethod
    def _resolve_writable_dir(preferred: Path, fallback: Path) -> Path:
        """
        Resolve writable runtime directory with fallback.
        """
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            if os.access(preferred, os.W_OK | os.X_OK):
                return preferred
        except OSError:
            pass

        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


    def _runtime_config_dir(self) -> Path:
        """
        Get writable distributed runtime config directory.
        """
        preferred = self._project_root() / "configs" / "distributed_runs"
        fallback = Path("/tmp") / "phoenix" / "configs" / "distributed_runs"
        return self._resolve_writable_dir(preferred, fallback)


    def _runtime_log_dir(self) -> Path:
        """
        Get writable distributed runtime log directory.
        """
        preferred = self._project_root() / "logs" / "distributed_runs"
        fallback = Path("/tmp") / "phoenix" / "logs" / "distributed_runs"
        return self._resolve_writable_dir(preferred, fallback)


    def _results_dir(self) -> Path:
        """
        Get writable distributed result directory.
        """
        preferred = self._project_root() / "results"
        fallback = Path("/tmp") / "phoenix" / "results"
        return self._resolve_writable_dir(preferred, fallback)


    async def _spawn_subprocess(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
    ) -> asyncio.subprocess.Process:
        """
        Spawn runtime subprocess with project-root working directory.
        """
        return await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self._project_root()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )


    async def _consume_stream(self, stream: asyncio.StreamReader | None, *, log_path: Path, prefix: str) -> None:
        """
        Consume subprocess stream and append lines to runtime log file.
        """
        if stream is None:
            return

        with log_path.open("a", encoding="utf-8") as file_obj:
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                file_obj.write(f"[{prefix}] {line}\n")
                file_obj.flush()


    def _write_config(self, path: Path, config_json: dict[str, Any]) -> None:
        """
        Write runtime config payload to JSON file.
        """
        path.write_text(json.dumps(config_json, ensure_ascii=False, indent=2), encoding="utf-8")


    def _build_state(
        self,
        *,
        role: str,
        runtime_id: str,
        config_path: Path,
        log_path: Path,
        results_path: Path | None,
        total_rounds: int,
    ) -> dict[str, Any]:
        """
        Build runtime state payload for in-memory tracking.
        """
        return {
            "role": role,
            "runtime_id": runtime_id,
            "status": "starting",
            "pid": None,
            "started_at": None,
            "ended_at": None,
            "exit_code": None,
            "config_path": str(config_path),
            "log_path": str(log_path),
            "results_path": str(results_path) if results_path else None,
            "total_rounds": total_rounds,
        }


    def _load_metrics(self, file_path: Path | None) -> dict[str, Any]:
        """
        Load metrics payload from result file or return empty payload.
        """
        if file_path is None or not file_path.exists() or not file_path.is_file():
            return self._empty_metrics_payload()

        try:
            loaded = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_metrics_payload()

        if not isinstance(loaded, dict):
            return self._empty_metrics_payload()
        return loaded


    @staticmethod
    def _runtime_key(role: str, runtime_id: str) -> str:
        """
        Build in-memory runtime dictionary key.
        """
        return f"{role}:{runtime_id}"


    @staticmethod
    def _runtime_log_filename(*, role: str, runtime_id: str) -> str:
        """
        Build runtime log filename for role/runtime pair.
        """
        return f"{runtime_id}_{role}.log"


    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        """
        Convert value to integer with fallback default.
        """
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed


    @staticmethod
    def _empty_metrics_payload() -> dict[str, Any]:
        """
        Return empty distributed metrics payload structure.
        """
        return {
            "experiment_info": {
                "basic": {},
                "federated": {},
                "attack": {},
                "defense": {},
                "security": {},
            },
            "global_results": {
                "rounds": [],
                "global_loss": [],
                "global_accuracy": [],
            },
            "client_results": {},
        }


    @staticmethod
    def _build_command(config_path: Path, *, role: str) -> list[str]:
        """
        Build distributed experiment runner command.
        """
        return [
            sys.executable,
            "apps/backend/runners/experiment_runner.py",
            "--mode",
            "distributed",
            "--role",
            role,
            "--config",
            str(config_path),
            "--disable-status-server",
        ]


    @staticmethod
    def _clone_state(state: dict[str, Any]) -> dict[str, Any]:
        """
        Return deep copy of runtime state for response use.
        """
        return deepcopy(state)
