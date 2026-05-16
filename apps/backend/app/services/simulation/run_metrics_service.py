"""
Simulation run metrics service.

Handles metrics loading, normalization, and persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import exceptions
from app.models.simulation import SimulationRun, SimulationRunStatus
from app.repositories.simulation import SimulationRunRepository

TRAINING_RESULT_ARTIFACT = "training_result"
TERMINAL_RUN_STATUSES = {
    SimulationRunStatus.SUCCEEDED,
    SimulationRunStatus.FAILED,
    SimulationRunStatus.CANCELLED,
}


class SimulationRunMetricsService:
    """
    Sub-service for simulation run metrics reads and normalization.
    """


    def __init__(self, session: AsyncSession):
        self.session = session
        self.run_repository = SimulationRunRepository(session)


    async def get_run_metrics(self, run_id: str) -> dict[str, Any]:
        """
        Get normalized metrics payload for run, preferring live result files.
        """
        run = await self.run_repository.get_run(run_id)
        if not run:
            raise exceptions.RunNotFound("Run not found")

        if run.status in TERMINAL_RUN_STATUSES and isinstance(run.metrics_json, dict) and run.metrics_json:
            return self._normalize_metrics_payload(run.metrics_json)

        live_result_path = self._results_dir() / self._build_live_results_filename(run_id)
        loaded = self._load_metrics_file(live_result_path)
        if loaded is not None:
            await self._persist_run_metrics_if_changed(run, loaded)
            return loaded

        results = await self.run_repository.list_results(run_id)
        training_results = [result for result in results if result.artifact_type == TRAINING_RESULT_ARTIFACT]
        for artifact in reversed(training_results):
            artifact_path = self.resolve_result_artifact_path(artifact.path)
            loaded = self._load_metrics_file(artifact_path)
            if loaded is not None:
                await self._persist_run_metrics_if_changed(run, loaded)
                return loaded

        if isinstance(run.metrics_json, dict) and run.metrics_json:
            return self._normalize_metrics_payload(run.metrics_json)

        return self.empty_metrics_payload()


    async def persist_run_metrics_if_changed(self, run: SimulationRun, metrics: dict[str, Any]) -> None:
        """
        Persist metrics JSON only when value changed.
        """
        current = run.metrics_json if isinstance(run.metrics_json, dict) else {}
        if self._stable_json(current) == self._stable_json(metrics):
            return

        await self.run_repository.update_run_metrics(run, metrics)
        await self.session.commit()
        await self.session.refresh(run)


    def resolve_result_artifact_path(self, artifact_path: str) -> Path:
        """
        Resolve artifact path to absolute path.
        """
        path = Path(artifact_path)
        if path.is_absolute():
            return path
        return self._project_root() / path


    def load_metrics_file(self, file_path: Path) -> dict[str, Any] | None:
        """
        Load and normalize metrics file when valid.
        """
        return self._load_metrics_file(file_path)


    def guess_live_result_file_for_run(self, run: SimulationRun) -> Path | None:
        """
        Best-effort lookup of live result file for one run.
        """
        return self._guess_live_result_file_for_run(run)


    @staticmethod
    def empty_metrics_payload() -> dict[str, Any]:
        """
        Return empty simulation metrics payload structure.
        """
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
            "llm_runtime": {},
            "llm_artifacts": [],
            "client_results": {},
        }


    async def _persist_run_metrics_if_changed(self, run: SimulationRun, metrics: dict[str, Any]) -> None:
        """
        Persist metrics JSON only when value changed.
        """
        await self.persist_run_metrics_if_changed(run, metrics)


    def _project_root(self) -> Path:
        """
        Return project root directory.
        """
        return Path(__file__).resolve().parents[5]


    def _results_dir(self) -> Path:
        """
        Return simulation run results directory.
        """
        path = self._project_root() / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path


    def _build_live_results_filename(self, run_id: str) -> str:
        """
        Build live result filename for one run ID.
        """
        return f"live_results_{run_id}.json"


    def _load_metrics_file(self, file_path: Path) -> dict[str, Any] | None:
        """
        Load and normalize metrics file when valid.
        """
        if not file_path.exists() or not file_path.is_file():
            return None

        try:
            loaded = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(loaded, dict):
            return None
        return self._normalize_metrics_payload(loaded)


    def _guess_live_result_file_for_run(self, run: SimulationRun) -> Path | None:
        """
        Best-effort lookup of live result file for one run.
        """
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

        def _mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        candidates.sort(key=_mtime, reverse=True)
        return candidates[0]


    def _normalize_metrics_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize simulation metrics payload into stable response schema.
        """
        normalized = self.empty_metrics_payload()

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
        """
        Convert list-like value to float list with invalid values dropped.
        """
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
        """
        Convert list-like value to int list with invalid values dropped.
        """
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
        """
        Serialize value to stable JSON string for equality checks.
        """
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
