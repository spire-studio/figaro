"""
Service layer for simulation job management.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import exceptions
from app.models.simulation import SimulationJob, SimulationJobStatus, SimulationRunStatus
from app.repositories.simulation import SimulationJobRepository, SimulationRunRepository
from app.services.llm_resources import augment_config_schema_with_llm_resources
from app.services.simulation.compatibility import canonicalize_runtime_config, validate_runtime_config_or_raise


class SimulationJobService:
    """Application service for simulation job operations."""

    JOB_NAME_CONFLICT_MESSAGE = "Job name already exists"
    _SCHEMA_META_KEYS = {"role", "depends_on", "hidden", "ui"}


    def __init__(self, session: AsyncSession):
        self.session = session
        self.job_repository = SimulationJobRepository(session)
        self.run_repository = SimulationRunRepository(session)


    async def create_job(self, name: str, description: str | None) -> SimulationJob:
        """
        Create a simulation job with default normalized config.
        """
        await self._ensure_unique_job_name(name)
        try:
            default_config = self.normalize_simulation_config({})
            job = await self.job_repository.create_job(
                name=name,
                description=description,
                config_json=default_config,
            )
            await self.session.commit()
            await self.session.refresh(job)
            return job
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE) from exc


    async def update_job(
        self,
        job_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
        status: SimulationJobStatus | None = None,
    ) -> SimulationJob:
        """
        Update simulation job metadata.
        """
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise exceptions.JobNotFound("Job not found")

        if name is not None and name != job.name:
            await self._ensure_unique_job_name(name, exclude_job_id=job_id)

        try:
            await self.job_repository.update_job(
                job,
                name=name,
                description=description,
                status=status,
            )
            await self.session.commit()
            await self.session.refresh(job)
            return job
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE) from exc


    async def update_job_config(self, job_id: int, config: dict[str, Any]) -> SimulationJob:
        """
        Update simulation job config after schema normalization and validation.
        """
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise exceptions.JobNotFound("Job not found")

        normalized = self.normalize_simulation_config(config)
        await self.job_repository.update_job(
            job,
            config_json=normalized,
            status=SimulationJobStatus.READY,
        )
        await self.session.commit()
        await self.session.refresh(job)
        return job


    async def list_jobs(self) -> list[SimulationJob]:
        """
        List simulation jobs ordered by creation time.
        """
        return await self.job_repository.list_jobs()


    async def get_job(self, job_id: int) -> SimulationJob | None:
        """
        Get a simulation job by ID.
        """
        return await self.job_repository.get_job(job_id)


    async def get_job_or_raise(self, job_id: int) -> SimulationJob:
        """
        Get a simulation job by ID or raise not-found.
        """
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise exceptions.JobNotFound("Job not found")
        return job


    async def copy_job(self, source_job_id: int, name: str | None = None) -> SimulationJob:
        """
        Copy an existing job and optionally override the copied name.
        """
        source = await self.job_repository.get_job(source_job_id)
        if not source:
            raise exceptions.JobNotFound("Source job not found")

        cloned_name = name or f"{source.name} (copy)"
        await self._ensure_unique_job_name(cloned_name)

        try:
            cloned = await self.job_repository.create_job(
                name=cloned_name,
                description=source.description,
                config_json=deepcopy(source.config_json if isinstance(source.config_json, dict) else {}),
            )
            await self.session.commit()
            await self.session.refresh(cloned)
            return cloned
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE) from exc


    async def get_job_config(self, job_id: int) -> dict[str, Any]:
        """
        Get normalized config payload for a job.
        """
        job = await self.get_job_or_raise(job_id)
        if not isinstance(job.config_json, dict):
            raise exceptions.JobConfigNotFound("Job config not found")
        return job.config_json


    async def delete_job(self, job_id: int) -> None:
        """
        Delete a job when no queued/running runs remain.
        """
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise exceptions.JobNotFound("Job not found")

        runs = await self.run_repository.list_runs_for_job(job_id)
        if any(run.status in {SimulationRunStatus.QUEUED, SimulationRunStatus.RUNNING} for run in runs):
            raise exceptions.ActiveRunsConflict("Cannot delete job with queued or running runs")

        await self.job_repository.delete_job(job_id)
        await self.session.commit()

        for run in runs:
            self._cleanup_run_local_files(run.id)


    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """
        Load and validate the YAML config schema.
        """
        schema_path = cls._config_schema_path()
        if not schema_path.exists():
            raise exceptions.ConfigSchemaNotFound("Config schema not found")

        loaded = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise exceptions.InternalServiceError("Config schema must be a YAML object.")
        return augment_config_schema_with_llm_resources(loaded, cls._project_root())

    @classmethod
    def normalize_simulation_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """
        Public config normalization API for simulation configurations.
        """
        if not isinstance(config, dict):
            raise exceptions.BadRequestError("Config must be a JSON object")
        return cls._normalize_simulation_config(config)


    async def _ensure_unique_job_name(self, name: str, *, exclude_job_id: int | None = None) -> None:
        """
        Validate that job name is unique within simulation jobs.
        """
        existing = await self.job_repository.get_job_by_name(name)
        if existing and (exclude_job_id is None or existing.id != exclude_job_id):
            raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE)


    @classmethod
    def _normalize_simulation_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """
        Build schema-backed simulation config and enforce fixed invariants.
        """
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
        """
        Keep dataset.num_clients and federated.num_clients aligned.

        The runtime primarily keys off federated.num_clients, while other UI
        flows may still read dataset.num_clients. Treat federated as the
        authoritative field when both are present; otherwise propagate the one
        explicit positive value to the other side.
        """
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
    def _validate_config_node(
        cls,
        config_node: dict[str, Any],
        schema_node: dict[str, Any],
        *,
        path_prefix: str,
    ) -> None:
        """
        Validate config tree recursively against schema node definition.
        """
        allowed_keys = {
            key
            for key, value in schema_node.items()
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
        """
        Validate one leaf field according to schema definition.
        """
        field_type = definition.get("type")

        if field_type == "text":
            if not isinstance(value, str):
                raise exceptions.BadRequestError(f"{path} must be a string")
            return

        if field_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise exceptions.BadRequestError(f"{path} must be a number")
            return

        if field_type == "bool":
            if not isinstance(value, bool):
                raise exceptions.BadRequestError(f"{path} must be a boolean")
            return

        if field_type == "list_int":
            if not isinstance(value, list):
                raise exceptions.BadRequestError(f"{path} must be a list of integers")
            for item in value:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise exceptions.BadRequestError(f"{path} must be a list of integers")
            return

        if field_type == "select":
            options = definition.get("options")
            if not isinstance(options, list) or value not in options:
                raise exceptions.BadRequestError(f"{path} has invalid option")


    def _cleanup_run_local_files(self, run_id: str) -> None:
        """
        Remove local files that belong to deleted simulation run IDs.
        """
        candidates = [
            self._project_root() / "configs" / "simulation_runs" / f"{run_id}.json",
            self._project_root() / "config" / "runs" / f"{run_id}.json",
            self._project_root() / "results" / f"live_results_{run_id}.json",
        ]
        candidates.extend((self._project_root() / "configs" / "simulation_runs").glob(f"{run_id}*.json"))
        candidates.extend((self._project_root() / "configs" / "simulation_runs").glob(f"*_{run_id}.json"))
        candidates.extend((self._project_root() / "results").glob(f"live_results_{run_id}*.json"))
        candidates.extend((self._project_root() / "results").glob(f"*_{run_id}_live_results.json"))
        candidates.extend((self._project_root() / "logs" / "simulation_runs").glob(f"{run_id}*_server.log"))
        candidates.extend((self._project_root() / "logs" / "simulation_runs").glob(f"*_{run_id}_server.log"))
        for path in candidates:
            try:
                if path.exists() and path.is_file():
                    path.unlink()
            except OSError:
                continue


    @classmethod
    def _config_schema_path(cls) -> Path:
        """
        Return absolute path to config schema YAML.
        """
        return cls._project_root() / "config_schema.yaml"


    @classmethod
    def _init_defaults_from_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Build default config tree from schema definitions.
        """
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
        """
        Deep-merge override tree into base tree.
        """
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
        """
        Return project root path.
        """
        return Path(__file__).resolve().parents[5]


    @staticmethod
    def _is_field_definition(node: dict[str, Any]) -> bool:
        """
        Return True when schema node is a leaf field definition.
        """
        return isinstance(node.get("type"), str)


    @staticmethod
    def _default_for_field(node: dict[str, Any]) -> Any:
        """
        Build default value for a single schema field definition.
        """
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
