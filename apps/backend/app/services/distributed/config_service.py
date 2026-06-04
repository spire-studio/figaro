"""
Distributed configuration service.

Builds and normalizes distributed runtime configuration payloads.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core import exceptions
from app.services.simulation.compatibility import validate_llm_simulation_mode_or_raise
from app.services.simulation.job_service import SimulationJobService


class DistributedConfigService:
    """
    Sub-service for distributed configuration normalization.
    """


    @classmethod
    def build_default_job_config(cls) -> dict[str, Any]:
        """
        Build default distributed server config from schema defaults.
        """
        schema = SimulationJobService.get_config_schema()
        defaults = cls._init_defaults_from_schema(schema)
        return cls.normalize_server_config(defaults)


    @classmethod
    def normalize_server_config(cls, config_json: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize config for distributed server role and security defaults.
        """
        normalized = deepcopy(config_json)
        cls._set_value_by_path(normalized, "system.mode", "distributed")
        cls._set_value_by_path(normalized, "system.role", "server")
        cls._set_value_by_path(normalized, "system.node_role", "server")
        expected_clients = cls.extract_expected_clients(normalized)
        cls._set_value_by_path(normalized, "federated.num_clients", expected_clients)
        cls._set_value_by_path(normalized, "dataset.num_clients", expected_clients)
        validate_llm_simulation_mode_or_raise(normalized)
        return normalized


    @classmethod
    def build_server_runtime_config(
        cls,
        config_json: dict[str, Any],
        *,
        server_ip: str,
        server_port: int,
        expected_clients: int,
    ) -> dict[str, Any]:
        """
        Build runtime config payload for distributed server process.
        """
        normalized = cls.normalize_server_config(config_json)
        cls._set_value_by_path(normalized, "distributed.server_ip", server_ip)
        cls._set_value_by_path(normalized, "distributed.port", server_port)
        cls._set_value_by_path(normalized, "federated.num_clients", expected_clients)
        cls._set_value_by_path(normalized, "dataset.num_clients", expected_clients)
        return normalized


    @classmethod
    def build_participant_runtime_config(
        cls,
        config_json: dict[str, Any],
        *,
        server_ip: str,
        server_port: int,
        expected_clients: int,
        assigned_participant_id: int,
    ) -> dict[str, Any]:
        """
        Build runtime config payload for distributed participant process.
        """
        normalized = deepcopy(config_json)
        cls._set_value_by_path(normalized, "system.mode", "distributed")
        cls._set_value_by_path(normalized, "system.role", "client")
        cls._set_value_by_path(normalized, "system.node_role", "client")
        cls._set_value_by_path(normalized, "distributed.server_ip", server_ip)
        cls._set_value_by_path(normalized, "distributed.port", server_port)
        cls._set_value_by_path(normalized, "distributed.client_id", assigned_participant_id)
        cls._set_value_by_path(normalized, "federated.num_clients", expected_clients)
        cls._set_value_by_path(normalized, "dataset.num_clients", expected_clients)
        validate_llm_simulation_mode_or_raise(normalized)
        return normalized


    @classmethod
    def extract_expected_clients(cls, config_json: dict[str, Any]) -> int:
        """
        Extract expected client count from config with validation.
        """
        value = cls._get_value_by_path(config_json, "federated.num_clients")
        if value is None:
            value = cls._get_value_by_path(config_json, "dataset.num_clients")

        parsed = 0
        if isinstance(value, (int, float, str)) and str(value).strip() != "":
            try:
                parsed = int(float(value))
            except (TypeError, ValueError):
                parsed = 0

        if parsed <= 0:
            raise exceptions.BadRequestError("federated.num_clients must be greater than 0")
        return parsed


    @classmethod
    def _init_defaults_from_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Build default config tree from schema definitions.
        """
        output: dict[str, Any] = {}
        for key, raw_definition in schema.items():
            if key in {"role", "depends_on", "hidden", "ui"}:
                continue
            if not isinstance(raw_definition, dict):
                continue

            field_type = raw_definition.get("type")
            if isinstance(field_type, str):
                default = raw_definition.get("default")
                if default is not None:
                    output[key] = default
                elif field_type == "bool":
                    output[key] = False
                elif field_type == "number":
                    output[key] = 0
                elif field_type == "list_int":
                    output[key] = []
                elif field_type == "select":
                    options = raw_definition.get("options")
                    output[key] = options[0] if isinstance(options, list) and options else ""
                else:
                    output[key] = ""
                continue

            output[key] = cls._init_defaults_from_schema(raw_definition)
        return output


    @staticmethod
    def _get_value_by_path(obj: dict[str, Any], path: str) -> Any:
        """
        Get nested dictionary value by dotted path.
        """
        current: Any = obj
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current


    @staticmethod
    def _set_value_by_path(obj: dict[str, Any], path: str, value: Any) -> None:
        """
        Set nested dictionary value by dotted path.
        """
        parts = [part for part in path.split(".") if part]
        if not parts:
            return

        current = obj
        for part in parts[:-1]:
            next_value = current.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                current[part] = next_value
            current = next_value
        current[parts[-1]] = value
