"""
Planning helpers for the bench-mode FL experiment executor.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from app.core.config import settings

SCHEMA_META_KEYS = {"role", "depends_on", "hidden", "ui"}


def _is_field_definition(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if isinstance(node.get("type"), str):
        return True
    return "default" in node and not any(
        key not in SCHEMA_META_KEYS and isinstance(value, dict)
        for key, value in node.items()
    )


def _default_for_field(node: dict[str, Any]) -> Any:
    if "default" in node:
        return copy.deepcopy(node["default"])
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
            return copy.deepcopy(options[0])
        return ""
    return ""


def build_initial_config(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Build a default FL experiment config from the simulation schema.
    """
    if not schema:
        return {
            "task": {"type": "classic_fl"},
            "dataset": {"name": "CIFAR-10", "distribution": "non_iid", "alpha": 0.5},
            "model": {"name": "Auto"},
            "federated": {
                "num_clients": 3,
                "num_rounds": 10,
                "clients_per_round": 2,
                "local_epochs": 5,
                "learning_rate": 0.01,
                "aggregation": "fedavg",
                "seed": 42,
            },
        }

    output: dict[str, Any] = {}
    for key, definition in schema.items():
        if key in SCHEMA_META_KEYS or not isinstance(definition, dict):
            continue
        if _is_field_definition(definition):
            output[key] = _default_for_field(definition)
            continue
        output[key] = build_initial_config(definition)
    return output


def deep_merge_config(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge config dictionaries while ignoring schema/internal metadata keys."""
    merged = copy.deepcopy(base)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if key.startswith("_"):
            continue
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge_config(base_value, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def lock_structured_constraints(config: dict[str, Any], constraints: dict[str, Any] | None) -> dict[str, Any]:
    """Apply user-selected structured constraints as final locked values."""
    if not isinstance(constraints, dict) or not constraints:
        return copy.deepcopy(config)
    return deep_merge_config(config, constraints)


def _option_ui(definition: dict[str, Any]) -> dict[str, Any]:
    ui = definition.get("ui")
    if not isinstance(ui, dict):
        return {}
    option_ui = ui.get("options")
    return option_ui if isinstance(option_ui, dict) else {}


def _is_disabled_option(definition: dict[str, Any], value: Any) -> bool:
    metadata = _option_ui(definition).get(str(value))
    return isinstance(metadata, dict) and metadata.get("disabled") is True


def collect_disabled_option_errors(
    config: dict[str, Any],
    schema: dict[str, Any],
    *,
    path_prefix: str = "",
) -> list[str]:
    """Return user-facing errors for disabled schema select options in config."""
    errors: list[str] = []
    for key, definition in schema.items():
        if key in SCHEMA_META_KEYS or not isinstance(definition, dict):
            continue
        full_path = f"{path_prefix}.{key}" if path_prefix else key
        if _is_field_definition(definition):
            if definition.get("type") == "select":
                value = config.get(key)
                if value is not None and _is_disabled_option(definition, value):
                    errors.append(f"{full_path}={value!r} is marked disabled in config_schema.yaml")
            continue
        child_config = config.get(key)
        if isinstance(child_config, dict):
            errors.extend(collect_disabled_option_errors(child_config, definition, path_prefix=full_path))
    return errors


def build_schema_prompt_context(schema: dict[str, Any]) -> dict[str, Any]:
    """Build compact schema context for LLM planning prompts."""
    fields: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], prefix: str = "") -> None:
        for key, definition in node.items():
            if key in SCHEMA_META_KEYS or not isinstance(definition, dict):
                continue
            path = f"{prefix}.{key}" if prefix else key
            if _is_field_definition(definition):
                ui = definition.get("ui") if isinstance(definition.get("ui"), dict) else {}
                options = definition.get("options") if isinstance(definition.get("options"), list) else None
                disabled_options = [
                    option
                    for option in (options or [])
                    if _is_disabled_option(definition, option)
                ]
                executable_options = [
                    option
                    for option in (options or [])
                    if option not in disabled_options
                ]
                field: dict[str, Any] = {
                    "path": path,
                    "type": definition.get("type"),
                    "default": _default_for_field(definition),
                }
                if options is not None:
                    field["options"] = options
                    field["executable_options"] = executable_options
                    field["disabled_options"] = disabled_options
                if isinstance(ui, dict):
                    if ui.get("label"):
                        field["label"] = ui["label"]
                    if ui.get("prompt_hint"):
                        field["prompt_hint"] = ui["prompt_hint"]
                    if ui.get("featured") is True:
                        field["featured"] = True
                fields.append(field)
                continue
            walk(definition, path)

    walk(schema)
    return {
        "defaults": build_initial_config(schema),
        "fields": fields,
    }


def dumps_for_prompt(value: Any) -> str:
    """Stable JSON rendering for LLM prompt context."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def select_llm_model(model_name: str | None) -> str:
    """Resolve the LLM model for planning requests."""
    if model_name and model_name.strip():
        return model_name.strip()
    return settings.default_llm_model
