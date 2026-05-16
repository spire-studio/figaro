from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.core import exceptions

CLASSIC_FL_TASK = "classic_fl"
LLM_PEFT_SFT_TASK = "llm_peft_sft"
LLM_SIMULATION_ONLY_MESSAGE = "LLM PEFT is currently supported in simulation mode only"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _ensure_fl_core_path() -> None:
    libs_path = _project_root() / "libs"
    if str(libs_path) not in sys.path:
        sys.path.insert(0, str(libs_path))


def _registry():
    _ensure_fl_core_path()
    from fl_core import simulation_registry

    return simulation_registry


def runtime_task_type(config: dict[str, Any]) -> str:
    """Return the training route encoded in config, defaulting legacy configs to classic FL."""
    task = config.get("task")
    if not isinstance(task, dict):
        return CLASSIC_FL_TASK
    raw_task_type = task.get("type", CLASSIC_FL_TASK)
    if not isinstance(raw_task_type, str) or not raw_task_type.strip():
        return CLASSIC_FL_TASK
    return raw_task_type.strip()


def runtime_mode(config: dict[str, Any]) -> str:
    """Return the selected runtime mode, defaulting missing legacy configs to simulation."""
    system = config.get("system")
    if not isinstance(system, dict):
        return "simulation"
    raw_mode = system.get("mode", "simulation")
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        return "simulation"
    return raw_mode.strip()


def validate_llm_simulation_mode_or_raise(config: dict[str, Any]) -> None:
    if runtime_task_type(config) == LLM_PEFT_SFT_TASK and runtime_mode(config) != "simulation":
        raise exceptions.BadRequestError(LLM_SIMULATION_ONLY_MESSAGE)


def canonicalize_runtime_config(config: dict[str, Any]) -> None:
    if runtime_task_type(config) != CLASSIC_FL_TASK:
        return

    registry = _registry()

    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return
    raw_dataset_name = dataset.get("name")
    if isinstance(raw_dataset_name, str):
        dataset["name"] = registry.canonical_dataset_name(raw_dataset_name)

    spec = registry.get_dataset_spec(dataset.get("name"))

    model = config.get("model")
    if not isinstance(model, dict):
        return
    raw_model_name = model.get("name")
    if isinstance(raw_model_name, str) and raw_model_name.strip().lower() != "auto":
        model["name"] = registry.canonical_model_name(raw_model_name)
    if spec is not None:
        model["input_shape"] = list(spec.input_shape)
        model["num_classes"] = spec.num_classes


def validate_runtime_config_or_raise(config: dict[str, Any]) -> None:
    validate_llm_simulation_mode_or_raise(config)

    if runtime_task_type(config) != CLASSIC_FL_TASK:
        return

    registry = _registry()
    errors = registry.validate_training_combination(config)
    if errors:
        raise exceptions.BadRequestError("; ".join(errors))


def runtime_capabilities() -> dict[str, Any]:
    return _registry().capabilities_payload()
