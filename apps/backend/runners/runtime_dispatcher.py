from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CLASSIC_FL_TASK = "classic_fl"
LLM_PEFT_SFT_TASK = "llm_peft_sft"
SUPPORTED_TASK_TYPES = {CLASSIC_FL_TASK, LLM_PEFT_SFT_TASK}
LLM_SIMULATION_ONLY_MESSAGE = "LLM PEFT is currently supported in simulation mode only"


def load_runtime_config(config_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a YAML/JSON object")
    return loaded


def get_task_type(config: dict[str, Any]) -> str:
    """Return the selected training route, defaulting legacy configs to classic FL."""
    task = config.get("task")
    if not isinstance(task, dict):
        return CLASSIC_FL_TASK

    task_type = task.get("type", CLASSIC_FL_TASK)
    if not isinstance(task_type, str) or not task_type.strip():
        return CLASSIC_FL_TASK
    return task_type.strip()


def get_runtime_mode(config: dict[str, Any]) -> str:
    """Return the runtime mode, defaulting missing legacy configs to simulation."""
    system = config.get("system")
    if not isinstance(system, dict):
        return "simulation"
    mode = system.get("mode", "simulation")
    if not isinstance(mode, str) or not mode.strip():
        return "simulation"
    return mode.strip()


def validate_runtime_route(config: dict[str, Any]) -> None:
    if get_task_type(config) == LLM_PEFT_SFT_TASK and get_runtime_mode(config) != "simulation":
        raise ValueError(LLM_SIMULATION_ONLY_MESSAGE)


def run_runtime(config_path: Path) -> bool:
    config = load_runtime_config(config_path)
    validate_runtime_route(config)
    task_type = get_task_type(config)

    if task_type == CLASSIC_FL_TASK:
        from classic_fl_runtime import run_classic_fl_runtime

        return run_classic_fl_runtime(config_path)

    if task_type == LLM_PEFT_SFT_TASK:
        from llm_peft_runtime import run_llm_peft_runtime

        return run_llm_peft_runtime(config_path)

    supported = ", ".join(sorted(SUPPORTED_TASK_TYPES))
    raise ValueError(f"Unsupported task.type={task_type!r}. Supported task types: {supported}")
