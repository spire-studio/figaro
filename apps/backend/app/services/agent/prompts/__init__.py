"""
Prompt builders for the bench-mode FL experiment executor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.agent.planning import dumps_for_prompt
from app.services.agent.state import AgentState

_PLAN_INSTRUCTIONS_PATH = Path(__file__).with_name("plan_instructions.txt")


def get_plan_instructions() -> str:
    """Return the system instructions used by the parse node."""
    return _PLAN_INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()


def build_plan_system_instructions(global_prompt: str) -> str:
    """
    Compose the experiment planner instructions with the user's request.
    """
    return (
        f"{get_plan_instructions()}\n\n"
        "User experiment request (highest priority):\n"
        f"{global_prompt.strip()}"
    )


def build_plan_prompt(
    *,
    state: AgentState,
    capabilities: dict[str, Any],
    base_config: dict[str, Any],
    schema_context: dict[str, Any] | None = None,
    config_constraints: dict[str, Any] | None = None,
) -> str:
    """Build the user prompt for the experiment-parsing LLM call."""
    constraints = config_constraints if isinstance(config_constraints, dict) else {}
    schema_payload = schema_context if isinstance(schema_context, dict) else {}
    return (
        f"Experiment request:\n{state.goal}\n\n"
        f"System mode: {state.system_mode}\n"
        f"Runtime capabilities:\n{dumps_for_prompt(capabilities)}\n\n"
        f"Schema context from config_schema.yaml:\n{dumps_for_prompt(schema_payload)}\n\n"
        f"Default base config after applying user constraints:\n{dumps_for_prompt(base_config)}\n\n"
        f"User-selected structured constraints:\n{dumps_for_prompt(constraints)}\n\n"
        "Parse the request into experiment configurations.\n"
        "User-selected structured constraints are locked controls and have higher priority than natural-language text.\n"
        "Every experiment must inherit the structured constraints. If text conflicts with them, keep the structured constraints.\n"
        "Use executable_options for select fields. Do not use disabled_options.\n"
        "For LLM PEFT simulation requests, set task.type=\"llm_peft_sft\" and use llm/sft/peft/federated fields.\n"
        "Respect dataset_model_compatibility. Use model.name=\"Auto\" when the user changes dataset without naming a model.\n"
        "Respect privacy/compression compatibility rules from Runtime capabilities.\n"
        "Return ONLY a JSON object with plan_summary and experiments list, no commentary."
    )
