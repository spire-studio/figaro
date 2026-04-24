"""
Prompt builders for the bench-mode FL experiment executor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
) -> str:
    """Build the user prompt for the experiment-parsing LLM call."""
    return (
        f"Experiment request:\n{state.goal}\n\n"
        f"System mode: {state.system_mode}\n"
        f"Capabilities: {capabilities}\n"
        f"Default base config: {base_config}\n\n"
        "Parse the request into experiment configurations.\n"
        "Return ONLY a JSON object with plan_summary and experiments list, no commentary."
    )
