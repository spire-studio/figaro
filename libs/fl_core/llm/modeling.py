from __future__ import annotations

import importlib.util


LLM_RUNTIME_DEPENDENCIES = ("transformers", "peft", "accelerate")


def missing_llm_runtime_dependencies() -> list[str]:
    """Return required LLM runtime dependencies that are not importable."""
    return [name for name in LLM_RUNTIME_DEPENDENCIES if importlib.util.find_spec(name) is None]


def require_llm_runtime_dependencies() -> None:
    missing = missing_llm_runtime_dependencies()
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing LLM PEFT runtime dependencies: {joined}")
