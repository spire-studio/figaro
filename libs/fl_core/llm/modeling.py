from __future__ import annotations

import importlib.util


OPTIONAL_DEPENDENCIES = ("transformers", "peft", "accelerate")


def missing_optional_dependencies() -> list[str]:
    """Return optional LLM runtime dependencies that are not importable."""
    return [name for name in OPTIONAL_DEPENDENCIES if importlib.util.find_spec(name) is None]


def require_optional_dependencies() -> None:
    missing = missing_optional_dependencies()
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing optional LLM PEFT dependencies: {joined}")

