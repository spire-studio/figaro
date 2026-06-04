from __future__ import annotations

from pathlib import Path
import tomllib


def test_llm_peft_dependencies_are_core_project_dependencies():
    project_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(pyproject["project"]["dependencies"])

    for package in ("transformers", "peft", "accelerate", "safetensors", "bitsandbytes"):
        assert package in dependencies
