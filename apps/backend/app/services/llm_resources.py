"""Local LLM model and dataset discovery for schema-backed UI options."""

from __future__ import annotations

from pathlib import Path
from typing import Any


LLM_MODEL_ROOT = Path("models") / "llm"
LLM_DATASET_ROOT = Path("datasets") / "llm"
DEFAULT_LLM_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LLM_DATASET_PATH = "./datasets/llm/train.jsonl"
DEFAULT_LLM_EVALUATION_DATASET_PATH = "./datasets/llm/validation.jsonl"
SFT_DATASET_SUFFIXES = {".jsonl"}


def discover_llm_model_options(project_root: Path) -> list[str]:
    """Return local model directory paths under models/llm as schema option values."""
    root = project_root / LLM_MODEL_ROOT
    if not root.exists() or not root.is_dir():
        return []

    candidates: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and _is_visible_relative(child, project_root):
            candidates.append(child)

    for config_file in root.rglob("config.json"):
        model_dir = config_file.parent
        if model_dir.is_dir() and _is_visible_relative(model_dir, project_root):
            candidates.append(model_dir)

    return _unique_sorted_relative_options(candidates, project_root)


def discover_llm_dataset_options(project_root: Path) -> list[str]:
    """Return local SFT JSONL files under datasets/llm as schema option values."""
    root = project_root / LLM_DATASET_ROOT
    if not root.exists() or not root.is_dir():
        return []

    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SFT_DATASET_SUFFIXES
        and _is_visible_relative(path, project_root)
    ]
    return _unique_sorted_relative_options(candidates, project_root)


def augment_config_schema_with_llm_resources(schema: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Inject discovered local LLM resource options into the mutable config schema."""
    model_options = _merge_options(
        [DEFAULT_LLM_BASE_MODEL],
        discover_llm_model_options(project_root),
    )
    dataset_options = _merge_options(
        [DEFAULT_LLM_DATASET_PATH],
        discover_llm_dataset_options(project_root),
    )
    evaluation_dataset_options = _merge_options(
        [DEFAULT_LLM_EVALUATION_DATASET_PATH],
        discover_llm_dataset_options(project_root),
    )

    _set_field_options(
        schema,
        ("llm", "base_model"),
        model_options,
        source_dir=f"./{LLM_MODEL_ROOT.as_posix()}",
    )
    _set_field_options(
        schema,
        ("sft", "dataset_path"),
        dataset_options,
        source_dir=f"./{LLM_DATASET_ROOT.as_posix()}",
    )
    _set_field_options(
        schema,
        ("evaluation", "dataset_path"),
        evaluation_dataset_options,
        source_dir=f"./{LLM_DATASET_ROOT.as_posix()}",
    )
    return schema


def _set_field_options(
    schema: dict[str, Any],
    path: tuple[str, ...],
    options: list[str],
    *,
    source_dir: str,
) -> None:
    node: Any = schema
    for part in path:
        if not isinstance(node, dict):
            return
        node = node.get(part)
    if not isinstance(node, dict):
        return

    node["options"] = options
    ui = node.setdefault("ui", {})
    if isinstance(ui, dict):
        ui["option_source"] = {
            "type": "local_directory",
            "path": source_dir,
        }


def _unique_sorted_relative_options(paths: list[Path], project_root: Path) -> list[str]:
    options = {_relative_option(path, project_root) for path in paths}
    return sorted(options, key=str.lower)


def _relative_option(path: Path, project_root: Path) -> str:
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        return path.as_posix()
    return f"./{relative.as_posix()}"


def _is_visible_relative(path: Path, project_root: Path) -> bool:
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        return False
    return all(part and not part.startswith(".") for part in relative.parts)


def _merge_options(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for group in groups:
        for option in group:
            if option in seen:
                continue
            seen.add(option)
            output.append(option)
    return output
