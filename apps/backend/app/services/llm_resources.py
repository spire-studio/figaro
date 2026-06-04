"""Local LLM model and dataset discovery for schema-backed UI options."""

from __future__ import annotations

from pathlib import Path
from typing import Any


LLM_MODEL_ROOT = Path("models") / "llm"
LLM_DATASET_ROOT = Path("datasets") / "llm"
DEFAULT_LLM_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LLM_DATASET_PATH = "./datasets/llm/train.jsonl"
DEFAULT_LLM_EVALUATION_DATASET_PATH = "./datasets/llm/validation.jsonl"
SFT_DATASET_SUFFIXES = {".jsonl", ".parquet"}
EVALUATION_DATASET_MARKERS = ("eval", "valid", "validation")


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
        model_dir = _model_option_dir_for_config(config_file, root)
        if model_dir.is_dir() and _is_visible_relative(model_dir, project_root):
            candidates.append(model_dir)

    return _unique_sorted_relative_options(candidates, project_root)


def discover_llm_dataset_options(project_root: Path) -> list[str]:
    """Return local SFT training files and dataset directories under datasets/llm."""
    return _discover_llm_dataset_options(project_root, include_evaluation_files=False)


def discover_llm_evaluation_dataset_options(project_root: Path) -> list[str]:
    """Return local evaluation files and dataset directories under datasets/llm."""
    return _discover_llm_dataset_options(project_root, include_evaluation_files=True)


def get_llm_resource_options(project_root: Path) -> dict[str, Any]:
    """Return dynamic local LLM resource options for API and schema callers."""
    model_options = discover_llm_model_options(project_root)
    dataset_options = discover_llm_dataset_options(project_root)
    evaluation_dataset_options = discover_llm_evaluation_dataset_options(project_root)
    return {
        "models": model_options,
        "datasets": dataset_options,
        "evaluation_datasets": evaluation_dataset_options,
        "default_model": model_options[0] if model_options else "",
        "default_dataset": dataset_options[0] if dataset_options else "",
        "default_evaluation_dataset": _preferred_evaluation_dataset(evaluation_dataset_options),
    }


def augment_config_schema_with_llm_resources(schema: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Inject discovered local LLM resource options into the mutable config schema."""
    resources = get_llm_resource_options(project_root)

    _set_field_options(
        schema,
        ("llm", "base_model"),
        resources["models"],
        source_dir=f"./{LLM_MODEL_ROOT.as_posix()}",
        default_value=resources["default_model"],
        empty_message="No local models found",
    )
    _set_field_options(
        schema,
        ("sft", "dataset_path"),
        resources["datasets"],
        source_dir=f"./{LLM_DATASET_ROOT.as_posix()}",
        default_value=resources["default_dataset"],
        empty_message="No training datasets found",
    )
    _set_field_options(
        schema,
        ("evaluation", "dataset_path"),
        resources["evaluation_datasets"],
        source_dir=f"./{LLM_DATASET_ROOT.as_posix()}",
        default_value=resources["default_evaluation_dataset"],
        empty_message="No evaluation datasets found",
    )
    _set_field_default(schema, ("evaluation", "enable"), bool(resources["default_evaluation_dataset"]))
    return schema


def _discover_llm_dataset_options(project_root: Path, *, include_evaluation_files: bool) -> list[str]:
    root = project_root / LLM_DATASET_ROOT
    if not root.exists() or not root.is_dir():
        return []

    candidates: list[Path] = []
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in SFT_DATASET_SUFFIXES
            or not _is_visible_relative(path, project_root)
        ):
            continue
        if not include_evaluation_files and _is_evaluation_dataset(path):
            continue
        if path.suffix.lower() == ".parquet":
            dataset_dir = _dataset_option_dir_for_parquet(path, root)
            if dataset_dir is not None and _is_visible_relative(dataset_dir, project_root):
                candidates.append(dataset_dir)
                continue
        candidates.append(path)
    return _unique_sorted_relative_options(candidates, project_root)


def _model_option_dir_for_config(config_file: Path, model_root: Path) -> Path:
    try:
        relative = config_file.relative_to(model_root)
    except ValueError:
        return config_file.parent

    parts = relative.parts
    if "snapshots" in parts:
        snapshot_index = parts.index("snapshots")
        if snapshot_index > 0:
            return model_root.joinpath(*parts[:snapshot_index])
    return config_file.parent


def _dataset_option_dir_for_parquet(path: Path, dataset_root: Path) -> Path | None:
    try:
        relative = path.relative_to(dataset_root)
    except ValueError:
        return None
    if len(relative.parts) <= 1:
        return None
    return dataset_root / relative.parts[0]


def _set_field_options(
    schema: dict[str, Any],
    path: tuple[str, ...],
    options: list[str],
    *,
    source_dir: str,
    default_value: str,
    empty_message: str,
) -> None:
    node: Any = schema
    for part in path:
        if not isinstance(node, dict):
            return
        node = node.get(part)
    if not isinstance(node, dict):
        return

    node["options"] = options
    node["default"] = default_value
    ui = node.setdefault("ui", {})
    if isinstance(ui, dict):
        ui["option_source"] = {
            "type": "local_directory",
            "path": source_dir,
        }
        ui["empty_message"] = empty_message


def _preferred_evaluation_dataset(dataset_options: list[str]) -> str:
    for option in dataset_options:
        if _is_evaluation_option(option):
            return option
    return ""


def _is_evaluation_option(option: str) -> bool:
    stem = Path(option).stem.lower()
    return any(marker in stem for marker in EVALUATION_DATASET_MARKERS)


def _is_evaluation_dataset(path: Path) -> bool:
    stem = path.stem.lower()
    return any(marker in stem for marker in EVALUATION_DATASET_MARKERS)


def _set_field_default(schema: dict[str, Any], path: tuple[str, ...], default_value: Any) -> None:
    node: Any = schema
    for part in path:
        if not isinstance(node, dict):
            return
        node = node.get(part)
    if isinstance(node, dict):
        node["default"] = default_value


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
