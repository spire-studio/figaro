from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FL_CORE_PARENT = PROJECT_ROOT / "libs"
if str(FL_CORE_PARENT) not in sys.path:
    sys.path.insert(0, str(FL_CORE_PARENT))

from fl_core.llm.config import LlmPeftRuntimeConfig, normalize_llm_peft_config
from fl_core.llm.data import load_sft_records, split_records_by_client
from fl_core.llm.aggregation import aggregate_adapter_state_dicts
from fl_core.llm.artifacts import (
    adapter_artifact_record,
    adapter_state_size_bytes,
    load_adapter_artifact,
    save_adapter_artifact,
    sha256_file,
)
from fl_core.llm.metrics import append_llm_round_metrics, empty_llm_metrics_payload
from fl_core.llm.modeling import missing_llm_runtime_dependencies
from fl_core.llm.trainer import LlmPeftTrainer


class LlmPeftRuntimeNotImplemented(RuntimeError):
    """Raised when the planned LLM PEFT route is selected before implementation."""


def run_llm_peft_runtime(config_path: Path) -> bool:
    """
    Bootstrap the Phase 2 LLM PEFT federated fine-tuning route.

    This route already owns config/data/metrics preparation. The heavy
    Transformers/PEFT training is routed through this entrypoint while keeping
    a clear metrics record when the deployed runtime environment is incomplete.
    """
    try:
        raw_config = _load_config(config_path)
        runtime_config = normalize_llm_peft_config(raw_config)
        metrics = empty_llm_metrics_payload(runtime_config)
        _write_metrics(runtime_config, metrics)

        print("LLM_PEFT_RUNTIME_SELECTED")
        print(f"LLM_BASE_MODEL: {runtime_config.llm.base_model}")
        print(f"SFT_DATASET: {runtime_config.sft.dataset_path}")
        print(f"PEFT_ADAPTER: method={runtime_config.peft.method} rank={runtime_config.peft.rank}")

        dataset_path = _resolve_runtime_path(runtime_config.sft.dataset_path)
        dataset_file_format = _effective_sft_file_format(
            dataset_path,
            configured_file_format=runtime_config.sft.file_format,
        )
        records = load_sft_records(
            dataset_path,
            data_format=runtime_config.sft.format,
            file_format=runtime_config.sft.file_format,
        )
        client_splits = split_records_by_client(
            records,
            num_clients=runtime_config.federated.num_clients,
            seed=runtime_config.federated.seed,
        )
        metrics["llm_dataset"] = {
            "num_records": len(records),
            "client_record_counts": [len(client_records) for client_records in client_splits],
            "format": runtime_config.sft.format,
            "file_format": dataset_file_format,
            "prompt_template": runtime_config.sft.prompt_template,
            "path": str(dataset_path),
        }
        evaluation_records = None
        if runtime_config.evaluation.enabled:
            if runtime_config.evaluation.dataset_path is None:
                raise ValueError("evaluation.dataset_path must be set when evaluation.enable is true")
            evaluation_path = _resolve_runtime_path(runtime_config.evaluation.dataset_path)
            evaluation_file_format = _effective_sft_file_format(
                evaluation_path,
                configured_file_format=runtime_config.sft.file_format,
            )
            evaluation_records = load_sft_records(
                evaluation_path,
                data_format=runtime_config.sft.format,
                file_format=runtime_config.sft.file_format,
            )[: runtime_config.evaluation.max_samples]
            metrics["llm_evaluation"] = {
                "enabled": True,
                "path": str(evaluation_path),
                "file_format": evaluation_file_format,
                "num_records": len(evaluation_records),
                "batch_size": runtime_config.evaluation.batch_size,
                "max_samples": runtime_config.evaluation.max_samples,
            }
        else:
            metrics["llm_evaluation"] = {"enabled": False}
        _write_metrics(runtime_config, metrics)

        missing = missing_llm_runtime_dependencies()
        if missing:
            print(
                "LLM_PEFT_RUNTIME_BLOCKED: missing LLM runtime dependencies: "
                + ", ".join(missing)
            )
            metrics["llm_runtime"] = {
                "status": "blocked",
                "reason": "missing_llm_runtime_dependencies",
                "missing_dependencies": missing,
            }
            _write_metrics(runtime_config, metrics)
            print(f"\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230: {_result_path(runtime_config)}")
            return False

        trainer = LlmPeftTrainer(runtime_config)
        global_adapter_state: dict[str, torch.Tensor] | None = None
        parent_adapter_path: str | None = None
        parent_adapter_sha256: str | None = None
        if runtime_config.peft.resume_adapter_path is not None:
            resume_path = _resolve_runtime_path(runtime_config.peft.resume_adapter_path)
            global_adapter_state, resume_metadata = load_adapter_artifact(resume_path)
            parent_adapter_path = str(resume_path)
            parent_adapter_sha256 = sha256_file(resume_path)
            metrics["llm_runtime"] = {
                "status": "resumed",
                "resume_adapter_path": parent_adapter_path,
                "resume_adapter_sha256": parent_adapter_sha256,
                "resume_metadata": resume_metadata,
            }
            _write_metrics(runtime_config, metrics)
            print(f"LLM_PEFT_RESUME_ADAPTER: {resume_path}")

        rng = random.Random(runtime_config.federated.seed)
        work_dir = _runtime_work_dir(runtime_config)
        adapter_dir = _adapter_dir(runtime_config)
        nonempty_client_ids = [idx for idx, client_records in enumerate(client_splits) if client_records]
        if not nonempty_client_ids:
            raise ValueError("No clients have SFT records")

        for round_num in range(1, runtime_config.federated.num_rounds + 1):
            started_at = time.perf_counter()
            selected_ids = _select_clients(
                nonempty_client_ids,
                clients_per_round=runtime_config.federated.clients_per_round,
                rng=rng,
            )
            print(f"LLM_PEFT_ROUND_START: round={round_num} clients={selected_ids}")
            updates = []
            for client_id in selected_ids:
                print(
                    "LLM_PEFT_CLIENT_START: "
                    f"round={round_num} client={client_id} records={len(client_splits[client_id])}"
                )
                update = trainer.train_client(
                    client_id=client_id,
                    records=client_splits[client_id],
                    round_num=round_num,
                    initial_adapter_state=global_adapter_state,
                    output_dir=work_dir / f"round_{round_num}" / f"client_{client_id}",
                )
                updates.append(update)
                _append_client_metrics(metrics, client_id=client_id, update=update)
                print(
                    "LLM_PEFT_CLIENT_DONE: "
                    f"round={round_num} client={client_id} "
                    f"train_loss={float(update.metrics.get('train_loss', 0.0)):.6f} "
                    f"records={update.num_examples} tokens={update.num_tokens}"
                )

            global_adapter_state = aggregate_adapter_state_dicts(
                updates,
                strategy=runtime_config.federated.aggregation,
            )
            adapter_size = adapter_state_size_bytes(global_adapter_state)
            adapter_path = save_adapter_artifact(
                adapter_dir / f"round_{round_num}_global_adapter.pt",
                global_adapter_state,
                metadata={
                    "round": round_num,
                    "selected_clients": selected_ids,
                    "aggregation": runtime_config.federated.aggregation,
                    "base_model": runtime_config.llm.base_model,
                    "peft_method": runtime_config.peft.method,
                    "parent_adapter_path": parent_adapter_path,
                    "parent_adapter_sha256": parent_adapter_sha256,
                },
            )
            artifact_record = adapter_artifact_record(
                adapter_path,
                round_num=round_num,
                size_bytes=adapter_size,
                selected_clients=selected_ids,
                parent_path=parent_adapter_path,
                parent_sha256=parent_adapter_sha256,
            )
            metrics.setdefault("llm_artifacts", []).append(artifact_record)
            parent_adapter_path = artifact_record["path"]
            parent_adapter_sha256 = artifact_record["sha256"]
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            total_tokens = sum(update.num_tokens for update in updates)
            train_loss = _weighted_train_loss(updates)
            validation_loss = None
            if evaluation_records is not None:
                evaluation_metrics = trainer.evaluate_adapter(
                    records=evaluation_records,
                    adapter_state=global_adapter_state,
                    output_dir=work_dir / f"round_{round_num}" / "evaluation",
                )
                validation_loss = float(evaluation_metrics.get("validation_loss", 0.0))
                metrics["llm_evaluation"] = {
                    **metrics.get("llm_evaluation", {}),
                    "last_round": round_num,
                    "last_validation_loss": validation_loss,
                    "last_num_examples": int(evaluation_metrics.get("num_examples", len(evaluation_records))),
                    "last_num_tokens": int(evaluation_metrics.get("num_tokens", 0)),
                }
            append_llm_round_metrics(
                metrics,
                round_num=round_num,
                train_loss=train_loss,
                validation_loss=validation_loss,
                token_throughput=total_tokens / elapsed,
                adapter_size_bytes=adapter_size,
            )
            metrics["llm_runtime"] = {
                "status": "running",
                "last_round": round_num,
                "latest_adapter_path": artifact_record["path"],
                "latest_adapter_sha256": artifact_record["sha256"],
            }
            _write_metrics(runtime_config, metrics)
            print(f"LLM_PEFT_ROUND_DONE: round={round_num} train_loss={train_loss:.6f}")

        metrics["llm_runtime"] = {
            "status": "completed",
            "completed_rounds": runtime_config.federated.num_rounds,
            "latest_adapter_path": parent_adapter_path,
            "latest_adapter_sha256": parent_adapter_sha256,
        }
        _write_metrics(runtime_config, metrics)
        print("LLM_PEFT_RUNTIME_COMPLETED")
        print(f"\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230: {_result_path(runtime_config)}")
        return True
    except Exception as exc:
        print(f"LLM_PEFT_RUNTIME_FAILED: {exc}")
        return False


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a YAML/JSON object")
    return loaded


def _write_metrics(config: LlmPeftRuntimeConfig, payload: dict[str, Any]) -> None:
    result_path = _result_path(config)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _result_path(config: LlmPeftRuntimeConfig) -> Path:
    override_name = os.environ.get("FIGARO_RESULTS_FILE", "").strip()
    if override_name:
        return PROJECT_ROOT / config.results_dir / Path(override_name).name
    return PROJECT_ROOT / config.results_dir / "live_results_llm_peft.json"


def _runtime_work_dir(config: LlmPeftRuntimeConfig) -> Path:
    path = PROJECT_ROOT / config.results_dir / "llm_peft_work" / _run_artifact_id()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _adapter_dir(config: LlmPeftRuntimeConfig) -> Path:
    path = PROJECT_ROOT / config.results_dir / "llm_peft_adapters" / _run_artifact_id()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_runtime_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _effective_sft_file_format(path: Path, *, configured_file_format: str) -> str:
    if configured_file_format != "auto":
        return configured_file_format
    if path.is_file():
        detected = _sft_file_format_from_suffix(path)
        return detected or "auto"
    if path.is_dir():
        formats = {
            detected
            for candidate in path.rglob("*")
            if candidate.is_file()
            and (detected := _sft_file_format_from_suffix(candidate)) is not None
        }
        if len(formats) == 1:
            return next(iter(formats))
        if len(formats) > 1:
            return "mixed"
    return "auto"


def _sft_file_format_from_suffix(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    return None


def _run_artifact_id() -> str:
    override_name = os.environ.get("FIGARO_RESULTS_FILE", "").strip()
    if not override_name:
        return "manual"
    stem = Path(override_name).name
    if stem.endswith(".json"):
        stem = stem[:-5]
    prefix = "live_results_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return sanitized or "manual"


def _select_clients(client_ids: list[int], *, clients_per_round: int, rng: random.Random) -> list[int]:
    selected_count = min(clients_per_round, len(client_ids))
    return sorted(rng.sample(client_ids, selected_count))


def _append_client_metrics(metrics: dict[str, Any], *, client_id: int, update) -> None:
    client_key = f"client_{client_id}"
    clients = metrics.setdefault("client_results", {})
    series = clients.setdefault(
        client_key,
        {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []},
    )
    train_loss = float(update.metrics.get("train_loss", 0.0))
    series["train_loss"].append(train_loss)
    series["train_acc"].append(0.0)
    series["test_loss"].append(0.0)
    series["test_acc"].append(0.0)


def _weighted_train_loss(updates) -> float:
    total_examples = sum(max(0, int(update.num_examples)) for update in updates)
    if total_examples <= 0:
        return 0.0
    weighted = 0.0
    for update in updates:
        weighted += float(update.metrics.get("train_loss", 0.0)) * max(0, int(update.num_examples))
    return weighted / total_examples
