from __future__ import annotations

import math
from typing import Any

from .config import LlmPeftRuntimeConfig


def empty_llm_metrics_payload(config: LlmPeftRuntimeConfig | None = None) -> dict[str, Any]:
    """Return a metrics payload compatible with Figaro plus LLM-specific series."""
    experiment_info = {
        "basic": {"task_type": "llm_peft_sft"},
        "federated": {},
        "security": {},
    }
    if config is not None:
        experiment_info["basic"].update(
            {
                "base_model": config.llm.base_model,
                "sft_format": config.sft.format,
                "peft_method": config.peft.method,
            }
        )
        experiment_info["federated"] = {
            "num_rounds": config.federated.num_rounds,
            "num_clients": config.federated.num_clients,
            "clients_per_round": config.federated.clients_per_round,
            "local_epochs": config.federated.local_epochs,
            "learning_rate": config.federated.learning_rate,
            "aggregation": config.federated.aggregation,
        }

    return {
        "experiment_info": experiment_info,
        "global_results": {
            "rounds": [],
            "global_loss": [],
            "global_accuracy": [],
        },
        "llm_results": {
            "rounds": [],
            "train_loss": [],
            "validation_loss": [],
            "perplexity": [],
            "token_throughput": [],
            "adapter_size_bytes": [],
        },
        "llm_dataset": {},
        "llm_evaluation": {},
        "llm_runtime": {},
        "llm_artifacts": [],
        "client_results": {},
    }


def append_llm_round_metrics(
    payload: dict[str, Any],
    *,
    round_num: int,
    train_loss: float,
    validation_loss: float | None = None,
    token_throughput: float | None = None,
    adapter_size_bytes: int | None = None,
) -> None:
    """Append one global LLM round to a metrics payload in-place."""
    llm_results = payload.setdefault("llm_results", {})
    llm_results.setdefault("rounds", []).append(int(round_num))
    llm_results.setdefault("train_loss", []).append(float(train_loss))
    if validation_loss is not None:
        llm_results.setdefault("validation_loss", []).append(float(validation_loss))
    else:
        llm_results.setdefault("validation_loss", [])
    llm_results.setdefault("perplexity", []).append(_perplexity(validation_loss if validation_loss is not None else train_loss))
    llm_results.setdefault("token_throughput", []).append(float(token_throughput) if token_throughput is not None else 0.0)
    llm_results.setdefault("adapter_size_bytes", []).append(int(adapter_size_bytes or 0))

    global_results = payload.setdefault("global_results", {})
    global_results.setdefault("rounds", []).append(int(round_num))
    global_results.setdefault("global_loss", []).append(float(validation_loss if validation_loss is not None else train_loss))
    global_results.setdefault("global_accuracy", []).append(0.0)


def _perplexity(loss: float) -> float:
    try:
        return float(math.exp(float(loss)))
    except OverflowError:
        return float("inf")
