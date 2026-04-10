"""
Planning helpers for the bench-mode FL experiment executor.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings


def build_initial_config(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Build a default FL experiment config from the simulation schema.
    """
    federated = schema.get("federated") or {}
    dataset = schema.get("dataset") or {}
    model = schema.get("model") or {}
    return {
        "dataset": {"name": dataset.get("name", {}).get("default", "cifar10")},
        "model": {"name": model.get("name", {}).get("default", "cnn")},
        "federated": {
            "num_clients": federated.get("num_clients", {}).get("default", 10),
            "num_rounds": federated.get("num_rounds", {}).get("default", 20),
            "clients_per_round": federated.get("clients_per_round", {}).get("default", 5),
            "local_epochs": federated.get("local_epochs", {}).get("default", 5),
            "learning_rate": federated.get("learning_rate", {}).get("default", 0.01),
        },
    }


def select_llm_model(model_name: str | None) -> str:
    """Resolve the LLM model for planning requests."""
    if model_name and model_name.strip():
        return model_name.strip()
    return settings.default_llm_model
