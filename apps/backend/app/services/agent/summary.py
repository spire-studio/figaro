"""
Result scoring and summary helpers for the bench-mode experiment executor.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState, ExperimentRecord


def get_last_global_accuracy(metrics: dict[str, Any]) -> float | None:
    """Extract the final global accuracy from run metrics."""
    global_results = metrics.get("global_results") or {}
    accuracy = global_results.get("global_accuracy") or []
    if not accuracy:
        return None
    last = accuracy[-1]
    return float(last) if isinstance(last, (int, float)) else None


def get_last_llm_train_loss(metrics: dict[str, Any]) -> float | None:
    """Extract the final LLM train loss from run metrics."""
    llm_results = metrics.get("llm_results") or {}
    train_loss = llm_results.get("train_loss") or []
    if not train_loss:
        return None
    last = train_loss[-1]
    return float(last) if isinstance(last, (int, float)) else None


def get_last_llm_validation_loss(metrics: dict[str, Any]) -> float | None:
    """Extract the final non-zero LLM validation loss from run metrics."""
    llm_results = metrics.get("llm_results") or {}
    validation_loss = llm_results.get("validation_loss") or []
    for value in reversed(validation_loss):
        if isinstance(value, (int, float)) and float(value) > 0:
            return float(value)
    return None


def get_last_llm_loss(metrics: dict[str, Any]) -> float | None:
    """Return validation loss when present, otherwise train loss."""
    return get_last_llm_validation_loss(metrics) or get_last_llm_train_loss(metrics)


def get_agent_run_score(metrics: dict[str, Any]) -> float | None:
    """Return the comparable score used by Agent result ranking.

    Classic FL keeps higher-is-better accuracy. LLM PEFT uses negative loss
    so lower loss ranks higher while preserving the same max() flow.
    """
    llm_loss = get_last_llm_loss(metrics)
    if llm_loss is not None:
        return -llm_loss
    return get_last_global_accuracy(metrics)


def _format_score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _is_llm_record(record: ExperimentRecord) -> bool:
    return get_last_llm_loss(record.metrics) is not None


def _format_record_metric(record: ExperimentRecord) -> str:
    validation_loss = get_last_llm_validation_loss(record.metrics)
    if validation_loss is not None:
        return f"val_loss={validation_loss:.4f}"
    llm_loss = get_last_llm_train_loss(record.metrics)
    if llm_loss is not None:
        return f"train_loss={llm_loss:.4f}"
    return _format_score(record.score)


def _get_federated_summary(config: dict[str, Any]) -> str:
    federated = config.get("federated") or {}
    if not isinstance(federated, dict):
        return "unknown"

    num_clients = federated.get("num_clients", "-")
    clients_per_round = federated.get("clients_per_round", "-")
    num_rounds = federated.get("num_rounds", "-")
    local_epochs = federated.get("local_epochs", "-")
    learning_rate = federated.get("learning_rate", "-")
    return (
        f"clients={num_clients}, per_round={clients_per_round}, "
        f"rounds={num_rounds}, local_epochs={local_epochs}, lr={learning_rate}"
    )


def build_results_table(state: AgentState) -> str:
    """
    Build a comparison table across all experiment results.
    """
    if not state.experiment_results:
        return "No experiment results available."

    lines: list[str] = []
    metric_header = "Metric" if any(_is_llm_record(record) for record in state.experiment_results) else "Accuracy"
    header = f"{'Name':<30} {metric_header:<12} {'Rounds':<8} {'Config Summary'}"
    lines.append(header)
    lines.append("-" * len(header))

    for record in state.experiment_results:
        name = record.name or f"exp-{record.iteration}"
        score = _format_record_metric(record)
        global_results = record.metrics.get("global_results") or {}
        llm_results = record.metrics.get("llm_results") or {}
        rounds = len((llm_results.get("rounds") if isinstance(llm_results, dict) else None) or global_results.get("rounds") or [])
        config_summary = _get_federated_summary(record.config)
        lines.append(f"{name:<30} {score:<12} {rounds:<8} {config_summary}")

    return "\n".join(lines)


def build_summary_text(*, state: AgentState) -> str:
    """Build a job-level summary across all experiment results."""
    if not state.experiment_results:
        return "No experiment results to summarize."

    total = len(state.experiment_results)
    scored = [r for r in state.experiment_results if r.score is not None]

    lines: list[str] = []
    lines.append(f"Completed {total} experiments.")

    if scored:
        best = max(scored, key=lambda r: r.score)  # type: ignore[arg-type]
        worst = min(scored, key=lambda r: r.score)  # type: ignore[arg-type]
        if any(_is_llm_record(record) for record in scored):
            lines.append(
                f"Best LLM loss: {_format_record_metric(best)} ({best.name or f'exp-{best.iteration}'})"
            )
            lines.append(
                f"Worst LLM loss: {_format_record_metric(worst)} ({worst.name or f'exp-{worst.iteration}'})"
            )
        else:
            lines.append(
                f"Best accuracy: {_format_score(best.score)} ({best.name or f'exp-{best.iteration}'})"
            )
            lines.append(
                f"Worst accuracy: {_format_score(worst.score)} ({worst.name or f'exp-{worst.iteration}'})"
            )

    lines.append("")
    lines.append(build_results_table(state))

    return "\n".join(lines)


# --- Compatibility shims used by runtime_service ---


def update_best_result(
    state: AgentState, latest: ExperimentRecord
) -> tuple[float | None, bool]:
    """No-op compatibility shim. Returns (None, False)."""
    return None, False


def build_iteration_summary_text(
    *,
    state: AgentState,
    latest: ExperimentRecord,
    latest_score: float | None,
    previous_best_score: float | None,
    is_new_best: bool,
) -> str:
    """Simple per-experiment summary."""
    name = latest.name or f"exp-{latest.iteration}"
    return (
        f"Experiment '{name}' completed with "
        f"global accuracy {_format_score(latest_score)}."
    )


def derive_iteration_decision(
    *,
    objective: Any = None,
    latest_score: float | None = None,
    previous_best_score: float | None = None,
    is_new_best: bool = False,
) -> str:
    """Bench mode does not optimize — all results are recorded."""
    return "recorded"


def build_lessons_learned(
    *,
    state: AgentState,
    latest: ExperimentRecord,
    previous_best_score: float | None = None,
    is_new_best: bool = False,
) -> list[str]:
    """Bench mode does not generate iterative lessons."""
    return []

def calculate_communication_cost(config: dict[str, Any]) -> int:
    """Estimate communication cost: total rounds * clients per round."""
    federated = config.get("federated") or {}
    rounds = int(federated.get("num_rounds", 0))
    clients = int(federated.get("clients_per_round", 0))
    return rounds * clients

def calculate_computation_cost(config: dict[str, Any]) -> int:
    """Estimate computation cost: total rounds * clients per round * local epochs."""
    federated = config.get("federated") or {}
    epochs = int(federated.get("local_epochs", 0))
    return calculate_communication_cost(config) * epochs

def get_best_experiment(state: AgentState) -> ExperimentRecord | None:
    """Find the best performing experiment (the Winner)."""
    scored = [r for r in state.experiment_results if r.score is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: r.score)
