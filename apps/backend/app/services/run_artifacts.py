"""Shared naming helpers for run-local config, log, and result files."""

from __future__ import annotations

from datetime import datetime, timezone


def run_artifact_timestamp(value: datetime | None = None) -> str:
    """Return a stable UTC timestamp prefix for one run's local artifacts."""
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y%m%d_%H%M%SZ")


def run_artifact_stem(run_id: str, *, timestamp: str) -> str:
    return f"{timestamp}_{run_id}"


def legacy_run_artifact_stem(run_id: str, *, timestamp: str) -> str:
    return f"{run_id}_{timestamp}"


def run_config_filename(run_id: str, *, timestamp: str) -> str:
    return f"{run_artifact_stem(run_id, timestamp=timestamp)}.json"


def run_log_filename(run_id: str, *, timestamp: str, role: str) -> str:
    return f"{run_artifact_stem(run_id, timestamp=timestamp)}_{role}.log"


def live_results_filename(run_id: str, *, timestamp: str) -> str:
    return f"{run_artifact_stem(run_id, timestamp=timestamp)}_live_results.json"


def legacy_live_results_filename(run_id: str, *, timestamp: str) -> str:
    return f"live_results_{legacy_run_artifact_stem(run_id, timestamp=timestamp)}.json"
