#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from core_runtime import FederatedLearningFramework


def _set_value_by_path(obj: dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.split(".") if part]
    current = obj
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a YAML/JSON object")
    return loaded


def _apply_runtime_profile(config: dict[str, Any], *, mode: str, role: str | None) -> dict[str, Any]:
    profiled = dict(config)
    _set_value_by_path(profiled, "system.mode", mode)

    if mode == "simulation":
        _set_value_by_path(profiled, "system.role", "server")
        _set_value_by_path(profiled, "system.node_role", "server")
        return profiled

    resolved_role = role or "server"
    _set_value_by_path(profiled, "system.role", resolved_role)
    _set_value_by_path(profiled, "system.node_role", resolved_role)
    return profiled


def _run_framework(config_path: Path) -> int:
    framework = FederatedLearningFramework(config_path=str(config_path))
    return 0 if framework.run() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Phoenix runtime entrypoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to runtime config file")
    parser.add_argument("--mode", choices=("simulation", "distributed"), required=True)
    parser.add_argument("--role", choices=("server", "client"), default=None)
    parser.add_argument(
        "--disable-status-server",
        action="store_true",
        help="Reserved runtime flag; currently a no-op.",
    )
    args = parser.parse_args()

    config_json = _load_config(args.config)
    profiled = _apply_runtime_profile(config_json, mode=args.mode, role=args.role)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fp:
            yaml.safe_dump(profiled, fp, allow_unicode=True, sort_keys=False)
            tmp_path = Path(fp.name)
        return _run_framework(tmp_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
