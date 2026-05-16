from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

RUNNERS_DIR = Path(__file__).resolve().parents[2] / "runners"
if str(RUNNERS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNERS_DIR))

from runtime_dispatcher import get_runtime_mode, get_task_type, run_runtime  # noqa: E402


def _write_config(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_get_task_type_defaults_legacy_configs_to_classic_fl():
    assert get_task_type({}) == "classic_fl"
    assert get_task_type({"task": {}}) == "classic_fl"


def test_get_runtime_mode_defaults_legacy_configs_to_simulation():
    assert get_runtime_mode({}) == "simulation"
    assert get_runtime_mode({"system": {}}) == "simulation"


def test_run_runtime_dispatches_classic_route(tmp_path, monkeypatch):
    called = {}

    def fake_run(path: Path) -> bool:
        called["path"] = path
        return True

    fake_module = types.SimpleNamespace(run_classic_fl_runtime=fake_run)
    monkeypatch.setitem(sys.modules, "classic_fl_runtime", fake_module)

    config_path = _write_config(tmp_path, {"task": {"type": "classic_fl"}})

    assert run_runtime(config_path) is True
    assert called["path"] == config_path


def test_run_runtime_dispatches_llm_peft_route(tmp_path, monkeypatch):
    called = {}

    def fake_run(path: Path) -> bool:
        called["path"] = path
        return False

    fake_module = types.SimpleNamespace(run_llm_peft_runtime=fake_run)
    monkeypatch.setitem(sys.modules, "llm_peft_runtime", fake_module)

    config_path = _write_config(tmp_path, {"task": {"type": "llm_peft_sft"}})

    assert run_runtime(config_path) is False
    assert called["path"] == config_path


def test_run_runtime_rejects_llm_peft_distributed_route(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "task": {"type": "llm_peft_sft"},
            "system": {"mode": "distributed"},
        },
    )

    with pytest.raises(ValueError, match="simulation mode only"):
        run_runtime(config_path)


def test_run_runtime_rejects_unknown_task_type(tmp_path):
    config_path = _write_config(tmp_path, {"task": {"type": "unknown"}})

    with pytest.raises(ValueError, match="Unsupported task.type"):
        run_runtime(config_path)
