from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
import torch

RUNNERS_DIR = Path(__file__).resolve().parents[2] / "runners"
if str(RUNNERS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNERS_DIR))

import llm_peft_runtime  # noqa: E402
from fl_core.llm.aggregation import AdapterClientUpdate  # noqa: E402
from fl_core.llm.artifacts import save_adapter_artifact, sha256_file  # noqa: E402


def test_llm_peft_runtime_writes_blocked_metrics_for_missing_dependencies(tmp_path, monkeypatch):
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "hello", "completion": "world"}),
                json.dumps({"prompt": "foo", "completion": "bar"}),
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"type": "llm_peft_sft"},
                "llm": {"base_model": "tiny-model", "tokenizer": "auto", "max_seq_length": 128, "precision": "fp32"},
                "sft": {"dataset_path": str(dataset_path), "format": "prompt_completion", "prompt_template": "plain"},
                "peft": {
                    "method": "lora",
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": "q_proj",
                    "quantization": "none",
                },
                "federated": {
                    "num_clients": 2,
                    "num_rounds": 1,
                    "clients_per_round": 1,
                    "local_epochs": 1,
                    "learning_rate": 0.0002,
                    "aggregation": "fedavg",
                    "seed": 1,
                },
                "logging": {"results_dir": "results"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_peft_runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("FIGARO_RESULTS_FILE", "live_results_test.json")
    monkeypatch.setattr(llm_peft_runtime, "missing_llm_runtime_dependencies", lambda: ["transformers", "peft"])

    assert llm_peft_runtime.run_llm_peft_runtime(config_path) is False

    metrics_path = tmp_path / "results" / "live_results_test.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["llm_dataset"]["num_records"] == 2
    assert metrics["llm_dataset"]["client_record_counts"] == [1, 1]
    assert metrics["llm_runtime"]["reason"] == "missing_llm_runtime_dependencies"
    assert metrics["llm_runtime"]["missing_dependencies"] == ["transformers", "peft"]


def test_llm_peft_runtime_runs_client_training_and_aggregates_adapters(tmp_path, monkeypatch):
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "p0", "completion": "c0"}),
                json.dumps({"prompt": "p1", "completion": "c1"}),
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"type": "llm_peft_sft"},
                "llm": {"base_model": "tiny-model", "tokenizer": "auto", "max_seq_length": 128, "precision": "fp32"},
                "sft": {
                    "dataset_path": str(dataset_path),
                    "format": "prompt_completion",
                    "prompt_template": "plain",
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                },
                "peft": {
                    "method": "lora",
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": "q_proj",
                    "quantization": "none",
                },
                "federated": {
                    "num_clients": 2,
                    "num_rounds": 1,
                    "clients_per_round": 2,
                    "local_epochs": 1,
                    "learning_rate": 0.0002,
                    "aggregation": "fedavg",
                    "seed": 1,
                },
                "logging": {"results_dir": "results"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def train_client(self, *, client_id, records, round_num, initial_adapter_state=None, output_dir=None):
            assert round_num == 1
            assert initial_adapter_state is None
            assert output_dir is not None
            value = float(client_id + 1)
            return AdapterClientUpdate(
                adapter_state={"lora_A": torch.tensor([value])},
                num_examples=len(records),
                num_tokens=10,
                metrics={"train_loss": value},
            )

    monkeypatch.setattr(llm_peft_runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("FIGARO_RESULTS_FILE", "live_results_test.json")
    monkeypatch.setattr(llm_peft_runtime, "missing_llm_runtime_dependencies", lambda: [])
    monkeypatch.setattr(llm_peft_runtime, "LlmPeftTrainer", FakeTrainer)

    assert llm_peft_runtime.run_llm_peft_runtime(config_path) is True

    metrics_path = tmp_path / "results" / "live_results_test.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["llm_results"]["rounds"] == [1]
    assert metrics["llm_results"]["train_loss"] == [1.5]
    assert metrics["llm_runtime"]["status"] == "completed"
    assert len(metrics["llm_artifacts"]) == 1
    assert metrics["llm_artifacts"][0]["round"] == 1
    assert metrics["llm_artifacts"][0]["selected_clients"] == [0, 1]
    adapter_path = Path(metrics["llm_runtime"]["latest_adapter_path"])
    assert adapter_path.exists()
    assert metrics["llm_runtime"]["latest_adapter_sha256"] == sha256_file(adapter_path)


def test_llm_peft_runtime_resumes_from_adapter_artifact(tmp_path, monkeypatch):
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(json.dumps({"prompt": "p0", "completion": "c0"}), encoding="utf-8")
    resume_path = save_adapter_artifact(
        tmp_path / "resume_adapter.pt",
        {"lora_A": torch.tensor([9.0])},
        metadata={"round": 7},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"type": "llm_peft_sft"},
                "llm": {"base_model": "tiny-model", "tokenizer": "auto", "max_seq_length": 128, "precision": "fp32"},
                "sft": {
                    "dataset_path": str(dataset_path),
                    "format": "prompt_completion",
                    "prompt_template": "plain",
                },
                "peft": {
                    "method": "lora",
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": "q_proj",
                    "resume_adapter_path": str(resume_path),
                    "quantization": "none",
                },
                "federated": {
                    "num_clients": 1,
                    "num_rounds": 1,
                    "clients_per_round": 1,
                    "local_epochs": 1,
                    "learning_rate": 0.0002,
                    "aggregation": "fedavg",
                    "seed": 1,
                },
                "logging": {"results_dir": "results"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def train_client(self, *, client_id, records, round_num, initial_adapter_state=None, output_dir=None):
            assert initial_adapter_state is not None
            assert torch.allclose(initial_adapter_state["lora_A"], torch.tensor([9.0]))
            return AdapterClientUpdate(
                adapter_state={"lora_A": torch.tensor([10.0])},
                num_examples=len(records),
                num_tokens=5,
                metrics={"train_loss": 0.5},
            )

    monkeypatch.setattr(llm_peft_runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("FIGARO_RESULTS_FILE", "live_results_resume.json")
    monkeypatch.setattr(llm_peft_runtime, "missing_llm_runtime_dependencies", lambda: [])
    monkeypatch.setattr(llm_peft_runtime, "LlmPeftTrainer", FakeTrainer)

    assert llm_peft_runtime.run_llm_peft_runtime(config_path) is True

    metrics = json.loads((tmp_path / "results" / "live_results_resume.json").read_text(encoding="utf-8"))
    artifact = metrics["llm_artifacts"][0]
    assert artifact["parent_path"] == str(resume_path)
    assert artifact["parent_sha256"] == sha256_file(resume_path)
    assert metrics["llm_runtime"]["latest_adapter_sha256"] == artifact["sha256"]
