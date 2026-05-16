from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
LIBS_DIR = PROJECT_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from fl_core.llm.aggregation import AdapterClientUpdate, aggregate_adapter_state_dicts
from fl_core.llm.artifacts import (
    adapter_artifact_record,
    load_adapter_artifact,
    save_adapter_artifact,
    sha256_file,
)
from fl_core.llm.config import normalize_llm_peft_config, parse_target_modules
from fl_core.llm.data import SftRecord, format_sft_record_text, load_jsonl_sft_records, split_records_by_client
from fl_core.llm.metrics import append_llm_round_metrics, empty_llm_metrics_payload
from fl_core.llm.trainer import TokenizedSftDataset


def _base_config(tmp_path: Path) -> dict:
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "Q1", "completion": "A1"}),
                json.dumps({"prompt": "Q2", "completion": "A2"}),
            ]
        ),
        encoding="utf-8",
    )
    return {
        "task": {"type": "llm_peft_sft"},
        "llm": {
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "tokenizer": "auto",
            "max_seq_length": 512,
            "precision": "bf16",
        },
        "sft": {
            "dataset_path": str(dataset_path),
            "format": "prompt_completion",
            "prompt_template": "plain",
        },
        "peft": {
            "method": "lora",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": "q_proj, v_proj",
            "quantization": "none",
        },
        "federated": {
            "num_clients": 2,
            "num_rounds": 1,
            "clients_per_round": 1,
            "local_epochs": 1,
            "learning_rate": 0.0002,
            "aggregation": "fedavg",
            "seed": 7,
        },
    }


def test_normalize_llm_peft_config_parses_nested_sections(tmp_path):
    resume_path = tmp_path / "adapter.pt"
    config = _base_config(tmp_path)
    config["peft"]["resume_adapter_path"] = str(resume_path)

    normalized = normalize_llm_peft_config(config)

    assert normalized.task_type == "llm_peft_sft"
    assert normalized.llm.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert normalized.sft.dataset_path.name == "train.jsonl"
    assert normalized.sft.per_device_train_batch_size == 1
    assert normalized.sft.gradient_accumulation_steps == 1
    assert normalized.peft.target_modules == ("q_proj", "v_proj")
    assert normalized.peft.resume_adapter_path == resume_path
    assert normalized.federated.num_clients == 2


def test_parse_target_modules_rejects_empty_values():
    with pytest.raises(ValueError, match="target_modules"):
        parse_target_modules(" , ")


def test_load_jsonl_sft_records_prompt_completion_and_split(tmp_path):
    config = normalize_llm_peft_config(_base_config(tmp_path))
    records = load_jsonl_sft_records(config.sft.dataset_path, data_format=config.sft.format)
    splits = split_records_by_client(records, num_clients=2, seed=123)

    assert len(records) == 2
    assert sorted(len(split) for split in splits) == [1, 1]
    assert format_sft_record_text(records[0], data_format="prompt_completion", prompt_template="plain") == "Q1A1"


def test_load_jsonl_sft_records_messages_format(tmp_path):
    dataset_path = tmp_path / "messages.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            }
        ),
        encoding="utf-8",
    )

    [record] = load_jsonl_sft_records(dataset_path, data_format="messages")

    assert record.messages[0]["role"] == "user"
    assert "<|im_start|>user" in format_sft_record_text(record, data_format="messages", prompt_template="chatml")


def test_aggregate_adapter_state_dicts_weighted_by_examples():
    update_a = AdapterClientUpdate(
        adapter_state={"lora_A": torch.tensor([1.0, 3.0])},
        num_examples=1,
    )
    update_b = AdapterClientUpdate(
        adapter_state={"lora_A": torch.tensor([3.0, 5.0])},
        num_examples=3,
    )

    aggregated = aggregate_adapter_state_dicts([update_a, update_b])

    assert torch.allclose(aggregated["lora_A"], torch.tensor([2.5, 4.5]))


def test_aggregate_adapter_state_dicts_rejects_shape_mismatch():
    update_a = AdapterClientUpdate(adapter_state={"lora_A": torch.tensor([1.0])}, num_examples=1)
    update_b = AdapterClientUpdate(adapter_state={"lora_A": torch.tensor([[1.0]])}, num_examples=1)

    with pytest.raises(ValueError, match="mismatched shape"):
        aggregate_adapter_state_dicts([update_a, update_b])


def test_adapter_artifact_round_trip_and_lineage_record(tmp_path):
    artifact_path = save_adapter_artifact(
        tmp_path / "adapter.pt",
        {"lora_A": torch.tensor([1.0, 2.0])},
        metadata={"round": 1},
    )

    adapter_state, metadata = load_adapter_artifact(artifact_path)
    record = adapter_artifact_record(
        artifact_path,
        round_num=1,
        size_bytes=8,
        selected_clients=[0, 1],
        parent_path="previous.pt",
        parent_sha256="abc",
    )

    assert torch.allclose(adapter_state["lora_A"], torch.tensor([1.0, 2.0]))
    assert metadata["round"] == 1
    assert record["sha256"] == sha256_file(artifact_path)
    assert record["parent_sha256"] == "abc"


def test_empty_llm_metrics_payload_and_round_append(tmp_path):
    config = normalize_llm_peft_config(_base_config(tmp_path))
    payload = empty_llm_metrics_payload(config)
    append_llm_round_metrics(payload, round_num=1, train_loss=2.0, validation_loss=1.0, token_throughput=12.5)

    assert payload["experiment_info"]["basic"]["task_type"] == "llm_peft_sft"
    assert payload["llm_results"]["rounds"] == [1]
    assert payload["llm_artifacts"] == []
    assert payload["llm_results"]["perplexity"][0] == pytest.approx(2.71828, rel=1e-4)
    assert payload["global_results"]["global_loss"] == [1.0]


def test_tokenized_sft_dataset_uses_rendered_text():
    class FakeTokenizer:
        def __call__(self, text, *, truncation, max_length, padding):
            assert truncation is True
            assert padding is False
            input_ids = [ord(ch) for ch in text[:max_length]]
            return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    dataset = TokenizedSftDataset(
        [SftRecord(prompt="abc", completion="def")],
        tokenizer=FakeTokenizer(),
        data_format="prompt_completion",
        prompt_template="plain",
        max_seq_length=4,
    )

    item = dataset[0]
    assert item["input_ids"] == [97, 98, 99, 100]
    assert item["labels"] == item["input_ids"]
    assert dataset.num_tokens == 4
