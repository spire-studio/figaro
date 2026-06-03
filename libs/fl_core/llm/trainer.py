from __future__ import annotations

import importlib.util
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .aggregation import AdapterClientUpdate
from .config import LlmPeftRuntimeConfig
from .data import SftRecord, format_sft_record_text
from .modeling import require_llm_runtime_dependencies


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TokenizedSftDataset(Dataset):
    """Tiny torch Dataset wrapper for tokenizer-ready SFT records."""

    def __init__(
        self,
        records: list[SftRecord],
        *,
        tokenizer: Any,
        data_format: str,
        prompt_template: str,
        max_seq_length: int,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        self._items: list[dict[str, list[int]]] = []
        self.num_tokens = 0

        for record in records:
            text = format_sft_record_text(
                record,
                data_format=data_format,
                prompt_template=prompt_template,
            )
            encoded = _tokenize_text(tokenizer, text, max_seq_length=max_seq_length)
            input_ids = list(encoded.get("input_ids") or [])
            attention_mask = list(encoded.get("attention_mask") or [1] * len(input_ids))
            if not input_ids:
                continue
            labels = _labels_for_record(
                record,
                input_ids=input_ids,
                tokenizer=tokenizer,
                data_format=data_format,
                max_seq_length=max_seq_length,
            )
            self.num_tokens += len(input_ids)
            self._items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        if not self._items:
            raise ValueError("No non-empty tokenized SFT records")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._items[index]


class CausalLmSftDataCollator:
    """Pad causal-LM SFT batches while preserving label masks."""

    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * pad_len)
            attention_mask.append(list(feature.get("attention_mask") or [1] * len(feature["input_ids"])) + [0] * pad_len)
            labels.append(list(feature["labels"]) + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class LlmPeftTrainer:
    """Training facade for a single-client Transformers/PEFT LoRA loop."""

    config: LlmPeftRuntimeConfig

    def ensure_ready(self) -> None:
        """Validate that LLM runtime dependencies are installed."""
        require_llm_runtime_dependencies()
        if self.config.peft.quantization != "none" and importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("Missing QLoRA runtime dependency: bitsandbytes")

    def train_client(
        self,
        *,
        client_id: int,
        records: list[SftRecord],
        round_num: int,
        initial_adapter_state: dict[str, torch.Tensor] | None = None,
        output_dir: str | Path | None = None,
    ) -> AdapterClientUpdate:
        """Run local LoRA SFT for one client and return adapter-only weights."""
        self.ensure_ready()

        transformers = self._transformers_module()
        peft = self._peft_module()
        tokenizer, model = self._build_lora_model(
            transformers,
            peft,
            adapter_state=initial_adapter_state,
        )

        dataset = self._tokenized_dataset(records, tokenizer)
        data_collator = CausalLmSftDataCollator(tokenizer)

        train_output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="figaro-llm-client-"))
        train_output_dir.mkdir(parents=True, exist_ok=True)
        training_args = transformers.TrainingArguments(
            output_dir=str(train_output_dir),
            per_device_train_batch_size=self.config.sft.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.sft.gradient_accumulation_steps,
            num_train_epochs=self.config.federated.local_epochs,
            learning_rate=self.config.federated.learning_rate,
            logging_strategy="no",
            save_strategy="no",
            disable_tqdm=True,
            report_to=[],
            remove_unused_columns=False,
            fp16=self.config.llm.precision == "fp16" and torch.cuda.is_available(),
            bf16=self.config.llm.precision == "bf16" and torch.cuda.is_available(),
        )
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        train_output = trainer.train()
        adapter_state = {
            key: value.detach().cpu()
            for key, value in peft.get_peft_model_state_dict(model).items()
        }
        train_loss = _extract_train_loss(train_output)
        return AdapterClientUpdate(
            adapter_state=adapter_state,
            num_examples=len(records),
            num_tokens=dataset.num_tokens,
            metrics={
                "client_id": client_id,
                "round": round_num,
                "train_loss": train_loss,
                "num_examples": len(records),
                "num_tokens": dataset.num_tokens,
            },
        )

    def evaluate_adapter(
        self,
        *,
        records: list[SftRecord],
        adapter_state: dict[str, torch.Tensor] | None,
        output_dir: str | Path | None = None,
    ) -> dict[str, float]:
        """Evaluate a global LoRA adapter on validation SFT records."""
        self.ensure_ready()

        transformers = self._transformers_module()
        peft = self._peft_module()
        tokenizer, model = self._build_lora_model(
            transformers,
            peft,
            adapter_state=adapter_state,
        )

        dataset = self._tokenized_dataset(records, tokenizer)
        data_collator = CausalLmSftDataCollator(tokenizer)

        eval_output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="figaro-llm-eval-"))
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        training_args = transformers.TrainingArguments(
            output_dir=str(eval_output_dir),
            per_device_eval_batch_size=self.config.evaluation.batch_size,
            disable_tqdm=True,
            report_to=[],
            remove_unused_columns=False,
            fp16=self.config.llm.precision == "fp16" and torch.cuda.is_available(),
            bf16=self.config.llm.precision == "bf16" and torch.cuda.is_available(),
        )
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            eval_dataset=dataset,
            data_collator=data_collator,
        )
        metrics = trainer.evaluate()
        loss = _extract_eval_loss(metrics)
        return {
            "validation_loss": loss,
            "num_examples": float(len(records)),
            "num_tokens": float(dataset.num_tokens),
        }

    def _transformers_module(self) -> Any:
        with _without_project_root_on_sys_path():
            transformers = __import__("transformers", fromlist=[
                "AutoModelForCausalLM",
                "AutoTokenizer",
                "DataCollatorForLanguageModeling",
                "Trainer",
                "TrainingArguments",
            ])
        _quiet_transformers(transformers)
        return transformers

    def _peft_module(self) -> Any:
        return __import__("peft", fromlist=[
            "LoraConfig",
            "TaskType",
            "get_peft_model",
            "get_peft_model_state_dict",
            "set_peft_model_state_dict",
            "prepare_model_for_kbit_training",
        ])

    def _build_lora_model(
        self,
        transformers: Any,
        peft: Any,
        *,
        adapter_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[Any, Any]:
        tokenizer = self._load_tokenizer(transformers)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.llm.base_model,
            **self._model_load_kwargs(transformers),
        )
        if self.config.peft.quantization != "none":
            model = peft.prepare_model_for_kbit_training(model)

        lora_config = peft.LoraConfig(
            r=self.config.peft.rank,
            lora_alpha=self.config.peft.alpha,
            lora_dropout=self.config.peft.dropout,
            target_modules=list(self.config.peft.target_modules),
            task_type=peft.TaskType.CAUSAL_LM,
        )
        model = peft.get_peft_model(model, lora_config)
        if adapter_state:
            peft.set_peft_model_state_dict(model, adapter_state)
        return tokenizer, model

    def _load_tokenizer(self, transformers: Any) -> Any:
        tokenizer_name = self.config.llm.base_model if self.config.llm.tokenizer == "auto" else self.config.llm.tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
        return tokenizer

    def _tokenized_dataset(self, records: list[SftRecord], tokenizer: Any) -> TokenizedSftDataset:
        return TokenizedSftDataset(
            records,
            tokenizer=tokenizer,
            data_format=self.config.sft.format,
            prompt_template=self.config.sft.prompt_template,
            max_seq_length=self.config.llm.max_seq_length,
        )

    def _model_load_kwargs(self, transformers: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        dtype = self._torch_dtype()
        if dtype is not None:
            kwargs["dtype"] = dtype

        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

        if self.config.peft.quantization == "int8":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.peft.quantization == "nf4_4bit":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype or torch.float16,
            )
        return kwargs

    def _torch_dtype(self):
        if self.config.llm.precision == "fp16":
            return torch.float16
        if self.config.llm.precision == "bf16":
            return torch.bfloat16
        return None


def _extract_train_loss(train_output: Any) -> float:
    if hasattr(train_output, "training_loss"):
        try:
            return float(train_output.training_loss)
        except (TypeError, ValueError):
            pass
    metrics = getattr(train_output, "metrics", None)
    if isinstance(metrics, dict):
        for key in ("train_loss", "loss"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    continue
    return 0.0


def _quiet_transformers(transformers: Any) -> None:
    logging_module = getattr(transformers, "logging", None)
    if logging_module is None:
        return

    disable_progress_bar = getattr(logging_module, "disable_progress_bar", None)
    if callable(disable_progress_bar):
        disable_progress_bar()

    set_verbosity_error = getattr(logging_module, "set_verbosity_error", None)
    if callable(set_verbosity_error):
        set_verbosity_error()


def _tokenize_text(
    tokenizer: Any,
    text: str,
    *,
    max_seq_length: int,
    add_special_tokens: bool | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "truncation": True,
        "max_length": max_seq_length,
        "padding": False,
    }
    if add_special_tokens is not None:
        kwargs["add_special_tokens"] = add_special_tokens
    try:
        return tokenizer(text, **kwargs)
    except TypeError:
        kwargs.pop("add_special_tokens", None)
        return tokenizer(text, **kwargs)


def _labels_for_record(
    record: SftRecord,
    *,
    input_ids: list[int],
    tokenizer: Any,
    data_format: str,
    max_seq_length: int,
) -> list[int]:
    labels = list(input_ids)
    if data_format not in {"prompt_completion", "alpaca"}:
        return labels

    prompt_text = record.prompt or ""
    if not prompt_text:
        return labels

    prompt_encoded = _tokenize_text(
        tokenizer,
        prompt_text,
        max_seq_length=max_seq_length,
        add_special_tokens=False,
    )
    prompt_length = min(len(list(prompt_encoded.get("input_ids") or [])), len(labels))
    for index in range(prompt_length):
        labels[index] = -100
    return labels


def _extract_eval_loss(metrics: Any) -> float:
    if isinstance(metrics, dict):
        for key in ("eval_loss", "validation_loss", "loss"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    continue
    return 0.0


@contextmanager
def _without_project_root_on_sys_path():
    """Avoid shadowing optional third-party packages with repo data directories."""
    original_path = list(sys.path)
    sys.path[:] = [
        entry
        for entry in original_path
        if _resolve_sys_path_entry(entry) != PROJECT_ROOT
    ]
    _drop_local_datasets_module()
    try:
        yield
    finally:
        sys.path[:] = original_path


def _resolve_sys_path_entry(entry: str) -> Path | None:
    try:
        return Path(entry or ".").resolve()
    except OSError:
        return None


def _drop_local_datasets_module() -> None:
    module = sys.modules.get("datasets")
    if module is None:
        return

    module_paths = [getattr(module, "__file__", None)]
    module_paths.extend(list(getattr(module, "__path__", []) or []))
    for value in module_paths:
        if value is None:
            continue
        try:
            path = Path(value).resolve()
        except OSError:
            continue
        if path == PROJECT_ROOT / "datasets" or path.is_relative_to(PROJECT_ROOT / "datasets"):
            sys.modules.pop("datasets", None)
            return
