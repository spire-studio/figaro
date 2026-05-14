from __future__ import annotations

import importlib.util
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .aggregation import AdapterClientUpdate
from .config import LlmPeftRuntimeConfig
from .data import SftRecord, format_sft_record_text
from .modeling import require_optional_dependencies


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
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )
            input_ids = list(encoded.get("input_ids") or [])
            attention_mask = list(encoded.get("attention_mask") or [1] * len(input_ids))
            if not input_ids:
                continue
            self.num_tokens += len(input_ids)
            self._items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": list(input_ids),
                }
            )

        if not self._items:
            raise ValueError("No non-empty tokenized SFT records")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._items[index]


@dataclass
class LlmPeftTrainer:
    """Training facade for a single-client Transformers/PEFT LoRA loop."""

    config: LlmPeftRuntimeConfig

    def ensure_ready(self) -> None:
        """Validate that optional runtime dependencies are installed."""
        require_optional_dependencies()
        if self.config.peft.quantization != "none" and importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("Missing optional QLoRA dependency: bitsandbytes")

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

        transformers = __import__("transformers", fromlist=[
            "AutoModelForCausalLM",
            "AutoTokenizer",
            "DataCollatorForLanguageModeling",
            "Trainer",
            "TrainingArguments",
        ])
        peft = __import__("peft", fromlist=[
            "LoraConfig",
            "TaskType",
            "get_peft_model",
            "get_peft_model_state_dict",
            "set_peft_model_state_dict",
            "prepare_model_for_kbit_training",
        ])

        tokenizer_name = self.config.llm.base_model if self.config.llm.tokenizer == "auto" else self.config.llm.tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

        model_kwargs = self._model_load_kwargs(transformers)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.config.llm.base_model, **model_kwargs)
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
        if initial_adapter_state:
            peft.set_peft_model_state_dict(model, initial_adapter_state)

        dataset = TokenizedSftDataset(
            records,
            tokenizer=tokenizer,
            data_format=self.config.sft.format,
            prompt_template=self.config.sft.prompt_template,
            max_seq_length=self.config.llm.max_seq_length,
        )
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        train_output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="figaro-llm-client-"))
        train_output_dir.mkdir(parents=True, exist_ok=True)
        training_args = transformers.TrainingArguments(
            output_dir=str(train_output_dir),
            per_device_train_batch_size=self.config.sft.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.sft.gradient_accumulation_steps,
            num_train_epochs=self.config.federated.local_epochs,
            learning_rate=self.config.federated.learning_rate,
            logging_steps=1,
            save_strategy="no",
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

    def _model_load_kwargs(self, transformers: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        dtype = self._torch_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

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
