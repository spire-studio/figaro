from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LLM_PEFT_TASK_TYPE = "llm_peft_sft"


@dataclass(frozen=True)
class LlmModelConfig:
    base_model: str
    tokenizer: str
    max_seq_length: int
    precision: str


@dataclass(frozen=True)
class SftDatasetConfig:
    dataset_path: Path
    format: str
    prompt_template: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class PeftAdapterConfig:
    method: str
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    resume_adapter_path: Path | None
    quantization: str


@dataclass(frozen=True)
class LlmFederatedConfig:
    num_clients: int
    num_rounds: int
    clients_per_round: int
    local_epochs: int
    learning_rate: float
    aggregation: str
    seed: int


@dataclass(frozen=True)
class LlmPeftRuntimeConfig:
    task_type: str
    llm: LlmModelConfig
    sft: SftDatasetConfig
    peft: PeftAdapterConfig
    federated: LlmFederatedConfig
    results_dir: Path


def parse_target_modules(value: Any) -> tuple[str, ...]:
    """Normalize LoRA target_modules from comma-separated text or a list."""
    if isinstance(value, str):
        modules = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        modules = [str(part).strip() for part in value]
    else:
        modules = []

    normalized = tuple(module for module in modules if module)
    if not normalized:
        raise ValueError("peft.target_modules must contain at least one module name")
    return normalized


def normalize_llm_peft_config(config: Mapping[str, Any]) -> LlmPeftRuntimeConfig:
    """Validate and normalize the LLM PEFT subset of a Figaro runtime config."""
    task_cfg = _mapping(config.get("task"))
    task_type = _text(task_cfg.get("type", LLM_PEFT_TASK_TYPE), "task.type")
    if task_type != LLM_PEFT_TASK_TYPE:
        raise ValueError(f"Expected task.type={LLM_PEFT_TASK_TYPE!r}, got {task_type!r}")

    llm_cfg = _mapping(config.get("llm"))
    sft_cfg = _mapping(config.get("sft"))
    peft_cfg = _mapping(config.get("peft"))
    federated_cfg = _mapping(config.get("federated"))
    logging_cfg = _mapping(config.get("logging"))

    llm = LlmModelConfig(
        base_model=_text(llm_cfg.get("base_model"), "llm.base_model"),
        tokenizer=_text(llm_cfg.get("tokenizer", "auto"), "llm.tokenizer"),
        max_seq_length=_positive_int(llm_cfg.get("max_seq_length", 1024), "llm.max_seq_length"),
        precision=_choice(llm_cfg.get("precision", "bf16"), {"fp32", "fp16", "bf16"}, "llm.precision"),
    )
    sft = SftDatasetConfig(
        dataset_path=Path(_text(sft_cfg.get("dataset_path"), "sft.dataset_path")),
        format=_choice(sft_cfg.get("format", "prompt_completion"), {"prompt_completion", "messages"}, "sft.format"),
        prompt_template=_choice(sft_cfg.get("prompt_template", "plain"), {"plain", "chatml"}, "sft.prompt_template"),
        per_device_train_batch_size=_positive_int(
            sft_cfg.get("per_device_train_batch_size", 1),
            "sft.per_device_train_batch_size",
        ),
        gradient_accumulation_steps=_positive_int(
            sft_cfg.get("gradient_accumulation_steps", 1),
            "sft.gradient_accumulation_steps",
        ),
    )
    peft = PeftAdapterConfig(
        method=_choice(peft_cfg.get("method", "lora"), {"lora"}, "peft.method"),
        rank=_positive_int(peft_cfg.get("rank", 8), "peft.rank"),
        alpha=_positive_int(peft_cfg.get("alpha", 16), "peft.alpha"),
        dropout=_bounded_float(peft_cfg.get("dropout", 0.05), "peft.dropout", minimum=0.0, maximum=1.0),
        target_modules=parse_target_modules(peft_cfg.get("target_modules", "q_proj,v_proj")),
        resume_adapter_path=_optional_path(peft_cfg.get("resume_adapter_path")),
        quantization=_choice(peft_cfg.get("quantization", "none"), {"none", "int8", "nf4_4bit"}, "peft.quantization"),
    )
    federated = LlmFederatedConfig(
        num_clients=_positive_int(federated_cfg.get("num_clients", 1), "federated.num_clients"),
        num_rounds=_positive_int(federated_cfg.get("num_rounds", 1), "federated.num_rounds"),
        clients_per_round=_positive_int(federated_cfg.get("clients_per_round", 1), "federated.clients_per_round"),
        local_epochs=_positive_int(federated_cfg.get("local_epochs", 1), "federated.local_epochs"),
        learning_rate=_positive_float(federated_cfg.get("learning_rate", 0.0002), "federated.learning_rate"),
        aggregation=_choice(
            federated_cfg.get("aggregation", "fedavg"),
            {"fedavg", "weighted_avg", "simple_avg"},
            "federated.aggregation",
        ),
        seed=_int(federated_cfg.get("seed", 42), "federated.seed"),
    )

    if federated.clients_per_round > federated.num_clients:
        raise ValueError("federated.clients_per_round cannot exceed federated.num_clients")

    return LlmPeftRuntimeConfig(
        task_type=task_type,
        llm=llm,
        sft=sft,
        peft=peft,
        federated=federated,
        results_dir=Path(str(logging_cfg.get("results_dir", "./results"))),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return Path(stripped) if stripped else None
    return Path(str(value))


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        parsed = float(value)
        if parsed.is_integer():
            return int(parsed)
    raise ValueError(f"{path} must be an integer")


def _positive_int(value: Any, path: str) -> int:
    parsed = _int(value, path)
    if parsed <= 0:
        raise ValueError(f"{path} must be positive")
    return parsed


def _positive_float(value: Any, path: str) -> float:
    parsed = _float(value, path)
    if parsed <= 0:
        raise ValueError(f"{path} must be positive")
    return parsed


def _bounded_float(value: Any, path: str, *, minimum: float, maximum: float) -> float:
    parsed = _float(value, path)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{path} must be between {minimum} and {maximum}")
    return parsed


def _float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a number") from exc


def _choice(value: Any, choices: set[str], path: str) -> str:
    parsed = _text(value, path)
    if parsed not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{path} must be one of: {allowed}")
    return parsed
