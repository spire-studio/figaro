from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SftRecord:
    """One supervised fine-tuning record before tokenization."""

    prompt: str | None = None
    completion: str | None = None
    messages: tuple[dict[str, str], ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_sft_records(
    path: str | Path,
    *,
    data_format: str,
    file_format: str = "auto",
) -> list[SftRecord]:
    """Load SFT records from a JSONL/Parquet file or a dataset directory."""
    resolved = Path(path)
    parsed_file_format = _normalize_file_format(file_format)
    files = _resolve_sft_files(resolved, file_format=parsed_file_format)

    records: list[SftRecord] = []
    for file_path in files:
        detected_format = _detect_file_format(file_path)
        if detected_format == "jsonl":
            records.extend(_load_jsonl_sft_records_file(file_path, data_format=data_format))
        elif detected_format == "parquet":
            records.extend(_load_parquet_sft_records_file(file_path, data_format=data_format))

    if not records:
        raise ValueError(f"SFT dataset is empty: {resolved}")
    return records


def load_jsonl_sft_records(path: str | Path, *, data_format: str) -> list[SftRecord]:
    """Load prompt/completion or chat messages SFT records from JSONL."""
    return load_sft_records(path, data_format=data_format, file_format="jsonl")


def _load_jsonl_sft_records_file(path: Path, *, data_format: str) -> list[SftRecord]:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"SFT dataset not found: {resolved}")

    records: list[SftRecord] = []
    for line_number, raw_line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"SFT JSONL line {line_number} must be an object")
        records.append(_record_from_payload(payload, data_format=data_format, line_number=line_number))

    return records


def _load_parquet_sft_records_file(path: Path, *, data_format: str) -> list[SftRecord]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet SFT datasets require pyarrow. Install the project dependencies before loading parquet data."
        ) from exc

    table = pq.read_table(path)
    records: list[SftRecord] = []
    for row_number, payload in enumerate(table.to_pylist(), start=1):
        if not isinstance(payload, Mapping):
            raise ValueError(f"Parquet row {row_number} must be an object")
        records.append(_record_from_payload(payload, data_format=data_format, line_number=row_number))
    return records


def split_records_by_client(
    records: list[SftRecord],
    *,
    num_clients: int,
    seed: int,
    shuffle: bool = True,
) -> list[list[SftRecord]]:
    """Deterministically partition SFT records across clients."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not records:
        raise ValueError("records must not be empty")

    ordered = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    clients = [[] for _ in range(num_clients)]
    for idx, record in enumerate(ordered):
        clients[idx % num_clients].append(record)
    return clients


def format_sft_record_text(record: SftRecord, *, data_format: str, prompt_template: str) -> str:
    """Render a record into text for rough token accounting or simple tokenizers."""
    if data_format in {"prompt_completion", "alpaca"}:
        return f"{record.prompt or ''}{record.completion or ''}"
    if data_format != "messages":
        raise ValueError(f"Unsupported SFT format: {data_format}")

    if prompt_template == "chatml":
        chunks = []
        for message in record.messages:
            chunks.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
        return "\n".join(chunks)

    chunks = [f"{message['role']}: {message['content']}" for message in record.messages]
    return "\n".join(chunks)


def _record_from_payload(payload: Mapping[str, Any], *, data_format: str, line_number: int) -> SftRecord:
    if data_format == "prompt_completion":
        prompt = payload.get("prompt")
        completion = payload.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise ValueError(f"Line {line_number} must contain string prompt and completion fields")
        metadata = {key: value for key, value in payload.items() if key not in {"prompt", "completion"}}
        return SftRecord(prompt=prompt, completion=completion, metadata=metadata)

    if data_format == "messages":
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Line {line_number} must contain a non-empty messages list")
        normalized_messages: list[dict[str, str]] = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"Line {line_number} message {message_index} must be an object")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(f"Line {line_number} message {message_index} must contain role and content")
            normalized_messages.append({"role": role, "content": content})
        metadata = {key: value for key, value in payload.items() if key != "messages"}
        return SftRecord(messages=tuple(normalized_messages), metadata=metadata)

    if data_format == "alpaca":
        instruction = payload.get("instruction")
        input_text = payload.get("input", "")
        output = payload.get("output")
        if not isinstance(instruction, str) or not isinstance(output, str):
            raise ValueError(f"Line {line_number} must contain string instruction and output fields")
        if input_text is None:
            input_text = ""
        if not isinstance(input_text, str):
            raise ValueError(f"Line {line_number} input field must be a string when present")
        if input_text.strip():
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer:"
        else:
            prompt = f"Instruction: {instruction}\nAnswer:"
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"instruction", "input", "output"}
        }
        return SftRecord(prompt=prompt, completion=f" {output}", metadata=metadata)

    raise ValueError(f"Unsupported SFT format: {data_format}")


def _resolve_sft_files(path: Path, *, file_format: str) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {path}")

    if path.is_file():
        detected_format = _detect_file_format(path)
        if detected_format is None:
            raise ValueError(f"Unsupported SFT dataset file extension: {path.suffix}")
        if file_format != "auto" and detected_format != file_format:
            raise ValueError(f"SFT dataset file is {detected_format}, not {file_format}: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"SFT dataset not found: {path}")

    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and (detected_format := _detect_file_format(candidate)) is not None
        and (file_format == "auto" or detected_format == file_format)
    ]
    if not files:
        expected = "JSONL or Parquet" if file_format == "auto" else file_format.upper()
        raise FileNotFoundError(f"No {expected} SFT files found in dataset directory: {path}")
    return sorted(files, key=lambda item: item.as_posix().lower())


def _detect_file_format(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    return None


def _normalize_file_format(file_format: str) -> str:
    normalized = str(file_format).strip().lower()
    if normalized not in {"auto", "jsonl", "parquet"}:
        raise ValueError("file_format must be one of: auto, jsonl, parquet")
    return normalized
