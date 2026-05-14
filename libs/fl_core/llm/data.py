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


def load_jsonl_sft_records(path: str | Path, *, data_format: str) -> list[SftRecord]:
    """Load prompt/completion or chat messages SFT records from JSONL."""
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

    if not records:
        raise ValueError(f"SFT dataset is empty: {resolved}")
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
    if data_format == "prompt_completion":
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

    raise ValueError(f"Unsupported SFT format: {data_format}")

