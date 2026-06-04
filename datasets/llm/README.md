# Local LLM Datasets

Place supervised fine-tuning JSONL files here.

Example:

```text
datasets/llm/train.jsonl
datasets/llm/validation.jsonl
datasets/llm/alpaca/train.jsonl
```

Each JSONL row should use either prompt/completion fields:

```json
{"prompt":"Question","completion":"Answer"}
```

or messages fields when `sft.format` is set to `messages`.

JSONL files under this path are discovered by the backend and exposed as selectable `sft.dataset_path` options in the frontend.

Validation JSONL files under this path are also exposed as selectable `evaluation.dataset_path` options for LLM PEFT evaluation.
