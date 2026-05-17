---
name: add-llm-dataset
description: Use when adding or documenting a local supervised fine-tuning dataset for Figaro LLM PEFT runs. Explains how to place JSONL files under datasets/llm, supported prompt_completion and messages formats, frontend sft.dataset_path discovery, and verification.
---

# Add Local LLM Dataset

Use this workflow when the user wants Figaro to recognize a local SFT dataset in the frontend.

## Target Directory

Place LLM supervised fine-tuning datasets under:

```text
datasets/llm/
```

The backend discovers `.jsonl` files recursively. The frontend option value will be the project-relative file path:

```text
./datasets/llm/<dataset>.jsonl
```

Nested paths are supported:

```text
./datasets/llm/alpaca/train.jsonl
```

Use a separate validation file when enabling LLM evaluation:

```text
./datasets/llm/validation.jsonl
```

## Supported Formats

Use JSON Lines: one JSON object per line.

For `sft.format = prompt_completion`, each row should contain:

```json
{"prompt":"Question","completion":"Answer"}
```

For `sft.format = messages`, each row should contain a chat-style message list:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Keep every line valid JSON. Do not wrap the file in a top-level array.

## Discovery Rules

Figaro scans `datasets/llm/` when the backend returns config schema.

- Files ending in `.jsonl` are exposed as selectable `sft.dataset_path` options.
- The same JSONL files are exposed as selectable `evaluation.dataset_path` options.
- Non-JSONL files are ignored.
- Hidden paths are ignored.
- Dataset files are ignored by Git; only `datasets/llm/README.md` should be tracked.

Restart or refresh the backend schema after adding a dataset if the option does not appear.

## Frontend Selection

In Simulation or Agent config:

1. Select `task.type = llm_peft_sft`.
2. Open the SFT Dataset fields.
3. Choose the JSONL file from `Dataset path`, or type it manually.
4. Set `Dataset format` to match the JSONL rows.
5. To compute validation loss/perplexity, enable `evaluation.enable` and select a validation JSONL file.

Example values:

```text
sft.dataset_path = ./datasets/llm/train.jsonl
sft.format = prompt_completion
evaluation.dataset_path = ./datasets/llm/validation.jsonl
```

## Verification

Check that the schema exposes the dataset:

```bash
curl http://localhost:8000/api/v1/jobs/config/schema
```

Look for the path in:

```text
sft.dataset_path.options
```

Then run a small LLM PEFT simulation config and confirm the runtime metrics include `llm_dataset.num_records` and `llm_dataset.client_record_counts`.
