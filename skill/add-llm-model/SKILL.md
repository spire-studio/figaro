---
name: add-llm-model
description: Use when adding or documenting a local LLM model for Figaro. Explains how to place Hugging Face-compatible model directories under models/llm, how Figaro discovers them as llm.base_model options, and how to verify the frontend/runtime selection.
---

# Add Local LLM Model

Use this workflow when the user wants Figaro to recognize a local LLM model in the frontend.

## Target Directory

Place local model directories under:

```text
models/llm/
```

Use one directory per model. The frontend option value will be the project-relative path:

```text
./models/llm/<model-directory>
```

## Expected Model Layout

Use a Hugging Face-compatible causal language model directory. A typical layout is:

```text
models/llm/Qwen2.5-0.5B-Instruct/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

Sharded weights are also fine if `transformers.AutoModelForCausalLM.from_pretrained()` can load the directory.

## Discovery Rules

Figaro scans `models/llm/` when the backend returns config schema.

- Direct child directories under `models/llm/` are exposed as selectable `llm.base_model` options.
- Nested directories that contain `config.json` are also exposed.
- Hidden directories are ignored.
- Model files are ignored by Git; only `models/llm/README.md` should be tracked.

Restart or refresh the backend schema after adding a model if the option does not appear.

## Frontend Selection

In Simulation or Agent config:

1. Select `task.type = llm_peft_sft`.
2. Open the LLM fields.
3. Choose the local model path from `Base model`, or type it manually.

Example value:

```text
./models/llm/Qwen2.5-0.5B-Instruct
```

Keep `llm.tokenizer = auto` unless the tokenizer lives in a different directory.

## Verification

Check that the schema exposes the model:

```bash
curl http://localhost:8000/api/v1/jobs/config/schema
```

Look for the path in:

```text
llm.base_model.options
```

Then run a small LLM PEFT simulation config and confirm the runtime log prints `LLM_BASE_MODEL` with the selected local path.

