# Local LLM Models

Place local Hugging Face-compatible model directories here.

Example:

```text
models/llm/Qwen2.5-0.5B-Instruct/config.json
models/llm/Qwen2.5-0.5B-Instruct/model.safetensors
models/llm/Qwen2.5-0.5B-Instruct/tokenizer.json
```

Directories under this path are discovered by the backend and exposed as selectable `llm.base_model` options in the frontend.

