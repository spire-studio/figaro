from app.services.llm_resources import (
    DEFAULT_LLM_BASE_MODEL,
    DEFAULT_LLM_DATASET_PATH,
    DEFAULT_LLM_EVALUATION_DATASET_PATH,
    augment_config_schema_with_llm_resources,
    discover_llm_dataset_options,
    discover_llm_model_options,
    get_llm_resource_options,
)


def test_discovers_local_llm_model_directories(tmp_path):
    model_dir = tmp_path / "models" / "llm" / "tiny-model"
    model_dir.mkdir(parents=True)
    nested_model_dir = tmp_path / "models" / "llm" / "nested" / "snapshot"
    nested_model_dir.mkdir(parents=True)
    (nested_model_dir / "config.json").write_text("{}", encoding="utf-8")
    hidden_model_dir = tmp_path / "models" / "llm" / ".hidden-model"
    hidden_model_dir.mkdir(parents=True)

    options = discover_llm_model_options(tmp_path)

    assert "./models/llm/tiny-model" in options
    assert "./models/llm/nested/snapshot" in options
    assert "./models/llm/.hidden-model" not in options


def test_discovers_hf_snapshot_model_directory(tmp_path):
    snapshot_dir = tmp_path / "models" / "llm" / "Qwen2.5-7B-Instruct" / "snapshots" / "abc"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    options = discover_llm_model_options(tmp_path)

    assert "./models/llm/Qwen2.5-7B-Instruct" in options
    assert "./models/llm/Qwen2.5-7B-Instruct/snapshots/abc" not in options


def test_discovers_local_llm_jsonl_datasets(tmp_path):
    dataset_dir = tmp_path / "datasets" / "llm"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")
    (dataset_dir / "custom.jsonl").write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")
    nested_dir = dataset_dir / "alpaca"
    nested_dir.mkdir()
    (nested_dir / "train.jsonl").write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")
    (dataset_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    options = discover_llm_dataset_options(tmp_path)

    assert options == [
        "./datasets/llm/alpaca/train.jsonl",
        "./datasets/llm/custom.jsonl",
        "./datasets/llm/train.jsonl",
    ]


def test_discovers_parquet_dataset_directory(tmp_path):
    parquet_path = (
        tmp_path
        / "datasets"
        / "llm"
        / "alpaca"
        / "data"
        / "train-00000-of-00001.parquet"
    )
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"parquet-placeholder")

    options = discover_llm_dataset_options(tmp_path)

    assert "./datasets/llm/alpaca" in options
    assert "./datasets/llm/alpaca/data/train-00000-of-00001.parquet" in options


def test_augments_schema_with_default_and_discovered_llm_options(tmp_path):
    (tmp_path / "models" / "llm" / "tiny-model").mkdir(parents=True)
    dataset_dir = tmp_path / "datasets" / "llm"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")
    (dataset_dir / "custom.jsonl").write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")
    schema = {
        "llm": {
            "base_model": {
                "type": "text",
                "default": DEFAULT_LLM_BASE_MODEL,
                "ui": {},
            }
        },
        "sft": {
            "dataset_path": {
                "type": "text",
                "default": DEFAULT_LLM_DATASET_PATH,
                "ui": {},
            }
        },
        "evaluation": {
            "dataset_path": {
                "type": "text",
                "default": DEFAULT_LLM_EVALUATION_DATASET_PATH,
                "ui": {},
            }
        },
    }

    augmented = augment_config_schema_with_llm_resources(schema, tmp_path)

    assert augmented["llm"]["base_model"]["options"] == [
        DEFAULT_LLM_BASE_MODEL,
        "./models/llm/tiny-model",
    ]
    assert augmented["llm"]["base_model"]["ui"]["option_source"]["path"] == "./models/llm"
    assert augmented["sft"]["dataset_path"]["options"] == [
        DEFAULT_LLM_DATASET_PATH,
        "./datasets/llm/custom.jsonl",
    ]
    assert augmented["sft"]["dataset_path"]["ui"]["option_source"]["path"] == "./datasets/llm"
    assert augmented["evaluation"]["dataset_path"]["options"] == [
        DEFAULT_LLM_EVALUATION_DATASET_PATH,
        "./datasets/llm/custom.jsonl",
        "./datasets/llm/train.jsonl",
    ]
    assert augmented["evaluation"]["dataset_path"]["ui"]["option_source"]["path"] == "./datasets/llm"


def test_get_llm_resource_options_returns_defaults_and_discovered_values(tmp_path):
    (tmp_path / "models" / "llm" / "tiny-model" / "config.json").parent.mkdir(parents=True)
    (tmp_path / "models" / "llm" / "tiny-model" / "config.json").write_text("{}", encoding="utf-8")
    dataset_path = tmp_path / "datasets" / "llm" / "custom.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")

    resources = get_llm_resource_options(tmp_path)

    assert resources["default_model"] == DEFAULT_LLM_BASE_MODEL
    assert resources["models"] == [DEFAULT_LLM_BASE_MODEL, "./models/llm/tiny-model"]
    assert resources["datasets"] == [DEFAULT_LLM_DATASET_PATH, "./datasets/llm/custom.jsonl"]
