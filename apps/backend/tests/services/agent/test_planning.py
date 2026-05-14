from app.core.config import settings
from app.services.agent.planning import (
    build_initial_config,
    build_schema_prompt_context,
    collect_disabled_option_errors,
    deep_merge_config,
    lock_structured_constraints,
    select_llm_model,
)


def test_select_llm_model_prefers_model_name(monkeypatch):
    monkeypatch.setattr(settings, "default_llm_model", "default-model")
    assert select_llm_model("  gpt-4.1-mini  ") == "gpt-4.1-mini"


def test_select_llm_model_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(settings, "default_llm_model", "fallback-model")
    assert select_llm_model(None) == "fallback-model"
    assert select_llm_model("   ") == "fallback-model"


def test_build_initial_config_uses_schema_defaults():
    schema = {
        "dataset": {"name": {"default": "mnist"}},
        "model": {"name": {"default": "cnn"}},
        "federated": {
            "num_clients": {"default": 20},
            "num_rounds": {"default": 50},
            "clients_per_round": {"default": 10},
            "local_epochs": {"default": 2},
            "learning_rate": {"default": 0.01},
        },
    }

    cfg = build_initial_config(schema)
    assert cfg["dataset"]["name"] == "mnist"
    assert cfg["model"]["name"] == "cnn"
    assert cfg["federated"]["num_clients"] == 20
    assert cfg["federated"]["num_rounds"] == 50
    assert cfg["federated"]["clients_per_round"] == 10
    assert cfg["federated"]["local_epochs"] == 2
    assert cfg["federated"]["learning_rate"] == 0.01


def test_build_initial_config_hardcoded_defaults_on_empty_schema():
    cfg = build_initial_config({})
    assert cfg["task"]["type"] == "classic_fl"
    assert cfg["dataset"]["name"] == "CIFAR-10"
    assert cfg["model"]["name"] == "Auto"
    assert cfg["federated"]["num_clients"] == 3
    assert cfg["federated"]["num_rounds"] == 10


def test_lock_structured_constraints_override_llm_patch_values():
    constrained_base = {
        "dataset": {"name": "FEMNIST", "alpha": 0.5},
        "model": {"name": "LeNet"},
        "federated": {"aggregation": "fedavg"},
    }
    llm_patch = {
        "dataset": {"name": "CIFAR-10", "alpha": 0.1},
        "model": {"name": "Auto"},
    }
    constraints = {"dataset": {"name": "FEMNIST"}, "model": {"name": "LeNet"}}

    merged = deep_merge_config(constrained_base, llm_patch)
    locked = lock_structured_constraints(merged, constraints)

    assert locked["dataset"]["name"] == "FEMNIST"
    assert locked["dataset"]["alpha"] == 0.1
    assert locked["model"]["name"] == "LeNet"


def test_schema_prompt_context_marks_disabled_options():
    schema = {
        "federated": {
            "aggregation": {
                "type": "select",
                "options": ["fedavg", "scaffold"],
                "default": "fedavg",
                "ui": {"options": {"scaffold": {"disabled": True}}},
            }
        }
    }

    context = build_schema_prompt_context(schema)
    field = context["fields"][0]
    assert field["path"] == "federated.aggregation"
    assert field["executable_options"] == ["fedavg"]
    assert field["disabled_options"] == ["scaffold"]

    errors = collect_disabled_option_errors({"federated": {"aggregation": "scaffold"}}, schema)
    assert errors == ["federated.aggregation='scaffold' is marked disabled in config_schema.yaml"]
