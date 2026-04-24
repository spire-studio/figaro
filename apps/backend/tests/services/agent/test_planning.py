from app.core.config import settings
from app.services.agent.planning import build_initial_config, select_llm_model


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
    assert cfg["dataset"]["name"] == "cifar10"
    assert cfg["model"]["name"] == "cnn"
    assert cfg["federated"]["num_clients"] == 10
    assert cfg["federated"]["num_rounds"] == 20
