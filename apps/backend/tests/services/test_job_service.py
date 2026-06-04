import pytest

from app.core import exceptions
from app.services.simulation.job_service import SimulationJobService


def test_normalize_simulation_config_public_api_returns_invariants():
    normalized = SimulationJobService.normalize_simulation_config({})
    assert normalized["task"]["type"] == "classic_fl"
    assert normalized["system"]["mode"] == "simulation"
    assert normalized["system"]["node_role"] == "server"


def test_normalize_simulation_config_public_api_rejects_non_object():
    with pytest.raises(exceptions.BadRequestError, match="Config must be a JSON object"):
        SimulationJobService.normalize_simulation_config("not-an-object")  # type: ignore[arg-type]


def test_normalize_simulation_config_keeps_dataset_and_federated_client_counts_aligned():
    normalized = SimulationJobService.normalize_simulation_config(
        {
            "dataset": {"num_clients": 2},
            "federated": {"num_clients": 5},
        }
    )

    assert normalized["federated"]["num_clients"] == 5
    assert normalized["dataset"]["num_clients"] == 5


def test_normalize_simulation_config_allows_auto_model_for_dataset_switch():
    normalized = SimulationJobService.normalize_simulation_config(
        {
            "dataset": {"name": "MNIST"},
            "model": {"name": "Auto"},
        }
    )

    assert normalized["dataset"]["name"] == "MNIST"
    assert normalized["model"]["name"] == "Auto"
    assert normalized["model"]["input_shape"] == [1, 28, 28]
    assert normalized["model"]["num_classes"] == 10


def test_normalize_simulation_config_rejects_incompatible_model():
    with pytest.raises(exceptions.BadRequestError, match="not compatible"):
        SimulationJobService.normalize_simulation_config(
            {
                "dataset": {"name": "AG News"},
                "model": {"name": "ResNet18"},
            }
        )


def test_normalize_simulation_config_bypasses_classic_compatibility_for_llm_peft_route():
    normalized = SimulationJobService.normalize_simulation_config(
        {
            "task": {"type": "llm_peft_sft"},
            "dataset": {"name": "AG News"},
            "model": {"name": "ResNet18"},
        }
    )

    assert normalized["task"]["type"] == "llm_peft_sft"
    assert normalized["dataset"]["name"] == "AG News"
    assert normalized["model"]["name"] == "ResNet18"


def test_normalize_simulation_config_rejects_ckks_with_sparsification():
    with pytest.raises(exceptions.BadRequestError, match="not compatible"):
        SimulationJobService.normalize_simulation_config(
            {
                "privacy": {"homomorphic_encryption": {"enable": True}},
                "compression": {"sparsification": {"enable": True}},
            }
        )


def test_normalize_simulation_config_accepts_dp_with_compression():
    normalized = SimulationJobService.normalize_simulation_config(
        {
            "privacy": {
                "differential_privacy": {
                    "enable": True,
                    "clipping_norm": 1.0,
                    "noise_multiplier": 0.1,
                }
            },
            "compression": {
                "sparsification": {
                    "enable": True,
                    "method": "random_k",
                    "ratio": 0.25,
                }
            },
        }
    )

    assert normalized["privacy"]["differential_privacy"]["enable"] is True
    assert normalized["compression"]["sparsification"]["method"] == "random_k"


def test_normalize_simulation_config_rejects_secure_aggregation_with_weighted_avg():
    with pytest.raises(exceptions.BadRequestError, match="equal-weight"):
        SimulationJobService.normalize_simulation_config(
            {
                "federated": {"aggregation": "weighted_avg"},
                "privacy": {"secure_aggregation": {"enable": True}},
            }
        )


def test_normalize_simulation_config_rejects_secure_aggregation_with_compression():
    with pytest.raises(exceptions.BadRequestError, match="not compatible"):
        SimulationJobService.normalize_simulation_config(
            {
                "privacy": {"secure_aggregation": {"enable": True}},
                "compression": {"sparsification": {"enable": True}},
            }
        )
