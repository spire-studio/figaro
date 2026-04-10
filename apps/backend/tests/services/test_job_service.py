import pytest

from app.core import exceptions
from app.services.simulation.job_service import SimulationJobService


def test_normalize_simulation_config_public_api_returns_invariants():
    normalized = SimulationJobService.normalize_simulation_config({})
    assert normalized["system"]["mode"] == "simulation"
    assert normalized["system"]["node_role"] == "server"


def test_normalize_simulation_config_public_api_rejects_non_object():
    with pytest.raises(exceptions.BadRequestError, match="Config must be a JSON object"):
        SimulationJobService.normalize_simulation_config("not-an-object")  # type: ignore[arg-type]


def test_normalize_simulation_config_derives_malicious_clients_from_count_selector():
    normalized = SimulationJobService.normalize_simulation_config(
        {
            "federated": {"num_clients": 5},
            "attack": {
                "enable": True,
                "attacker_selector": {"mode": "count", "count": 2},
                "attack_type": "label_flipping",
            },
        }
    )

    assert normalized["attack"]["enable"] is True
    assert normalized["attack"]["malicious_clients"] == [0, 1]
    assert normalized["defense"]["defense_params"]["num_malicious"] == 2


def test_normalize_simulation_config_requires_attacker_selection_when_attack_enabled():
    with pytest.raises(exceptions.BadRequestError, match="at least one client"):
        SimulationJobService.normalize_simulation_config(
            {
                "federated": {"num_clients": 5},
                "attack": {
                    "enable": True,
                    "attacker_selector": {"mode": "explicit_ids"},
                    "attack_type": "label_flipping",
                    "malicious_clients": [],
                },
            }
        )


def test_normalize_simulation_config_ratio_selector_is_deterministic():
    normalized = SimulationJobService.normalize_simulation_config(
        {
            "federated": {"num_clients": 5},
            "attack": {
                "enable": True,
                "attacker_selector": {"mode": "ratio", "ratio": 0.4},
                "attack_type": "label_flipping",
            },
        }
    )

    assert normalized["attack"]["malicious_clients"] == [0, 1]


def test_normalize_simulation_config_disables_attack_cleanly_when_enable_is_false():
    normalized = SimulationJobService.normalize_simulation_config(
        {
            "federated": {"num_clients": 5},
            "attack": {
                "enable": False,
                "attacker_selector": {"mode": "count", "count": 2},
                "malicious_clients": [0, 1],
            },
        }
    )

    assert normalized["attack"]["enable"] is False
    assert normalized["attack"]["malicious_clients"] == []
    assert normalized["defense"]["defense_params"]["num_malicious"] == 0


def test_normalize_simulation_config_keeps_dataset_and_federated_client_counts_aligned():
    normalized = SimulationJobService.normalize_simulation_config(
        {
            "dataset": {"num_clients": 2},
            "federated": {"num_clients": 5},
        }
    )

    assert normalized["federated"]["num_clients"] == 5
    assert normalized["dataset"]["num_clients"] == 5
