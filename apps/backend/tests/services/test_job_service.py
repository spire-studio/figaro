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


def test_normalize_simulation_config_keeps_dataset_and_federated_client_counts_aligned():
    normalized = SimulationJobService.normalize_simulation_config(
        {
            "dataset": {"num_clients": 2},
            "federated": {"num_clients": 5},
        }
    )

    assert normalized["federated"]["num_clients"] == 5
    assert normalized["dataset"]["num_clients"] == 5
