from app.services.agent.history_service import AgentOptimizationHistoryService


def test_config_version_candidates_only_include_experiments():
    snapshot = {
        "draft_experiments": [
            {"name": "draft alpha", "config_patch": {"dataset": {"alpha": 0.2}}},
        ],
        "experiments": [
            {"run_id": "run-1", "iteration": 1, "name": "alpha_0.1", "config": {"dataset": {"alpha": 0.1}}},
            {"run_id": "run-2", "iteration": 2, "name": "alpha_0.3", "config": {"dataset": {"alpha": 0.3}}},
        ],
        "best_config": {"dataset": {"alpha": 0.3}},
    }

    candidates = AgentOptimizationHistoryService._build_config_version_candidates(snapshot)

    assert [candidate["source"] for candidate in candidates] == ["experiment", "experiment"]
    assert [candidate["label"] for candidate in candidates] == ["alpha_0.1", "alpha_0.3"]
