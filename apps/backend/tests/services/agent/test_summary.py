from app.services.agent.state import AgentState, ExperimentRecord
from app.services.agent.summary import (
    build_results_table,
    build_summary_text,
    get_agent_run_score,
    get_last_global_accuracy,
    get_last_llm_train_loss,
    get_last_llm_validation_loss,
)


def test_get_last_global_accuracy_returns_last_value():
    metrics = {"global_results": {"global_accuracy": [0.1, 0.2, 0.35]}}
    assert get_last_global_accuracy(metrics) == 0.35


def test_get_last_global_accuracy_returns_none_for_empty():
    assert get_last_global_accuracy({}) is None
    assert get_last_global_accuracy({"global_results": {}}) is None


def test_agent_run_score_uses_negative_llm_train_loss():
    metrics = {"llm_results": {"train_loss": [1.2, 0.8]}}
    assert get_last_llm_train_loss(metrics) == 0.8
    assert get_agent_run_score(metrics) == -0.8


def test_agent_run_score_prefers_llm_validation_loss():
    metrics = {"llm_results": {"train_loss": [0.8], "validation_loss": [0.6]}}
    assert get_last_llm_validation_loss(metrics) == 0.6
    assert get_agent_run_score(metrics) == -0.6


def test_build_results_table_formats_experiments():
    state = AgentState(goal="test experiments")
    state.experiment_results = [
        ExperimentRecord(
            iteration=1,
            run_id="run-1",
            job_id=10,
            name="alpha-0.1",
            config={
                "federated": {
                    "num_clients": 10,
                    "clients_per_round": 5,
                    "num_rounds": 20,
                    "local_epochs": 5,
                    "learning_rate": 0.01,
                },
            },
            metrics={"global_results": {"rounds": [1, 2], "global_accuracy": [0.55, 0.72]}},
            score=0.72,
        ),
        ExperimentRecord(
            iteration=2,
            run_id="run-2",
            job_id=11,
            name="alpha-0.5",
            config={
                "federated": {
                    "num_clients": 10,
                    "clients_per_round": 5,
                    "num_rounds": 20,
                    "local_epochs": 5,
                    "learning_rate": 0.01,
                },
            },
            metrics={"global_results": {"rounds": [1, 2], "global_accuracy": [0.60, 0.80]}},
            score=0.80,
        ),
    ]

    table = build_results_table(state)
    assert "alpha-0.1" in table
    assert "alpha-0.5" in table
    assert "0.7200" in table
    assert "0.8000" in table


def test_build_summary_text_reports_best_and_worst():
    state = AgentState(goal="test summary")
    state.experiment_results = [
        ExperimentRecord(
            iteration=1,
            run_id="run-1",
            job_id=10,
            name="low",
            config={"federated": {}},
            metrics={"global_results": {"rounds": [1], "global_accuracy": [0.33]}},
            score=0.33,
        ),
        ExperimentRecord(
            iteration=2,
            run_id="run-2",
            job_id=11,
            name="high",
            config={"federated": {}},
            metrics={"global_results": {"rounds": [1], "global_accuracy": [0.85]}},
            score=0.85,
        ),
    ]

    summary = build_summary_text(state=state)
    assert "Completed 2 experiments" in summary
    assert "Best accuracy: 0.8500 (high)" in summary
    assert "Worst accuracy: 0.3300 (low)" in summary


def test_build_summary_text_reports_llm_loss_when_present():
    state = AgentState(goal="test llm")
    state.experiment_results = [
        ExperimentRecord(
            iteration=1,
            run_id="run-1",
            job_id=10,
            name="rank-8",
            config={"task": {"type": "llm_peft_sft"}, "federated": {}},
            metrics={"llm_results": {"rounds": [1], "train_loss": [0.9]}},
            score=-0.9,
        ),
        ExperimentRecord(
            iteration=2,
            run_id="run-2",
            job_id=11,
            name="rank-16",
            config={"task": {"type": "llm_peft_sft"}, "federated": {}},
            metrics={"llm_results": {"rounds": [1], "train_loss": [0.7]}},
            score=-0.7,
        ),
    ]

    summary = build_summary_text(state=state)
    assert "Best LLM loss: train_loss=0.7000 (rank-16)" in summary
    assert "Worst LLM loss: train_loss=0.9000 (rank-8)" in summary
