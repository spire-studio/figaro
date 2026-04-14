from app.services.agent.state import ConfigChange, ExperimentRecord


def test_experiment_record_fields_round_trip():
    record = ExperimentRecord(
        iteration=1,
        run_id="run-1",
        job_id=12,
        iteration_goal="build baseline",
        plan_summary="lower learning rate",
        config_patch={"federated": {"learning_rate": 0.005}},
        config_diff=[ConfigChange(path="federated.learning_rate", change_type="updated", old_value=0.01, new_value=0.005)],
        config={"federated": {"num_clients": 5}},
        metrics={"global_results": {"global_accuracy": [0.7]}},
        notes="first",
    )

    assert record.iteration == 1
    assert record.run_id == "run-1"
    assert record.job_id == 12
    assert record.iteration_goal == "build baseline"
    assert record.plan_summary == "lower learning rate"
    assert record.config_patch["federated"]["learning_rate"] == 0.005
    assert record.config_diff[0].path == "federated.learning_rate"
    assert record.config["federated"]["num_clients"] == 5
    assert record.metrics["global_results"]["global_accuracy"][-1] == 0.7
    assert record.notes == "first"
