from app.services.agent.state import AgentState, ConfigChange, ExperimentPlan, ExperimentRecord


def test_experiment_record_fields_round_trip():
    record = ExperimentRecord(
        iteration=1,
        run_id="run-1",
        job_id=12,
        iteration_goal="build baseline",
        plan_summary="keep attack disabled",
        config_patch={"attack": {"enable": False}},
        config_diff=[ConfigChange(path="attack.enable", change_type="updated", old_value=True, new_value=False)],
        config={"federated": {"num_clients": 5}},
        metrics={"global_results": {"global_accuracy": [0.7]}},
        notes="first",
    )

    assert record.iteration == 1
    assert record.run_id == "run-1"
    assert record.job_id == 12
    assert record.iteration_goal == "build baseline"
    assert record.plan_summary == "keep attack disabled"
    assert record.config_patch["attack"]["enable"] is False
    assert record.config_diff[0].path == "attack.enable"
    assert record.config["federated"]["num_clients"] == 5
    assert record.metrics["global_results"]["global_accuracy"][-1] == 0.7
    assert record.notes == "first"


def test_agent_state_uses_isolated_mutable_defaults():
    s1 = AgentState(goal="a")
    s2 = AgentState(goal="b")

    s1.current_config["attack"] = {"enable": False}
    s1.current_plan = ExperimentPlan(iteration=1, iteration_goal="baseline")
    s1.history.append(
        ExperimentRecord(
            iteration=1,
            run_id="run-1",
            job_id=1,
            config={},
            metrics={},
        )
    )

    assert s2.current_config == {}
    assert s2.current_plan is None
    assert s2.history == []
    assert s2.max_iterations == 10
    assert s2.iteration == 0
    assert s2.terminated is False
