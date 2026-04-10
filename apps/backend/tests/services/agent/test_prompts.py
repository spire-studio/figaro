from app.services.agent.prompts import (
    build_plan_prompt,
    build_plan_system_instructions,
    get_plan_instructions,
)
from app.services.agent.state import AgentState, ConfigChange, ExperimentRecord


def test_get_plan_instructions_loaded_from_file():
    text = get_plan_instructions()
    assert "strict JSON object" in text
    assert "must override conflicting defaults" in text
    assert "use 5 clients" in text
    assert "`config_patch`" in text
    assert "Do not enable an attack without selecting at least one malicious client." in text


def test_build_plan_system_instructions_includes_global_prompt():
    instructions = build_plan_system_instructions("Always use only 5 clients.")
    assert "strong system constraint" in instructions
    assert "Always use only 5 clients." in instructions


def test_build_plan_prompt_includes_history_and_context():
    state = AgentState(
        goal="maximize global accuracy and use only 5 clients",
        system_mode="simulation",
        model_name="gpt-4.1-mini",
        history=[
            ExperimentRecord(
                iteration=1,
                run_id="run-1",
                job_id=1,
                iteration_goal="baseline",
                plan_summary="disable attack to establish a control run",
                config_patch={"attack": {"enable": False}},
                config_diff=[
                    ConfigChange(
                        path="attack.enable",
                        change_type="updated",
                        old_value=True,
                        new_value=False,
                    )
                ],
                score=0.7,
                result_summary="The baseline run reached 0.7 accuracy.",
                decision="new_best",
                lessons_learned=["Disabling attack improved accuracy."],
                config={"attack": {"enable": False}},
                metrics={"global_results": {"global_accuracy": [0.6, 0.7]}},
            )
        ],
        best_metrics={"global_results": {"global_accuracy": [0.7]}},
    )

    prompt = build_plan_prompt(
        state=state,
        capabilities={"datasets": ["mnist"]},
        base_config={"attack": {"enable": False}},
    )

    assert "Optimization job global prompt:" in prompt
    assert "use only 5 clients" in prompt
    assert "focused goal for this iteration" in prompt
    assert "must also define how attackers are selected" in prompt
    assert "Base config:" in prompt
    assert "Model name: gpt-4.1-mini" in prompt
    assert "Iteration 1:" in prompt
    assert "Lessons: Disabling attack improved accuracy." in prompt
    assert "Return ONLY JSON, no commentary." in prompt
