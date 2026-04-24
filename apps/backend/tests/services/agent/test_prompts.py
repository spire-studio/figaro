from app.services.agent.prompts import (
    build_plan_prompt,
    build_plan_system_instructions,
    get_plan_instructions,
)
from app.services.agent.state import AgentState


def test_get_plan_instructions_loaded_from_file():
    text = get_plan_instructions()
    assert "federated learning experiment planner" in text
    assert "plan_summary" in text
    assert "experiments" in text


def test_build_plan_system_instructions_includes_global_prompt():
    instructions = build_plan_system_instructions("test alpha=0.1,0.3")
    assert "User experiment request" in instructions
    assert "test alpha=0.1,0.3" in instructions


def test_build_plan_prompt_includes_history_and_context():
    state = AgentState(
        goal="test alpha=0.1,0.3",
        system_mode="simulation",
        model_name="gpt-4.1-mini",
    )

    prompt = build_plan_prompt(
        state=state,
        capabilities={"datasets": ["mnist"]},
        base_config={"federated": {"num_clients": 3}},
    )

    assert "Experiment request" in prompt
    assert "test alpha=0.1,0.3" in prompt
    assert "Capabilities" in prompt
    assert "Default base config" in prompt
    assert "Return ONLY a JSON object" in prompt
