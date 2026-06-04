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
    assert "LLM PEFT simulation" in instructions
    assert "validation or perplexity" in instructions


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
        schema_context={
            "fields": [
                {
                    "path": "federated.aggregation",
                    "executable_options": ["fedavg"],
                    "disabled_options": ["scaffold"],
                }
            ]
        },
        config_constraints={"model": {"name": "ResNet"}},
    )

    assert "Experiment request" in prompt
    assert "test alpha=0.1,0.3" in prompt
    assert "Runtime capabilities" in prompt
    assert "Default base config" in prompt
    assert "Schema context from config_schema.yaml" in prompt
    assert "User-selected structured constraints" in prompt
    assert "disabled_options" in prompt
    assert "evaluation.enable=true" in prompt
    assert "ResNet" in prompt
    assert "Return ONLY a JSON object" in prompt
