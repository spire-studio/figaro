from app.services.agent.objectives import (
    AgentOptimizationObjective,
    normalize_objective,
    resolve_objective,
)


def test_normalize_objective_parses_accuracy():
    assert normalize_objective("accuracy") == AgentOptimizationObjective.ACCURACY


def test_normalize_objective_defaults_to_auto():
    assert normalize_objective("unknown") == AgentOptimizationObjective.AUTO
    assert normalize_objective(None) == AgentOptimizationObjective.AUTO


def test_resolve_objective_always_returns_accuracy():
    result = resolve_objective(goal="test something", requested_objective=None)
    assert result == AgentOptimizationObjective.ACCURACY

    result = resolve_objective(
        goal="test something",
        requested_objective=AgentOptimizationObjective.AUTO,
    )
    assert result == AgentOptimizationObjective.ACCURACY
