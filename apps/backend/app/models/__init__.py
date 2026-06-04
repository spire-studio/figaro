"""Model package marker — import all models so SQLModel registers them."""

from app.models.agent import (  # noqa: F401
    AgentConfigVersion,
    AgentExperiment,
    AgentExperimentRun,
    AgentExperimentRunLog,
    AgentExperimentRunResult,
    AgentOptimizationJob,
)
