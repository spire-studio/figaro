"""Model package marker — import all models so SQLModel registers them."""

from app.models.agent import (  # noqa: F401
    AgentExperiment,
    AgentExperimentRun,
    AgentExperimentRunLog,
    AgentExperimentRunResult,
)
