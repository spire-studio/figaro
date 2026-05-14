from __future__ import annotations

from pathlib import Path

from core_runtime import FederatedLearningFramework


def run_classic_fl_runtime(config_path: Path) -> bool:
    """Run the existing classic small-model federated learning runtime."""
    framework = FederatedLearningFramework(config_path=str(config_path))
    return framework.run()

