"""
Helpers for exposing executable simulation capabilities to the Agent.

The Agent should plan only combinations that the simulation runtime can
execute. The source of truth lives in libs/fl_core/simulation_registry.py;
this module is a thin backend-side adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from app.services.simulation.compatibility import runtime_capabilities


def get_platform_capabilities(project_root: Path | None = None) -> Dict[str, Any]:
    """
    Return supported simulation datasets, models, aggregations, and compatibility
    rules. The project_root argument is kept for compatibility with older tests.
    """
    _ = project_root
    return runtime_capabilities()
