"""
Model shared utilities.

Provides shared helper functions used by model definitions, such as timezone-
aware UTC timestamp generation.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
