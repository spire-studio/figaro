"""
Dependency helpers.
"""

from __future__ import annotations
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session


AsyncSessionDep = Annotated[AsyncSession, Depends(get_session)]

__all__ = ["AsyncSessionDep"]
