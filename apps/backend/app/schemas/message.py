"""
Shared message schema.
"""

from pydantic import BaseModel


class Message(BaseModel):
    """Schema for generic message responses."""

    message: str
    detail: str | None = None
