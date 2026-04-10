"""Custom exception types shared across the application."""

from __future__ import annotations

from typing import Any


class AppError(RuntimeError):
    """
    Base class for domain-specific exceptions.

    An optional ``context`` mapping can be provided to expose structured
    details for logs and observability.
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context or {})


class InternalServiceError(AppError):
    """Raised for unexpected internal/system failures."""


class BadRequestError(AppError):
    """Raised when request input violates a business rule."""


class ResourceNotFound(AppError):
    """Raised when a resource cannot be located."""


class ResourceConflict(AppError):
    """Raised when a request conflicts with current resource state."""


class JobNotFound(ResourceNotFound):
    """Raised when a job does not exist."""


class JobAlreadyExists(ResourceConflict):
    """Raised when a job with the same name already exists."""


class JobConfigNotFound(ResourceNotFound):
    """Raised when a job configuration does not exist."""


class RunNotFound(ResourceNotFound):
    """Raised when a run does not exist."""


class ConfigSchemaNotFound(ResourceNotFound):
    """Raised when the config schema file is missing."""


class ActiveRunsConflict(ResourceConflict):
    """Raised when a job cannot be deleted because active runs exist."""


class ActiveRunConflict(ResourceConflict):
    """Raised when a run cannot be deleted because it is active."""


__all__ = (
    "AppError",
    "InternalServiceError",
    "BadRequestError",
    "ResourceNotFound",
    "ResourceConflict",
    "JobNotFound",
    "JobAlreadyExists",
    "JobConfigNotFound",
    "RunNotFound",
    "ConfigSchemaNotFound",
    "ActiveRunsConflict",
    "ActiveRunConflict",
)
