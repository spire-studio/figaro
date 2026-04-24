"""FastAPI exception handlers that map domain errors to HTTP responses."""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core import exceptions
from app.core.logger import get_logger

logger = get_logger(__name__)


async def internal_server_error_handler(
    request: Request,
    exc: exceptions.InternalServiceError,
) -> JSONResponse:
    """Return a sanitized 500 while logging internal details."""
    logger.exception(
        "Internal service error",
        extra={
            "path": str(request.url),
            "context": getattr(exc, "context", None),
        },
    )
    message = "Internal server error"
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": message, "detail": message},
    )


async def bad_request_handler(
    _request: Request,
    exc: exceptions.AppError,
) -> JSONResponse:
    """Return 400 for domain errors and include context when present."""
    message = str(exc)
    context = getattr(exc, "context", None)
    content = {"message": message, "detail": context if context else message}
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=content,
    )


async def request_validation_error_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Normalize validation errors to a consistent 422 response body."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"message": "Request validation failed", "detail": exc.errors()},
    )


async def http_exception_handler(
    _request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Wrap HTTP exceptions to keep response shape consistent."""
    detail = exc.detail if exc.detail is not None else exc.__class__.__name__
    content = {"message": str(detail), "detail": detail}
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=exc.headers,
    )


async def resource_not_found_handler(
    _request: Request,
    exc: exceptions.ResourceNotFound,
) -> JSONResponse:
    """Return 404 when a resource cannot be located."""
    message = str(exc)
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": message, "detail": message},
    )


async def resource_conflict_handler(
    _request: Request,
    exc: exceptions.ResourceConflict,
) -> JSONResponse:
    """Return 409 when a request conflicts with resource state."""
    message = str(exc)
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"message": message, "detail": message},
    )


def register_exception_handlers(app) -> None:
    """Wire all handlers into the FastAPI app."""
    app.add_exception_handler(
        exceptions.InternalServiceError,
        internal_server_error_handler,
    )

    app.add_exception_handler(exceptions.ResourceNotFound, resource_not_found_handler)
    app.add_exception_handler(exceptions.JobNotFound, resource_not_found_handler)
    app.add_exception_handler(exceptions.RunNotFound, resource_not_found_handler)
    app.add_exception_handler(exceptions.JobConfigNotFound, resource_not_found_handler)
    app.add_exception_handler(
        exceptions.ConfigSchemaNotFound,
        resource_not_found_handler,
    )

    app.add_exception_handler(exceptions.ResourceConflict, resource_conflict_handler)
    app.add_exception_handler(exceptions.JobAlreadyExists, resource_conflict_handler)
    app.add_exception_handler(
        exceptions.ActiveRunsConflict,
        resource_conflict_handler,
    )
    app.add_exception_handler(
        exceptions.ActiveRunConflict,
        resource_conflict_handler,
    )

    app.add_exception_handler(exceptions.BadRequestError, bad_request_handler)
    app.add_exception_handler(exceptions.AppError, bad_request_handler)

    app.add_exception_handler(
        RequestValidationError,
        request_validation_error_handler,
    )
    app.add_exception_handler(HTTPException, http_exception_handler)
