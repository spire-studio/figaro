"""
Health check endpoint.
"""

from fastapi import APIRouter, status

from app.schemas.message import Message

health_router = APIRouter()


@health_router.get(
    "/fastapi",
    status_code=status.HTTP_200_OK,
    response_model=Message,
    summary="FastAPI Health Check",
)
async def fastapi_healthcheck() -> Message:
    """Health check endpoint for FastAPI."""
    return Message(message="ok")


__all__ = ["health_router"]
