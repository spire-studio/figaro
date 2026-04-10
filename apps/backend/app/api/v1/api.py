"""
Primary API router for version 1.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.agent import agent_router
from app.api.v1.endpoints.distributed import distributed_router
from app.api.v1.endpoints.health import health_router
from app.api.v1.endpoints.jobs import jobs_router
from app.api.v1.endpoints.runs import runs_router


api_router = APIRouter()

api_router.include_router(
    health_router,
    prefix="/health",
    tags=["Health"],
)

api_router.include_router(
    jobs_router,
    prefix="/jobs",
    tags=["Jobs"],
)

api_router.include_router(
    distributed_router,
    prefix="/distributed",
    tags=["Distributed"],
)

api_router.include_router(
    runs_router,
    prefix="/runs",
    tags=["Runs"],
)

api_router.include_router(
    agent_router,
    prefix="/agent",
    tags=["Agent"],
)
