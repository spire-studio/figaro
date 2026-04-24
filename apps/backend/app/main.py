"""
FastAPI application entrypoint.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logger import setup_logging
from app.core.db import init_db
from app.api.v1.api import api_router


# FastAPI lifespan function
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage startup/shutdown tasks for the FastAPI app."""
    setup_logging()
    await init_db()
    yield


app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    lifespan=lifespan,
)
register_exception_handlers(app)

is_local_env = settings.environment.lower() in {"local", "dev", "development", "test"}

# Register CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins if not is_local_env else [],
    allow_origin_regex=r".*" if is_local_env else None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.api_prefix)
