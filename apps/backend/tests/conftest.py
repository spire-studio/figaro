from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep local test runs independent from shell-level proxy variables.
for _proxy_key in (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
):
    os.environ.pop(_proxy_key, None)


# Provide stable defaults for tests that import global settings at module import time.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("PROJECT_NAME", "Phoenix Test")
os.environ.setdefault("API_PREFIX", "/api/v1")
os.environ.setdefault("FRONTEND_URL", "http://localhost:5173")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "phoenix_test")
os.environ.setdefault("POSTGRES_USER", "phoenix")
os.environ.setdefault("POSTGRES_PASSWORD", "phoenix")
os.environ.setdefault("POSTGRES_POOL_SIZE", "5")
os.environ.setdefault("POSTGRES_POOL_MAX_OVERFLOW", "10")
os.environ.setdefault("POSTGRES_POOL_TIMEOUT", "30")
os.environ.setdefault("POSTGRES_POOL_RECYCLE", "1800")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")
os.environ.setdefault("DEFAULT_LLM_MODEL", "gpt-4o-mini")
os.environ.setdefault("DEFAULT_LLM_TEMPERATURE", "0.2")
os.environ.setdefault("MAX_TOKENS", "2048")

import app.main as main_module
from app.core.db import get_session
from app.services.distributed.runtime_service import DistributedRuntimeService
from app.services.simulation.run_service import SimulationRunService


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test_backend.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    def _simulation_config_dir(self):
        path = tmp_path / "configs" / "simulation_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_results_dir(self):
        path = tmp_path / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _simulation_log_dir(self):
        path = tmp_path / "logs" / "simulation_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _distributed_config_dir(self):
        path = tmp_path / "configs" / "distributed_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _distributed_log_dir(self):
        path = tmp_path / "logs" / "distributed_runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _distributed_results_dir(self):
        path = tmp_path / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _create_tables():
        import app.models.agent  # noqa: F401
        import app.models.distributed  # noqa: F401
        import app.models.simulation  # noqa: F401

        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def _drop_tables():
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
        await engine.dispose()

    async def override_get_session():
        async with session_maker() as session:
            yield session

    async def _noop_init_db():
        return None

    # Avoid touching real DB on TestClient startup.
    main_module.init_db = _noop_init_db
    main_module.app.dependency_overrides[get_session] = override_get_session
    monkeypatch.setattr(SimulationRunService, "_job_config_dir", _simulation_config_dir)
    monkeypatch.setattr(SimulationRunService, "_results_dir", _run_results_dir)
    monkeypatch.setattr(SimulationRunService, "_runtime_log_dir", _simulation_log_dir)
    monkeypatch.setattr(DistributedRuntimeService, "_runtime_config_dir", _distributed_config_dir)
    monkeypatch.setattr(DistributedRuntimeService, "_runtime_log_dir", _distributed_log_dir)
    monkeypatch.setattr(DistributedRuntimeService, "_results_dir", _distributed_results_dir)
    asyncio.run(_create_tables())
    try:
        with TestClient(main_module.app) as test_client:
            yield test_client
    finally:
        main_module.app.dependency_overrides.clear()
        asyncio.run(_drop_tables())
