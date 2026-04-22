"""
Database engine and session helpers.

This module provides database engine and session helpers for the application.
"""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlmodel import SQLModel

from app.core.config import settings

# =============================================================================
# Global Naming Convention
# =============================================================================
# Configure explicitly naming conventions for SQLAlchemy/Alembic.
# This prevents "unnamed constraint" errors during migrations.
naming_convention = {
    "ix": "ix_%(column_0_label)s",  # Index
    "uq": "uq_%(table_name)s_%(column_0_name)s",  # Unique
    "ck": "ck_%(table_name)s_%(constraint_name)s",  # Check
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",  # Foreign Key
    "pk": "pk_%(table_name)s",  # Primary Key
}

# Apply the convention to the global SQLModel metadata
SQLModel.metadata.naming_convention = naming_convention


# =============================================================================
# Engine & Session Setup
# =============================================================================
# Create async engine for SQLAlchemy
engine: AsyncEngine = create_async_engine(
    str(settings.sqlalchemy_database_url),
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_pool_max_overflow,
    pool_timeout=settings.postgres_pool_timeout,
    pool_recycle=settings.postgres_pool_recycle
)

# Create async sessionmaker for SQLAlchemy
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Keep ORM instances alive after commit.
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a single AsyncSession for request-scoped database work."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """
    Initialize database utilities.
    """
    # Ensure model metadata is registered before creating tables.
    import app.models.agent  # noqa: F401
    import app.models.distributed  # noqa: F401
    import app.models.simulation  # noqa: F401

    async with engine.begin() as conn:
        # In dev mode we hard-reset postgres schema so renamed tables/types are
        # fully cleaned up (including legacy enum dependencies).
        # if conn.dialect.name == "postgresql":
        #     await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        #     await conn.execute(text("CREATE SCHEMA public"))
        # else:
        #     await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
