"""
Application configuration settings.

This module provides centralized configuration settings for the application,
"""

from functools import lru_cache
from typing import Any, List
from urllib.parse import urlparse

from pydantic import (
    Field,
    PostgresDsn,
    computed_field
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized config backed by environment variables."""

    model_config = SettingsConfigDict(
        # .env file locations (first found wins):
        # 1. Current directory (Docker container)
        # 2. Parent directory (local development)
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Environment variables take precedence over .env file
        extra="ignore",
    )

    # Application configuration
    environment: str = Field(alias="ENVIRONMENT")
    debug: bool = Field(alias="DEBUG")
    project_name: str = Field(alias="PROJECT_NAME")
    api_prefix: str = Field(alias="API_PREFIX")
    frontend_url: str = Field(alias="FRONTEND_URL")
    allowed_origins_raw: str = Field(alias="ALLOWED_ORIGINS")

    @computed_field
    @property
    def allowed_origins(self) -> List[str]:
        """Parse and validate ALLOWED_ORIGINS from string to list of URLs."""
        return _validate_allowed_origins(_parse_allowed_origins(self.allowed_origins_raw))

    # PostgreSQL configuration
    postgres_host: str = Field(alias="POSTGRES_HOST")
    postgres_port: int = Field(alias="POSTGRES_PORT")
    postgres_db: str = Field(alias="POSTGRES_DB")
    postgres_user: str = Field(alias="POSTGRES_USER")
    postgres_password: str = Field(alias="POSTGRES_PASSWORD")
    postgres_pool_size: int = Field(alias="POSTGRES_POOL_SIZE")
    postgres_pool_max_overflow: int = Field(alias="POSTGRES_POOL_MAX_OVERFLOW")
    postgres_pool_timeout: int = Field(alias="POSTGRES_POOL_TIMEOUT")
    postgres_pool_recycle: int = Field(alias="POSTGRES_POOL_RECYCLE")

    # LLM API configuration
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_api_base: str = Field(alias="OPENAI_API_BASE")
    default_llm_model: str = Field(alias="DEFAULT_LLM_MODEL")
    default_llm_temperature: float = Field(alias="DEFAULT_LLM_TEMPERATURE")
    max_tokens: int = Field(alias="MAX_TOKENS")

    @computed_field
    @property
    def alembic_database_url(self) -> PostgresDsn:
        """Build a PostgreSQL DSN string for Alembic (sync)."""
        return self._build_postgres_dsn("postgresql+psycopg")

    @computed_field
    @property
    def sqlalchemy_database_url(self) -> PostgresDsn:
        """Build a PostgreSQL DSN string for SQLAlchemy (async)."""
        return self._build_postgres_dsn("postgresql+asyncpg")

    # =================================================
    # Helper methods
    # =================================================
    def _build_postgres_dsn(self, scheme: str) -> PostgresDsn:
        """Build a PostgreSQL DSN string from the configuration."""
        return PostgresDsn.build(
            scheme=scheme,
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            path=self.postgres_db,
        )


# =================================================
# Tool functions
# =================================================

def _parse_allowed_origins(value: str | List[str] | Any) -> List[str]:
    """
    Parses the ALLOWED_ORIGINS configuration from comma-separated string.

    Format: "https://example.com,https://www.example.com"
    Returns a list of strings (not yet validated as URLs).
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if not isinstance(value, str):
        raise ValueError(f"ALLOWED_ORIGINS must be a string or list, got {type(value)}")

    # Remove leading/trailing whitespace and surrounding quotes
    cleaned = value.strip().strip('"').strip("'")

    # Split by comma and clean each item
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def _validate_allowed_origins(value: List[str]) -> List[str]:
    """
    Validates ALLOWED_ORIGINS list of strings as URLs.
    This is called after BeforeValidator has normalized the input to a list.
    """
    raw_origins = value

    if not raw_origins:
        raise ValueError("ALLOWED_ORIGINS must include at least one valid URL.")

    validated_origins = [
        origin
        for origin in raw_origins
        if urlparse(origin).scheme in {"http", "https"} and urlparse(origin).netloc
    ]

    invalid_origins = set[str](raw_origins) - set[str | Any](validated_origins)
    if invalid_origins:
        raise ValueError(
            f"Invalid origin(s): {', '.join(invalid_origins)}. "
            "Origins must be full http(s) URLs."
        )

    return validated_origins


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the Settings class."""
    return Settings()


# Global settings instance
settings = get_settings()
