import pytest

from app.core.config import Settings, _parse_allowed_origins, _validate_allowed_origins


def _base_settings_kwargs() -> dict:
    return {
        "ENVIRONMENT": "local",
        "DEBUG": True,
        "PROJECT_NAME": "Phoenix",
        "API_PREFIX": "/api/v1",
        "FRONTEND_URL": "http://localhost:5173",
        "ALLOWED_ORIGINS": "http://localhost:5173, http://127.0.0.1:5173",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": 5432,
        "POSTGRES_DB": "phoenix",
        "POSTGRES_USER": "phoenix",
        "POSTGRES_PASSWORD": "phoenix",
        "POSTGRES_POOL_SIZE": 5,
        "POSTGRES_POOL_MAX_OVERFLOW": 10,
        "POSTGRES_POOL_TIMEOUT": 30,
        "POSTGRES_POOL_RECYCLE": 1800,
        "OPENAI_API_KEY": "test-key",
        "OPENAI_API_BASE": "https://api.openai.com/v1",
        "DEFAULT_LLM_MODEL": "gpt-4o-mini",
        "DEFAULT_LLM_TEMPERATURE": 0.2,
        "MAX_TOKENS": 2048,
    }


def test_parse_allowed_origins_from_string():
    value = "https://example.com, https://www.example.com"
    assert _parse_allowed_origins(value) == [
        "https://example.com",
        "https://www.example.com",
    ]


def test_parse_allowed_origins_from_list():
    value = [" https://a.com ", "", "https://b.com"]
    assert _parse_allowed_origins(value) == ["https://a.com", "https://b.com"]


def test_parse_allowed_origins_invalid_type():
    with pytest.raises(ValueError, match="ALLOWED_ORIGINS must be a string or list"):
        _parse_allowed_origins(123)


def test_validate_allowed_origins_valid():
    value = ["https://example.com", "http://localhost:3000"]
    assert _validate_allowed_origins(value) == value


def test_validate_allowed_origins_invalid():
    value = ["https://good.com", "bad.com", "ftp://nope.com"]
    with pytest.raises(ValueError, match="Invalid origin"):
        _validate_allowed_origins(value)


def test_validate_allowed_origins_empty():
    with pytest.raises(ValueError, match="must include at least one valid URL"):
        _validate_allowed_origins([])


def test_settings_computed_allowed_origins():
    kwargs = _base_settings_kwargs()
    kwargs["ALLOWED_ORIGINS"] = "https://one.local, https://two.local"
    settings = Settings(**kwargs)
    assert settings.allowed_origins == ["https://one.local", "https://two.local"]


def test_settings_field_naming_and_alias_mapping():
    settings = Settings(**_base_settings_kwargs())
    assert settings.environment == "local"
    assert settings.debug is True
    assert settings.project_name == "Phoenix"
    assert settings.api_prefix == "/api/v1"
    assert settings.frontend_url == "http://localhost:5173"
    assert settings.postgres_host == "localhost"
    assert settings.postgres_port == 5432
    assert settings.postgres_db == "phoenix"
    assert settings.openai_api_key == "test-key"
    assert settings.openai_api_base == "https://api.openai.com/v1"
    assert settings.default_llm_model == "gpt-4o-mini"
    assert settings.default_llm_temperature == 0.2
    assert settings.max_tokens == 2048


def test_settings_build_database_urls():
    settings = Settings(**_base_settings_kwargs())
    alembic_url = str(settings.alembic_database_url)
    sqlalchemy_url = str(settings.sqlalchemy_database_url)

    assert alembic_url.startswith("postgresql+psycopg://phoenix:phoenix@localhost:5432/")
    assert sqlalchemy_url.startswith("postgresql+asyncpg://phoenix:phoenix@localhost:5432/")
    assert alembic_url.endswith("/phoenix")
    assert sqlalchemy_url.endswith("/phoenix")
