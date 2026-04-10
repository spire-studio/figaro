from app.core.config import settings
from app.main import app


def test_app_metadata_uses_settings():
    assert app.title == settings.project_name


def test_health_route_is_registered():
    route_paths = {route.path for route in app.routes}
    assert f"{settings.api_prefix}/health/fastapi" in route_paths
