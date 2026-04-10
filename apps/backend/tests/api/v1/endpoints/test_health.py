from app.core.config import settings


def test_fastapi_healthcheck_ok(client):
    response = client.get(f"{settings.api_prefix}/health/fastapi")
    assert response.status_code == 200
    assert response.json()["message"] == "ok"
