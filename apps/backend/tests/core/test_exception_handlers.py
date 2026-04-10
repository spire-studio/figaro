import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from app.core import exceptions
import app.core.exception_handlers as handlers


def _build_request(path: str = "/test") -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _response_json(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


def test_internal_server_error_handler_returns_sanitized_payload(monkeypatch):
    captured = {}

    def _fake_exception(message, extra=None):
        captured["message"] = message
        captured["extra"] = extra

    monkeypatch.setattr(handlers.logger, "exception", _fake_exception)
    request = _build_request("/internal")
    exc = exceptions.InternalServiceError("db crashed", context={"job_id": 9})
    response = asyncio.run(handlers.internal_server_error_handler(request, exc))

    assert response.status_code == 500
    assert _response_json(response) == {
        "message": "Internal server error",
        "detail": "Internal server error",
    }
    assert captured["message"] == "Internal service error"
    assert captured["extra"]["context"] == {"job_id": 9}
    assert captured["extra"]["path"].endswith("/internal")


def test_bad_request_handler_includes_context_when_present():
    request = _build_request()
    exc = exceptions.BadRequestError("invalid payload", context={"field": "name"})
    response = asyncio.run(handlers.bad_request_handler(request, exc))

    assert response.status_code == 400
    assert _response_json(response) == {
        "message": "invalid payload",
        "detail": {"field": "name"},
    }


def test_bad_request_handler_uses_message_when_context_missing():
    request = _build_request()
    exc = exceptions.BadRequestError("invalid payload")
    response = asyncio.run(handlers.bad_request_handler(request, exc))

    assert response.status_code == 400
    assert _response_json(response) == {
        "message": "invalid payload",
        "detail": "invalid payload",
    }


def test_request_validation_error_handler_normalizes_payload():
    request = _build_request()
    exc = RequestValidationError(
        [{"loc": ("body", "name"), "msg": "Field required", "type": "missing"}]
    )
    response = asyncio.run(handlers.request_validation_error_handler(request, exc))

    assert response.status_code == 422
    payload = _response_json(response)
    assert payload["message"] == "Request validation failed"
    assert payload["detail"][0]["loc"] == ["body", "name"]


def test_http_exception_handler_wraps_message_and_headers():
    request = _build_request()
    exc = HTTPException(
        status_code=401,
        detail="unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )
    response = asyncio.run(handlers.http_exception_handler(request, exc))

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert _response_json(response) == {
        "message": "unauthorized",
        "detail": "unauthorized",
    }


def test_resource_error_handlers_map_expected_status_codes():
    request = _build_request()

    not_found = asyncio.run(
        handlers.resource_not_found_handler(request, exceptions.ResourceNotFound("missing"))
    )
    conflict = asyncio.run(
        handlers.resource_conflict_handler(request, exceptions.ResourceConflict("conflict"))
    )

    assert not_found.status_code == 404
    assert _response_json(not_found) == {"message": "missing", "detail": "missing"}
    assert conflict.status_code == 409
    assert _response_json(conflict) == {"message": "conflict", "detail": "conflict"}


def test_register_exception_handlers_registers_all_domain_handlers():
    app = FastAPI()
    handlers.register_exception_handlers(app)

    assert exceptions.InternalServiceError in app.exception_handlers
    assert exceptions.ResourceNotFound in app.exception_handlers
    assert exceptions.JobNotFound in app.exception_handlers
    assert exceptions.RunNotFound in app.exception_handlers
    assert exceptions.JobConfigNotFound in app.exception_handlers
    assert exceptions.ConfigSchemaNotFound in app.exception_handlers
    assert exceptions.ResourceConflict in app.exception_handlers
    assert exceptions.JobAlreadyExists in app.exception_handlers
    assert exceptions.ActiveRunsConflict in app.exception_handlers
    assert exceptions.ActiveRunConflict in app.exception_handlers
    assert exceptions.BadRequestError in app.exception_handlers
    assert exceptions.AppError in app.exception_handlers
    assert RequestValidationError in app.exception_handlers
    assert HTTPException in app.exception_handlers
