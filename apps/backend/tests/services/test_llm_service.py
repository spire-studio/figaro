import asyncio

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import APIError, APITimeoutError

from app.core.config import settings
from app.services.llm import LLMRegistry, LLMService


class _FakeModel:
    def __init__(self, *, responses=None, timeout_fail_times=0, api_fail_times=0):
        self._responses = list(responses or ["ok"])
        self._timeout_fail_times = timeout_fail_times
        self._api_fail_times = api_fail_times
        self.invocations = 0
        self.bound_tools = None
        self.bound_tool_choice = None

    async def ainvoke(self, _messages):
        self.invocations += 1
        if self._timeout_fail_times > 0:
            self._timeout_fail_times -= 1
            raise APITimeoutError(request=httpx.Request("POST", "https://example.com"))
        if self._api_fail_times > 0:
            self._api_fail_times -= 1
            raise APIError(
                "api error",
                request=httpx.Request("POST", "https://example.com"),
                body=None,
            )

        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return AIMessage(content=item)

    def bind_tools(self, tools, tool_choice=None):
        self.bound_tools = tools
        self.bound_tool_choice = tool_choice
        return self


@pytest.fixture
def registry_backup():
    original = list(LLMRegistry.LLMS)
    try:
        yield
    finally:
        LLMRegistry.LLMS = original


def test_llm_service_uses_requested_model(registry_backup, monkeypatch):
    async def _run():
        default_model = _FakeModel(responses=["from-default"])
        requested_model = _FakeModel(responses=["from-requested"])
        LLMRegistry.LLMS = [
            {"name": "default-test", "llm": default_model},
            {"name": "requested-test", "llm": requested_model},
        ]
        monkeypatch.setattr(settings, "default_llm_model", "default-test")

        service = LLMService()
        text, _ = await service.generate_text(
            instructions="be concise",
            input_text="hello",
            model="requested-test",
        )

        assert text == "from-requested"
        assert requested_model.invocations == 1
        assert default_model.invocations == 0

    asyncio.run(_run())


def test_llm_service_retries_before_success(registry_backup, monkeypatch):
    async def _run():
        model = _FakeModel(responses=["recovered"], timeout_fail_times=2)
        LLMRegistry.LLMS = [{"name": "retry-model", "llm": model}]
        monkeypatch.setattr(settings, "default_llm_model", "retry-model")

        service = LLMService()
        text, _ = await service.generate_text(
            instructions="retry",
            input_text="please",
        )

        assert text == "recovered"
        assert model.invocations == 3

    asyncio.run(_run())


def test_llm_service_generate_with_tools_binds_tools(registry_backup, monkeypatch):
    async def _run():
        model = _FakeModel(responses=["tool-result"])
        LLMRegistry.LLMS = [{"name": "tool-model", "llm": model}]
        monkeypatch.setattr(settings, "default_llm_model", "tool-model")

        service = LLMService()
        text, meta = await service.generate_with_tools(
            instructions="use tools",
            input_text="run",
            tools=[{"name": "noop"}],
            tool_choice="required",
        )

        assert text == "tool-result"
        assert model.bound_tools == [{"name": "noop"}]
        assert model.bound_tool_choice == "required"
        assert meta["tool_calls"] == []

    asyncio.run(_run())


def test_llm_service_does_not_use_fallback_models(registry_backup, monkeypatch):
    async def _run():
        failing = _FakeModel(api_fail_times=3)
        backup = _FakeModel(responses=["backup"])
        LLMRegistry.LLMS = [
            {"name": "primary", "llm": failing},
            {"name": "backup", "llm": backup},
        ]
        monkeypatch.setattr(settings, "default_llm_model", "primary")

        service = LLMService()
        with pytest.raises(APIError):
            await service.generate_text(
                instructions="fail",
                input_text="x",
                model="primary",
                fallback_models=["backup"],
            )

        assert failing.invocations == 3
        assert backup.invocations == 0

    asyncio.run(_run())

