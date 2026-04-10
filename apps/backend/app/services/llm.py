"""LLM service for managing LLM calls with retries."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIError, APITimeoutError, OpenAIError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logger import logger


def _build_openai_model(model_name: str, **kwargs: Any) -> BaseChatModel:
    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        max_tokens=settings.max_tokens,
        **kwargs,
    )


class LLMRegistry:
    """Registry of available LLM models with pre-initialized instances."""

    LLMS: List[Dict[str, Any]] = [
        {
            "name": "gpt-5.4",
            "llm": _build_openai_model("gpt-5.4"),
        },
        {
            "name": "gpt-5.3-codex-medium",
            "llm": _build_openai_model("gpt-5.3-codex-medium"),
        },
        {
            "name": "gpt-5.2",
            "llm": _build_openai_model("gpt-5.2"),
        },
        {
            "name": "gpt-5.1",
            "llm": _build_openai_model("gpt-5.1"),
        },
        {
            "name": "gpt-4o",
            "llm": _build_openai_model("gpt-4o"),
        },
    ]

    @classmethod
    def get(cls, model_name: str, **kwargs: Any) -> BaseChatModel:
        """Get an LLM by name with optional argument overrides."""
        model_entry = None
        for entry in cls.LLMS:
            if entry["name"] == model_name:
                model_entry = entry
                break

        if not model_entry:
            available_models = [entry["name"] for entry in cls.LLMS]
            raise ValueError(
                f"model '{model_name}' not found in registry. "
                f"available models: {', '.join(available_models)}"
            )

        if kwargs:
            logger.debug(
                "creating_llm_with_custom_args model_name=%s custom_args=%s",
                model_name,
                ",".join(kwargs.keys()),
            )
            return _build_openai_model(model_name, **kwargs)

        logger.debug("using_registered_llm_instance model_name=%s", model_name)
        return model_entry["llm"]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Get all registered LLM names in order."""
        return [entry["name"] for entry in cls.LLMS]

    @classmethod
    def get_default_name(cls) -> str:
        """Get default model name resolved against the registry."""
        names = cls.get_all_names()
        if settings.default_llm_model in names:
            return settings.default_llm_model
        if names:
            return names[0]
        return settings.default_llm_model


class LLMService:
    """Service for managing LLM calls with retries."""

    def __init__(self) -> None:
        self._llm: Optional[BaseChatModel] = None
        self.default_model_name = settings.default_llm_model

        try:
            self._llm = LLMRegistry.get(self.default_model_name)
            logger.info(
                "llm_service_initialized default_model=%s total_models=%s environment=%s",
                self.default_model_name,
                len(LLMRegistry.LLMS),
                settings.environment,
            )
        except Exception as exc:  # noqa: BLE001
            self._llm = LLMRegistry.LLMS[0]["llm"] if LLMRegistry.LLMS else None
            using_model = LLMRegistry.LLMS[0]["name"] if LLMRegistry.LLMS else "none"
            self.default_model_name = using_model
            logger.warning(
                "default_model_not_found_using_first requested=%s using=%s error=%s",
                settings.default_llm_model,
                using_model,
                str(exc),
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_llm_with_retry(
        self,
        llm: BaseChatModel,
        messages: List[BaseMessage],
    ) -> BaseMessage:
        """Call the LLM with automatic retry logic."""
        try:
            response = await llm.ainvoke(messages)
            logger.debug("llm_call_successful message_count=%s", len(messages))
            return response
        except (RateLimitError, APITimeoutError, APIError) as exc:
            logger.warning(
                "llm_call_failed_retrying error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
            )
            raise
        except OpenAIError as exc:
            logger.error(
                "llm_call_failed error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
            )
            raise

    async def generate_text(
        self,
        *,
        instructions: str,
        input_text: str,
        model: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate plain text from model output."""
        _ = fallback_models
        message = await self.call(
            instructions=instructions,
            input_text=input_text,
            model=model,
            fallback_models=None,
        )
        return self._extract_text(message), {
            "tool_calls": list(getattr(message, "tool_calls", []) or []),
            "response_metadata": getattr(message, "response_metadata", {}),
        }

    async def generate_with_tools(
        self,
        *,
        instructions: str,
        input_text: str,
        tools: list[Any],
        model: str | None = None,
        tool_choice: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate response with tools bound to the selected chat model."""
        _ = fallback_models
        message = await self.call_with_tools(
            instructions=instructions,
            input_text=input_text,
            tools=tools,
            model=model,
            tool_choice=tool_choice,
            fallback_models=None,
        )
        return self._extract_text(message), {
            "tool_calls": list(getattr(message, "tool_calls", []) or []),
            "response_metadata": getattr(message, "response_metadata", {}),
        }

    async def call(
        self,
        *,
        instructions: str,
        input_text: str,
        model: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> AIMessage:
        """Call the LLM once with retry only (no fallback)."""
        _ = fallback_models
        llm = self._resolve_model(model)
        messages = self._build_messages(
            instructions=instructions,
            input_text=input_text,
        )
        response = await self._call_llm_with_retry(llm, messages)
        if isinstance(response, AIMessage):
            return response
        if hasattr(response, "content"):
            return AIMessage(content=getattr(response, "content"))
        return AIMessage(content=str(response))

    async def call_with_tools(
        self,
        *,
        instructions: str,
        input_text: str,
        tools: list[Any],
        model: str | None = None,
        tool_choice: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> AIMessage:
        """Call the LLM once with tools bound and retry only."""
        _ = fallback_models
        base_model = self._resolve_model(model)
        if tool_choice is None:
            runnable = base_model.bind_tools(tools)
        else:
            runnable = base_model.bind_tools(tools, tool_choice=tool_choice)
        messages = self._build_messages(
            instructions=instructions,
            input_text=input_text,
        )
        response = await self._call_llm_with_retry(runnable, messages)
        if isinstance(response, AIMessage):
            return response
        if hasattr(response, "content"):
            return AIMessage(content=getattr(response, "content"))
        return AIMessage(content=str(response))

    def get_llm(self) -> Optional[BaseChatModel]:
        """Get the current LLM instance."""
        return self._llm

    def bind_tools(self, tools: List[Any]) -> "LLMService":
        """Bind tools to the current LLM."""
        if self._llm:
            self._llm = self._llm.bind_tools(tools)
            logger.debug(
                "tools_bound_to_llm",
                extra={"tool_count": len(tools)},
            )
        return self

    def _resolve_model(self, model_name: str | None) -> BaseChatModel:
        if model_name:
            logger.info("llm_using_requested_model model_name=%s", model_name)
            return LLMRegistry.get(model_name)
        if not self._llm:
            raise RuntimeError("llm not initialized")
        logger.info("llm_using_model default (%s)", self.default_model_name)
        return self._llm

    @staticmethod
    def _build_messages(
        *,
        instructions: str,
        input_text: str,
    ) -> list[BaseMessage]:
        return [
            SystemMessage(content=instructions),
            HumanMessage(content=input_text),
        ]

    @staticmethod
    def _extract_text(message: AIMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
        return str(content).strip()


llm_service = LLMService()
