"""Async OpenRouter client with retry and fallback model support."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx


DEFAULT_TIMEOUT_SECONDS = 20.0
RETRYABLE_STATUS_CODES = {429, 503}
BACKOFF_SECONDS = (1.0, 2.0, 4.0)
HTTP_REFERER = "https://semantic-nav-search.local"
APP_TITLE = "semantic-nav-search"


class LLMUnavailableError(RuntimeError):
    """Raised when primary and fallback LLM models cannot produce a response."""


class LLMResponseError(RuntimeError):
    """Raised when an LLM response has an unexpected shape."""


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for OpenRouter."""

    api_key: str
    model: str
    fallback_models: list[str]
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create configuration from environment variables."""
        api_key = _required_env("OPENROUTER_API_KEY")
        model = _required_env("OPENROUTER_MODEL")

        fallback_models = [
            model.strip()
            for model in _required_env("OPENROUTER_FALLBACK_MODELS").split(",")
            if model.strip()
        ]
        if not fallback_models:
            raise LLMUnavailableError("OPENROUTER_FALLBACK_MODELS must contain at least one model")

        timeout_raw = os.getenv("LLM_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise LLMUnavailableError(f"LLM_TIMEOUT must be a number, got: {timeout_raw}") from exc

        return cls(
            api_key=api_key,
            model=model,
            fallback_models=fallback_models,
            timeout_seconds=timeout_seconds,
        )


class LLMClient:
    """Client for OpenRouter chat completions."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()

    async def complete(self, prompt: str) -> str:
        """Return raw text content from the LLM response."""
        last_error = "unknown LLM error"
        for attempt, delay in enumerate(BACKOFF_SECONDS, start=1):
            try:
                return await self._complete_with_model(self.config.model, prompt)
            except (
                httpx.TimeoutException,
                httpx.HTTPStatusError,
                httpx.RequestError,
                LLMResponseError,
            ) as exc:
                last_error = str(exc)
                if not self._is_retryable(exc) or attempt == len(BACKOFF_SECONDS):
                    break
                await asyncio.sleep(delay)

        for fallback_model in self.config.fallback_models:
            try:
                return await self._complete_with_model(fallback_model, prompt)
            except (
                httpx.TimeoutException,
                httpx.HTTPStatusError,
                httpx.RequestError,
                LLMResponseError,
            ) as exc:
                last_error = str(exc)

        raise LLMUnavailableError(f"All LLM models failed. Last error: {last_error}")

    async def _complete_with_model(self, model: str, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": APP_TITLE,
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        timeout = httpx.Timeout(self.config.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMResponseError(f"Unexpected LLM response schema: {exc}") from exc

            if not isinstance(content, str) or not content.strip():
                raise LLMResponseError("LLM response content is empty or not a string")
            return content

    @staticmethod
    def _is_retryable(
        exc: httpx.TimeoutException | httpx.HTTPStatusError | httpx.RequestError | LLMResponseError,
    ) -> bool:
        if isinstance(exc, httpx.TimeoutException):
            return True
        if isinstance(exc, httpx.RequestError):
            return True
        if isinstance(exc, LLMResponseError):
            return False
        return exc.response.status_code in RETRYABLE_STATUS_CODES


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise LLMUnavailableError(f"{name} is required")
    return value
