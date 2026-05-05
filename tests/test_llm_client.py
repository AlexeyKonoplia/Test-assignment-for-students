"""Tests for OpenRouter runtime configuration."""

from __future__ import annotations

import pytest

from app.llm_client import LLMConfig, LLMUnavailableError


def test_llm_config_requires_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "fallback-model")

    with pytest.raises(LLMUnavailableError, match="OPENROUTER_MODEL is required"):
        LLMConfig.from_env()


def test_llm_config_reads_models_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "primary-model")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "fallback-a, fallback-b")
    monkeypatch.setenv("LLM_TIMEOUT", "10")

    config = LLMConfig.from_env()

    assert config.model == "primary-model"
    assert config.fallback_models == ["fallback-a", "fallback-b"]
    assert config.timeout_seconds == 10.0
