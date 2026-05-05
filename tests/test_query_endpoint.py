"""Integration tests for the /query endpoint with mocked LLM calls."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.data_loader import DataStore
from app.llm_client import LLMUnavailableError
from app.main import app
from app.query import parser as query_parser


class FakeLLMClient:
    response = '{"intent":"aggregation","params":{"metric":"max_speed"}}'
    error: Exception | None = None

    async def complete(self, prompt: str) -> str:
        if self.error:
            raise self.error
        return self.response


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, nav_df):
    FakeLLMClient.response = '{"intent":"aggregation","params":{"metric":"max_speed"}}'
    FakeLLMClient.error = None
    monkeypatch.setattr(query_parser, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(DataStore, "_df", nav_df)
    return TestClient(app)


def test_query_endpoint_handles_russian_query_through_mocked_llm(client: TestClient) -> None:
    response = client.post("/query", json={"query": "какая была максимальная скорость?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["intent"] == "aggregation"
    assert payload["result"]["metric"] == "max_speed"


def test_query_endpoint_returns_422_for_unknown_intent(client: TestClient) -> None:
    FakeLLMClient.response = '{"intent":"unknown","params":{"reason":"not navigation"}}'

    response = client.post("/query", json={"query": "что приготовить на ужин?"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["status"] == "error"
    assert detail["result"]["error"] == "unknown_intent"


def test_query_endpoint_isolates_llm_outage(client: TestClient) -> None:
    FakeLLMClient.error = LLMUnavailableError("OpenRouter unavailable")

    response = client.post("/query", json={"query": "покажи резкие торможения"})

    assert response.status_code == 422
    assert response.json()["detail"]["status"] == "error"


def test_query_endpoint_handles_malformed_llm_json(client: TestClient) -> None:
    FakeLLMClient.response = '{"intent":"aggregation","params":{"metric":"max_speed"}'

    response = client.post("/query", json={"query": "максималка"})

    assert response.status_code == 422
    assert response.json()["detail"]["result"]["error"] == "unknown_intent"
