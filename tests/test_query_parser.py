"""Tests for natural-language query parsing helpers."""

from __future__ import annotations

from app.query import parser as query_parser


def test_build_prompt_is_russian_first_but_keeps_english_examples() -> None:
    prompt = query_parser.build_prompt("где были проблемы с gps?")

    assert "Основной язык пользователей - русский" in prompt
    assert "Верни ТОЛЬКО валидный JSON" in prompt
    assert "Пользователь: где были проблемы с gps?" in prompt
    assert "Пользователь: top speed" in prompt
    assert "Пользователь: points in Moscow" in prompt


def test_normalize_aggregation_metric() -> None:
    payload = {"intent": "aggregation", "params": {"metric": "max_speed"}}

    assert query_parser.normalize_parsed_payload(payload) == {
        "intent": "aggregation",
        "params": {"metric": "max_speed"},
    }


def test_parse_llm_response_handles_nested_json_after_text() -> None:
    raw = 'Sure: {"intent":"geo_filter","params":{"region":"m11","meta":{"source":"test"}}}'

    assert query_parser.parse_llm_response(raw) == {
        "intent": "geo_filter",
        "params": {"region": "m11", "meta": {"source": "test"}},
    }


def test_parse_llm_response_prefers_fenced_json() -> None:
    raw = 'text {"broken": true\n```json\n{"intent":"aggregation","params":{"metric":"avg_speed"}}\n```'

    assert query_parser.parse_llm_response(raw) == {
        "intent": "aggregation",
        "params": {"metric": "avg_speed"},
    }


def test_normalize_m11_adds_route_corridor() -> None:
    normalized = query_parser.normalize_parsed_payload(
        {"intent": "geo_filter", "params": {"region": "m11"}}
    )

    assert normalized["params"]["region"] == "m11"
    assert normalized["params"]["route"]
    assert normalized["params"]["corridor_km"] > 0
