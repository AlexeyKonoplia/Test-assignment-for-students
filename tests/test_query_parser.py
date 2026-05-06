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


def test_normalize_time_slice_keeps_custom_hours() -> None:
    normalized = query_parser.normalize_parsed_payload(
        {"intent": "time_slice", "params": {"period": "custom", "start_hour": 10, "end_hour": 11}}
    )

    assert normalized == {
        "intent": "time_slice",
        "params": {"period": "custom", "start_hour": 10, "end_hour": 11},
    }


def test_normalize_time_slice_understands_russian_period_alias() -> None:
    normalized = query_parser.normalize_parsed_payload(
        {"intent": "time_slice", "params": {"period": "сумерки"}}
    )

    assert normalized == {
        "intent": "time_slice",
        "params": {"period": "twilight", "start_hour": 16, "end_hour": 19},
    }


def test_extract_hour_range_from_russian_morning_query() -> None:
    assert query_parser.extract_hour_range("что происходило в с 8 до 9 утра?") == {
        "period": "custom",
        "start_hour": 8,
        "end_hour": 9,
        "start_minute": 0,
        "end_minute": 0,
    }


def test_extract_hour_range_converts_evening_hours() -> None:
    assert query_parser.extract_hour_range("с 8 до 9 вечера") == {
        "period": "custom",
        "start_hour": 20,
        "end_hour": 21,
        "start_minute": 0,
        "end_minute": 0,
    }


def test_extract_hour_range_keeps_minutes() -> None:
    assert query_parser.extract_hour_range("что происходило с 13:30 до 14?") == {
        "period": "custom",
        "start_hour": 13,
        "end_hour": 14,
        "start_minute": 30,
        "end_minute": 0,
    }


def test_extract_single_hour_as_one_hour_range() -> None:
    assert query_parser.extract_time_range("что было в 14?") == {
        "period": "custom",
        "start_hour": 14,
        "end_hour": 15,
        "start_minute": 0,
        "end_minute": 0,
    }


def test_extract_single_evening_hour_as_one_hour_range() -> None:
    assert query_parser.extract_time_range("что было в 8 вечера?") == {
        "period": "custom",
        "start_hour": 20,
        "end_hour": 21,
        "start_minute": 0,
        "end_minute": 0,
    }
