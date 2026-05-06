import json
import re
from typing import Any

from app.llm_client import LLMClient, LLMUnavailableError
from app.query.contracts import ParsedQuery, normalize_payload
from app.query.prompt import build_prompt


JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
HOUR_RANGE_PATTERN = re.compile(
    r"(?:\bс\s*)?(\d{1,2})(?::(\d{2}))?\s*(?:до|-|–|—|to)\s*(\d{1,2})(?::(\d{2}))?"
    r"\s*(утра|вечера|дня|ночи|am|pm)?",
    re.IGNORECASE,
)


async def parse(query: str, client: LLMClient | None = None) -> ParsedQuery:
    """Parse a natural-language query into an intent and normalized parameters."""
    explicit_time_range = extract_hour_range(query)
    llm = client or LLMClient()
    try:
        raw = await llm.complete(build_prompt(query))
    except LLMUnavailableError as exc:
        if explicit_time_range:
            return ParsedQuery("time_slice", explicit_time_range, query)
        return ParsedQuery("unknown", {"reason": str(exc)}, query)

    parsed = parse_llm_response(raw)
    normalized = normalize_parsed_payload(parsed)
    if explicit_time_range:
        normalized = {"intent": "time_slice", "params": explicit_time_range}
    return ParsedQuery(
        intent=str(normalized.get("intent", "unknown")),
        params=dict(normalized.get("params", {})),
        original_query=query,
    )


def normalize_parsed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM-provided parameters against typed Python-owned contracts."""
    return normalize_payload(payload)


def parse_llm_response(raw: str) -> dict[str, Any]:
    """Parse raw LLM text as JSON without raising exceptions."""
    try:
        parsed = json.loads(raw.strip())
        return parsed if isinstance(parsed, dict) else _parse_error(raw)
    except json.JSONDecodeError:
        pass

    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else _parse_error(raw)
        except json.JSONDecodeError:
            continue

    return _parse_error(raw)


def extract_hour_range(query: str) -> dict[str, Any] | None:
    """Extract explicit hour ranges like 'с 8 до 9 утра' from the original query."""
    match = HOUR_RANGE_PATTERN.search(query)
    if not match:
        return None

    start_hour = _normalize_hour(int(match.group(1)), match.group(5))
    end_hour = _normalize_hour(int(match.group(3)), match.group(5))
    start_minute = _normalize_minute(match.group(2))
    end_minute = _normalize_minute(match.group(4))
    if not 0 <= start_hour <= 24 or not 0 <= end_hour <= 24:
        return None
    if not 0 <= start_minute <= 59 or not 0 <= end_minute <= 59:
        return None
    if start_hour == end_hour and start_minute == end_minute:
        return None
    if start_hour == 24 and start_minute != 0:
        return None
    if end_hour == 24 and end_minute != 0:
        return None

    return {
        "period": "custom",
        "start_hour": start_hour,
        "end_hour": end_hour,
        "start_minute": start_minute,
        "end_minute": end_minute,
    }


def _parse_error(raw: str) -> dict[str, Any]:
    return {"intent": "unknown", "params": {"reason": "LLM parse error: " + raw[:200]}}


def _normalize_hour(hour: int, meridiem: str | None) -> int:
    if hour == 24:
        return 24
    if not 0 <= hour <= 23:
        return -1

    if meridiem is None:
        return hour

    normalized = meridiem.lower()
    if normalized in {"pm", "вечера"} and 1 <= hour <= 11:
        return hour + 12
    if normalized == "дня" and 1 <= hour <= 7:
        return hour + 12
    if normalized in {"am", "утра", "ночи"} and hour == 12:
        return 0
    return hour


def _normalize_minute(raw_minute: str | None) -> int:
    if raw_minute is None:
        return 0
    minute = int(raw_minute)
    return minute if 0 <= minute <= 59 else -1


def _json_candidates(raw: str) -> list[str]:
    fenced = [match.group(1).strip() for match in JSON_FENCE_PATTERN.finditer(raw)]
    balanced = _balanced_json_objects(raw)
    return fenced + [candidate for candidate in balanced if candidate not in fenced]


def _balanced_json_objects(raw: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(raw[start : index + 1])
                start = None

    return candidates
