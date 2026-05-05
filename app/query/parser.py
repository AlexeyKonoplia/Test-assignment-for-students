import json
import re
from typing import Any

from app.llm_client import LLMClient, LLMUnavailableError
from app.query.contracts import ParsedQuery, normalize_payload
from app.query.prompt import build_prompt


JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


async def parse(query: str, client: LLMClient | None = None) -> ParsedQuery:
    """Parse a natural-language query into an intent and normalized parameters."""
    llm = client or LLMClient()
    try:
        raw = await llm.complete(build_prompt(query))
    except LLMUnavailableError as exc:
        return ParsedQuery("unknown", {"reason": str(exc)}, query)

    parsed = parse_llm_response(raw)
    normalized = normalize_parsed_payload(parsed)
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


def _parse_error(raw: str) -> dict[str, Any]:
    return {"intent": "unknown", "params": {"reason": "LLM parse error: " + raw[:200]}}


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
