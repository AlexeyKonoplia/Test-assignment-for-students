"""Typed contracts for parsed natural-language queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.domain import (
    DEFAULT_BRAKING_THRESHOLD,
    GEO_REGION_ALIASES,
    GEO_REGIONS,
    M11_CORRIDOR_KM,
    M11_ROUTE,
    TIME_PERIOD_ALIASES,
    TIME_PERIODS,
)


Intent = Literal["aggregation", "filter_quality", "time_slice", "braking", "geo_filter", "unknown"]

VALID_INTENTS = {"aggregation", "filter_quality", "time_slice", "braking", "geo_filter", "unknown"}
VALID_METRICS = {"max_speed", "min_speed", "avg_speed", "max_acceleration", "total_distance"}
VALID_QUALITIES = {"poor", "good"}


class StrictParams(BaseModel):
    model_config = ConfigDict(extra="ignore")


class AggregationParams(StrictParams):
    metric: str = "max_speed"

    @field_validator("metric")
    @classmethod
    def known_metric(cls, value: str) -> str:
        return value if value in VALID_METRICS else "max_speed"


class FilterQualityParams(StrictParams):
    quality: str = "poor"

    @field_validator("quality")
    @classmethod
    def known_quality(cls, value: str) -> str:
        return value if value in VALID_QUALITIES else "poor"


class TimeSliceParams(StrictParams):
    period: str = "custom"
    start_hour: int = Field(default=16, ge=0, le=24)
    end_hour: int = Field(default=19, ge=0, le=24)
    start_minute: int | None = Field(default=None, ge=0, le=59)
    end_minute: int | None = Field(default=None, ge=0, le=59)

    @model_validator(mode="after")
    def whole_hour_for_24(self) -> "TimeSliceParams":
        if self.start_hour == 24 and (self.start_minute or 0) != 0:
            raise ValueError("start_minute must be 0 when start_hour is 24")
        if self.end_hour == 24 and (self.end_minute or 0) != 0:
            raise ValueError("end_minute must be 0 when end_hour is 24")
        return self

    @classmethod
    def from_period(cls, period: str) -> "TimeSliceParams":
        normalized_period = normalize_time_period(period) or "twilight"
        start_hour, end_hour = TIME_PERIODS[normalized_period]
        return cls(period=normalized_period, start_hour=start_hour, end_hour=end_hour)

    @classmethod
    def from_payload(cls, params: dict[str, Any]) -> "TimeSliceParams":
        period = normalize_time_period(params.get("period"))
        has_start = "start_hour" in params
        has_end = "end_hour" in params

        if has_start and has_end:
            return cls(
                period=period or "custom",
                start_hour=params["start_hour"],
                end_hour=params["end_hour"],
                start_minute=params.get("start_minute"),
                end_minute=params.get("end_minute"),
            )

        if period:
            return cls.from_period(period)

        return cls.from_period("twilight")


class BrakingParams(StrictParams):
    threshold: float = DEFAULT_BRAKING_THRESHOLD

    @field_validator("threshold", mode="before")
    @classmethod
    def numeric_threshold(cls, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return DEFAULT_BRAKING_THRESHOLD


class GeoFilterParams(StrictParams):
    region: str = "m11"
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    route: list[dict[str, float]] | None = None
    corridor_km: float | None = None

    @classmethod
    def from_region(cls, raw_region: Any) -> "GeoFilterParams | None":
        normalized = normalize_region(raw_region)
        bounds = GEO_REGIONS.get(normalized)
        if bounds is None:
            return None
        params: dict[str, Any] = {"region": normalized, **bounds}
        if normalized == "m11":
            params.update({"route": M11_ROUTE, "corridor_km": M11_CORRIDOR_KM})
        return cls(**params)


class UnknownParams(StrictParams):
    reason: str = "Unknown query"


@dataclass(frozen=True)
class ParsedQuery:
    """Structured representation of a user's natural-language query."""

    intent: str
    params: dict[str, Any]
    original_query: str


def normalize_region(raw_region: Any) -> str:
    region = str(raw_region or "m11").strip()
    return GEO_REGION_ALIASES.get(region, GEO_REGION_ALIASES.get(region.lower(), region.lower()))


def normalize_time_period(raw_period: Any) -> str | None:
    period = str(raw_period or "").strip()
    if not period:
        return None
    normalized = TIME_PERIOD_ALIASES.get(period, TIME_PERIOD_ALIASES.get(period.lower()))
    return normalized if normalized in TIME_PERIODS else None


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an LLM payload into handler-ready params."""
    intent = str(payload.get("intent", "unknown"))
    params = payload.get("params", {})
    if not isinstance(params, dict) or intent not in VALID_INTENTS:
        return _unknown("Invalid LLM schema")

    try:
        if intent == "aggregation":
            return {"intent": intent, "params": AggregationParams(**params).model_dump()}
        if intent == "filter_quality":
            return {"intent": intent, "params": FilterQualityParams(**params).model_dump()}
        if intent == "time_slice":
            time_params = TimeSliceParams.from_payload(params)
            return {
                "intent": intent,
                "params": time_params.model_dump(exclude_none=True),
            }
        if intent == "braking":
            return {"intent": intent, "params": BrakingParams(**params).model_dump()}
        if intent == "geo_filter":
            geo_params = GeoFilterParams.from_region(params.get("region", "m11"))
            if geo_params is None:
                return _unknown(f"Unknown geo region: {normalize_region(params.get('region'))}")
            return {"intent": intent, "params": geo_params.model_dump(exclude_none=True)}
        return _unknown(str(params.get("reason", "Unknown query")))
    except ValidationError as exc:
        return _unknown(f"Invalid parameters for {intent}: {exc.errors()[0]['msg']}")


def _unknown(reason: str) -> dict[str, Any]:
    return {"intent": "unknown", "params": UnknownParams(reason=reason).model_dump()}
