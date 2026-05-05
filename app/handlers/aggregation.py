"""Aggregation computations over navigation records."""

from __future__ import annotations

import pandas as pd

from app.domain import KMH_FACTOR


METRIC_MAX_SPEED = "max_speed"
METRIC_MIN_SPEED = "min_speed"
METRIC_AVG_SPEED = "avg_speed"
METRIC_MAX_ACCELERATION = "max_acceleration"
METRIC_TOTAL_DISTANCE = "total_distance"


def handle(df: pd.DataFrame, params: dict) -> dict:
    """Handle aggregate metric queries."""
    metric = params.get("metric", METRIC_MAX_SPEED)
    working = df

    if working.empty:
        return {"metric": metric, "total_points": 0, "value": None}

    if metric == METRIC_MAX_SPEED:
        return _point_metric(working, metric, working["horizontal_speed"].idxmax(), "speed_kmh", len(working))

    if metric == METRIC_MIN_SPEED:
        moving = working.loc[working["horizontal_speed"] > 0]
        target = moving if not moving.empty else working
        return _point_metric(target, metric, target["horizontal_speed"].idxmin(), "speed_kmh", len(working))

    if metric == METRIC_AVG_SPEED:
        return {
            "metric": metric,
            "total_points": int(len(working)),
            "avg_speed_kmh": round(float(working["horizontal_speed"].mean() * KMH_FACTOR), 3),
        }

    if metric == METRIC_MAX_ACCELERATION:
        idx = working["acceleration"].abs().idxmax()
        row = working.loc[idx]
        return {
            "metric": metric,
            "total_points": int(len(working)),
            "max_acceleration_mps2": round(float(abs(row["acceleration"])), 3),
            "signed_acceleration_mps2": round(float(row["acceleration"]), 3),
            "timestamp": _format_timestamp(row),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }

    if metric == METRIC_TOTAL_DISTANCE:
        return _total_distance(working)

    return {"error": "unsupported_metric", "metric": metric}


def _point_metric(
    df: pd.DataFrame, metric: str, idx: int, value_name: str, total_points: int
) -> dict:
    row = df.loc[idx]
    return {
        "metric": metric,
        "total_points": int(total_points),
        value_name: round(float(row["horizontal_speed"] * KMH_FACTOR), 3),
        "timestamp": _format_timestamp(row),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
    }


def _total_distance(df: pd.DataFrame) -> dict:
    working = df.sort_values(["_bag_id", "_timestamp"])
    per_bag: list[dict] = []
    total_meters = 0.0
    for bag_id, group in working.groupby("_bag_id", sort=False):
        bag = group.sort_values("_timestamp")
        dt = bag["_timestamp"].diff().fillna(0).clip(lower=0)
        distance_m = float((bag["horizontal_speed"] * dt).sum())
        total_meters += distance_m
        per_bag.append({"bag_id": str(bag_id), "distance_km": round(distance_m / 1000.0, 3)})

    return {
        "metric": METRIC_TOTAL_DISTANCE,
        "total_points": int(len(working)),
        "total_distance_km": round(total_meters / 1000.0, 3),
        "per_bag": per_bag,
    }


def _format_timestamp(row: pd.Series) -> str:
    return pd.Timestamp(row["datetime_moscow"]).strftime("%Y-%m-%d %H:%M:%S")
