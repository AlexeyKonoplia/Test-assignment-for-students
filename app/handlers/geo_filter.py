"""Geographic bounding-box filtering."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.domain import KMH_FACTOR
from app.handlers.pagination import limited


MAX_POINTS = 1000
EARTH_KM_PER_DEGREE = 111.32


def handle(df: pd.DataFrame, params: dict) -> dict:
    """Return navigation points inside the requested latitude/longitude bounds."""
    lat_min = float(params["lat_min"])
    lat_max = float(params["lat_max"])
    lon_min = float(params["lon_min"])
    lon_max = float(params["lon_max"])
    region = params.get("region", "custom")

    mask = (
        (df["latitude"] >= lat_min)
        & (df["latitude"] <= lat_max)
        & (df["longitude"] >= lon_min)
        & (df["longitude"] <= lon_max)
    )
    if params.get("route"):
        mask = mask & _route_corridor_mask(df, params["route"], float(params.get("corridor_km", 15.0)))

    filtered = df.loc[mask].sort_values(["_bag_id", "_timestamp"])
    visits = _visits(df, mask)
    entry_time = visits[0]["entry_time"] if visits else None
    exit_time = visits[-1]["exit_time"] if visits else None
    time_spent_minutes = round(float(sum(visit["duration_minutes"] for visit in visits)), 2)
    total = int(len(filtered))
    page = limited(
        (_row_to_point(row) for _, row in filtered.head(MAX_POINTS).iterrows()),
        total,
        MAX_POINTS,
    )

    return {
        "region": region,
        "bounds": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
        "total_points": total,
        "total_available": page["total_available"],
        "limit": page["limit"],
        "truncated": page["truncated"],
        "entry_time": entry_time,
        "exit_time": exit_time,
        "time_spent_minutes": time_spent_minutes,
        "visits": visits,
        "points": page["items"],
    }


def _visits(df: pd.DataFrame, mask: pd.Series) -> list[dict]:
    visits: list[dict] = []
    working = df.assign(_inside_region=mask).sort_values(["_bag_id", "_timestamp"])
    for bag_id, bag in working.groupby("_bag_id", sort=False):
        inside = bag["_inside_region"]
        starts = inside & ~inside.shift(fill_value=False)
        segment_ids = starts.cumsum()
        for _, segment in bag.loc[inside].groupby(segment_ids[inside]):
            entry_row = segment.iloc[0]
            exit_row = segment.iloc[-1]
            visits.append(
                {
                    "bag_id": str(bag_id),
                    "entry_time": _format_timestamp(entry_row),
                    "exit_time": _format_timestamp(exit_row),
                    "duration_minutes": round(
                        float((exit_row["_timestamp"] - entry_row["_timestamp"]) / 60.0),
                        2,
                    ),
                    "points": int(len(segment)),
                }
            )
    return visits


def _route_corridor_mask(df: pd.DataFrame, route: list[dict], corridor_km: float) -> pd.Series:
    if len(route) < 2:
        return pd.Series([True] * len(df), index=df.index)

    distances = np.full(len(df), np.inf)
    latitudes = df["latitude"].to_numpy(dtype=float)
    longitudes = df["longitude"].to_numpy(dtype=float)
    for index in range(len(route) - 1):
        segment_distances = _distance_to_segment_array_km(
            latitudes,
            longitudes,
            float(route[index]["lat"]),
            float(route[index]["lon"]),
            float(route[index + 1]["lat"]),
            float(route[index + 1]["lon"]),
        )
        distances = np.minimum(distances, segment_distances)
    return pd.Series(distances <= corridor_km, index=df.index)


def _distance_to_segment_array_km(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> np.ndarray:
    ref_lat = np.radians((latitudes + start_lat + end_lat) / 3.0)
    px = longitudes * EARTH_KM_PER_DEGREE * np.cos(ref_lat)
    py = latitudes * EARTH_KM_PER_DEGREE
    ax = start_lon * EARTH_KM_PER_DEGREE * np.cos(ref_lat)
    ay = start_lat * EARTH_KM_PER_DEGREE
    bx = end_lon * EARTH_KM_PER_DEGREE * np.cos(ref_lat)
    by = end_lat * EARTH_KM_PER_DEGREE
    dx = bx - ax
    dy = by - ay
    denominator = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / denominator
    t = np.clip(np.nan_to_num(t), 0.0, 1.0)
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return np.hypot(px - nearest_x, py - nearest_y)


def _distance_to_route_km(lat: float, lon: float, route: list[dict]) -> float:
    return min(
        _distance_to_segment_km(lat, lon, route[index], route[index + 1])
        for index in range(len(route) - 1)
    )


def _distance_to_segment_km(lat: float, lon: float, start: dict, end: dict) -> float:
    ref_lat = math.radians((lat + float(start["lat"]) + float(end["lat"])) / 3.0)

    def project(point_lat: float, point_lon: float) -> tuple[float, float]:
        return (
            point_lon * EARTH_KM_PER_DEGREE * math.cos(ref_lat),
            point_lat * EARTH_KM_PER_DEGREE,
        )

    px, py = project(lat, lon)
    ax, ay = project(float(start["lat"]), float(start["lon"]))
    bx, by = project(float(end["lat"]), float(end["lon"]))
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def _row_to_point(row: pd.Series) -> dict:
    return {
        "timestamp": _format_timestamp(row),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "speed": round(float(row["horizontal_speed"] * KMH_FACTOR), 3),
    }


def _format_timestamp(row: pd.Series) -> str:
    return pd.Timestamp(row["datetime_moscow"]).strftime("%Y-%m-%d %H:%M:%S")
