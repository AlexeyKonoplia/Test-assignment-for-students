"""Position quality filtering for navigation records."""

from __future__ import annotations

import pandas as pd

from app.domain import GOOD_POS_TYPES, KMH_FACTOR, POOR_POS_TYPE
from app.handlers.pagination import limited


MAX_POINTS = 1000


def handle(df: pd.DataFrame, params: dict) -> dict:
    """Return points matching requested GNSS quality."""
    quality = params.get("quality", "poor")
    if quality == "good":
        filtered = df.loc[df["pos_type__type"].isin(GOOD_POS_TYPES)]
    else:
        filtered = df.loc[df["pos_type__type"] == POOR_POS_TYPE]

    total = int(len(filtered))
    percentage = round((total / len(df) * 100.0) if len(df) else 0.0, 2)
    page = limited(
        (_row_to_point(row) for _, row in filtered.head(MAX_POINTS).iterrows()),
        total,
        MAX_POINTS,
    )
    return {
        "quality": quality,
        "total_points": total,
        "total_available": page["total_available"],
        "limit": page["limit"],
        "truncated": page["truncated"],
        "percentage_of_total": percentage,
        "points": page["items"],
    }


def _row_to_point(row: pd.Series) -> dict:
    return {
        "timestamp": pd.Timestamp(row["datetime_moscow"]).strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "pos_type": int(row["pos_type__type"]),
        "horizontal_speed": round(float(row["horizontal_speed"] * KMH_FACTOR), 3),
    }
