import pandas as pd

from app.domain import KMH_FACTOR
from app.handlers.pagination import limited


MAX_POINTS = 1000
DEFAULT_START_HOUR = 16
DEFAULT_END_HOUR = 19


def handle(df: pd.DataFrame, params: dict) -> dict:
    """Return points inside a Moscow-hour range."""
    start_hour = int(params.get("start_hour", DEFAULT_START_HOUR))
    end_hour = int(params.get("end_hour", DEFAULT_END_HOUR))
    period = params.get("period", "custom")

    if start_hour > end_hour:
        mask = (df["hour_moscow"] >= start_hour) | (df["hour_moscow"] < end_hour)
    else:
        mask = (df["hour_moscow"] >= start_hour) & (df["hour_moscow"] < end_hour)

    filtered = df.loc[mask]
    total = int(len(filtered))
    page = limited(
        (_row_to_point(row) for _, row in filtered.head(MAX_POINTS).iterrows()),
        total,
        MAX_POINTS,
    )
    return {
        "period": period,
        "time_range": f"{start_hour:02d}:00-{end_hour:02d}:00",
        "total_points": total,
        "total_available": page["total_available"],
        "limit": page["limit"],
        "truncated": page["truncated"],
        "points": page["items"],
    }


def _row_to_point(row: pd.Series) -> dict:
    return {
        "timestamp": pd.Timestamp(row["datetime_moscow"]).strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "horizontal_speed": round(float(row["horizontal_speed"] * KMH_FACTOR), 3),
        "azimuth": round(float(row["azimuth"]), 3),
    }
