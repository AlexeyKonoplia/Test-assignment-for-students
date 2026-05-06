import pandas as pd

from app.domain import KMH_FACTOR
from app.handlers.pagination import limited


MAX_POINTS = 1000
DEFAULT_START_HOUR = 16
DEFAULT_END_HOUR = 19
DEFAULT_MINUTE = 0


def handle(df: pd.DataFrame, params: dict) -> dict:
    """Return points inside a Moscow time range."""
    start_hour = int(params.get("start_hour", DEFAULT_START_HOUR))
    end_hour = int(params.get("end_hour", DEFAULT_END_HOUR))
    start_minute = int(params.get("start_minute", DEFAULT_MINUTE))
    end_minute = int(params.get("end_minute", DEFAULT_MINUTE))
    period = params.get("period", "custom")
    start_total_minutes = start_hour * 60 + start_minute
    end_total_minutes = end_hour * 60 + end_minute
    row_total_minutes = df["datetime_moscow"].dt.hour * 60 + df["datetime_moscow"].dt.minute

    if start_total_minutes > end_total_minutes:
        mask = (row_total_minutes >= start_total_minutes) | (row_total_minutes < end_total_minutes)
    else:
        mask = (row_total_minutes >= start_total_minutes) & (row_total_minutes < end_total_minutes)

    filtered = df.loc[mask]
    total = int(len(filtered))
    page = limited(
        (_row_to_point(row) for _, row in filtered.head(MAX_POINTS).iterrows()),
        total,
        MAX_POINTS,
    )
    return {
        "period": period,
        "time_range": f"{start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}",
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
