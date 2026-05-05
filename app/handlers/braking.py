import pandas as pd

from app.domain import DEFAULT_BRAKING_THRESHOLD, KMH_FACTOR
from app.handlers.pagination import limited


MAX_EVENTS = 1000


def handle(df: pd.DataFrame, params: dict) -> dict:
    """Group consecutive records below the deceleration threshold into events."""
    threshold = float(params.get("threshold", DEFAULT_BRAKING_THRESHOLD))
    working = df.sort_values(["_bag_id", "_timestamp"]).reset_index(drop=True)
    events = []

    for _, bag in working.groupby("_bag_id", sort=False):
        bag_events = _events_for_bag(bag.reset_index(drop=True), threshold)
        events.extend(bag_events)

    decelerations = [event["max_deceleration"] for event in events]
    page = limited(events[:MAX_EVENTS], len(events), MAX_EVENTS)
    return {
        "threshold": threshold,
        "total_braking_events": len(events),
        "total_available": page["total_available"],
        "limit": page["limit"],
        "truncated": page["truncated"],
        "max_deceleration": round(float(min(decelerations)), 3) if decelerations else None,
        "avg_deceleration": round(float(sum(decelerations) / len(decelerations)), 3)
        if decelerations
        else None,
        "events": page["items"],
    }


def _events_for_bag(df: pd.DataFrame, threshold: float) -> list[dict]:
    mask = df["acceleration"] < threshold
    groups = (mask & ~mask.shift(fill_value=False)).cumsum()
    events = []
    for _, event_df in df[mask].groupby(groups[mask]):
        start_idx = int(event_df.index.min())
        end_idx = int(event_df.index.max())
        peak_idx = int(event_df["acceleration"].idxmin())
        peak_row = df.loc[peak_idx]
        before_idx = max(start_idx - 1, 0)
        after_idx = min(end_idx + 1, len(df) - 1)
        start_row = df.loc[start_idx]
        end_row = df.loc[end_idx]
        duration = int(end_row["_timestamp"] - start_row["_timestamp"])
        events.append(
            {
                "start_timestamp": _format_timestamp(start_row),
                "end_timestamp": _format_timestamp(end_row),
                "latitude": float(peak_row["latitude"]),
                "longitude": float(peak_row["longitude"]),
                "max_deceleration": round(float(peak_row["acceleration"]), 3),
                "speed_before": round(float(df.loc[before_idx, "horizontal_speed"] * KMH_FACTOR), 3),
                "speed_after": round(float(df.loc[after_idx, "horizontal_speed"] * KMH_FACTOR), 3),
                "duration_seconds": duration,
            }
        )
    return events


def _format_timestamp(row: pd.Series) -> str:
    return pd.Timestamp(row["datetime_moscow"]).strftime("%Y-%m-%d %H:%M:%S")
