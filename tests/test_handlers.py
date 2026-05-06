"""Unit tests for pure navigation handlers."""

from __future__ import annotations

import pandas as pd

from app.handlers import aggregation, braking, filter_quality, geo_filter, time_slice


def test_max_speed_returns_correct_value(nav_df: pd.DataFrame) -> None:
    result = aggregation.handle(nav_df, {"metric": "max_speed"})
    expected = round(float(nav_df["horizontal_speed"].max() * 3.6), 3)
    assert result["speed_kmh"] == expected
    assert "timestamp" in result
    assert "latitude" in result


def test_filter_quality_poor_returns_only_pos19(nav_df: pd.DataFrame) -> None:
    result = filter_quality.handle(nav_df, {"quality": "poor"})
    assert result["total_points"] >= 20
    assert result["points"]
    assert all(point["pos_type"] == 19 for point in result["points"])


def test_time_slice_twilight_returns_hours_16_to_19(nav_df: pd.DataFrame) -> None:
    result = time_slice.handle(nav_df, {"period": "twilight", "start_hour": 16, "end_hour": 19})
    assert result["total_points"] > 0
    timestamps = pd.to_datetime([point["timestamp"] for point in result["points"]])
    assert all(16 <= timestamp.hour < 19 for timestamp in timestamps)


def test_time_slice_custom_returns_requested_hour_range(nav_df: pd.DataFrame) -> None:
    result = time_slice.handle(nav_df, {"period": "custom", "start_hour": 16, "end_hour": 17})
    assert result["total_points"] > 0
    timestamps = pd.to_datetime([point["timestamp"] for point in result["points"]])
    assert all(16 <= timestamp.hour < 17 for timestamp in timestamps)


def test_time_slice_custom_returns_requested_minute_range(nav_df: pd.DataFrame) -> None:
    result = time_slice.handle(
        nav_df,
        {
            "period": "custom",
            "start_hour": 16,
            "start_minute": 30,
            "end_hour": 17,
            "end_minute": 0,
        },
    )
    assert result["total_points"] > 0
    assert result["time_range"] == "16:30-17:00"
    timestamps = pd.to_datetime([point["timestamp"] for point in result["points"]])
    assert all(
        (timestamp.hour * 60 + timestamp.minute) in range(16 * 60 + 30, 17 * 60)
        for timestamp in timestamps
    )


def test_time_slice_custom_wraps_midnight(nav_df: pd.DataFrame) -> None:
    result = time_slice.handle(nav_df, {"period": "custom", "start_hour": 22, "end_hour": 5})
    timestamps = pd.to_datetime([point["timestamp"] for point in result["points"]])
    assert all(timestamp.hour >= 22 or timestamp.hour < 5 for timestamp in timestamps)


def test_time_slice_returns_points_sorted_by_time(nav_df: pd.DataFrame) -> None:
    shuffled = nav_df.sample(frac=1, random_state=7).reset_index(drop=True)

    result = time_slice.handle(shuffled, {"period": "all_day", "start_hour": 0, "end_hour": 24})

    timestamps = pd.to_datetime([point["timestamp"] for point in result["points"]])
    assert timestamps.is_monotonic_increasing


def test_time_slice_empty_result_explains_loaded_data_range(nav_df: pd.DataFrame) -> None:
    result = time_slice.handle(nav_df, {"period": "twilight", "start_hour": 20, "end_hour": 21})

    assert result["total_points"] == 0
    assert result["points"] == []
    assert result["data_time_range"]
    assert "No points found in requested time range 20:00-21:00" in result["message"]


def test_braking_groups_consecutive_rows_as_one_event(nav_df: pd.DataFrame) -> None:
    result = braking.handle(nav_df, {"threshold": -0.04})
    assert result["total_braking_events"] == 2
    assert result["events"][0]["duration_seconds"] == 240


def test_geo_filter_m11_bounds(nav_df: pd.DataFrame) -> None:
    params = {"region": "м11", "lat_min": 55.5, "lat_max": 60.0, "lon_min": 30.0, "lon_max": 37.5}
    result = geo_filter.handle(nav_df, params)
    assert result["total_points"] == 140
    assert result["entry_time"] is not None
    assert result["exit_time"] is not None
    assert all(55.5 <= point["latitude"] <= 60.0 for point in result["points"])
    assert all(30.0 <= point["longitude"] <= 37.5 for point in result["points"])


def test_geo_filter_time_spent_sums_separate_visits(nav_df: pd.DataFrame) -> None:
    working = nav_df.copy()
    gap_index = working[(working["_bag_id"] == "bag_a") & (working["nov_header__sequence_number"] == 20)].index
    working.loc[gap_index, "latitude"] = 61.0
    params = {"region": "custom", "lat_min": 55.5, "lat_max": 60.0, "lon_min": 30.0, "lon_max": 37.5}

    result = geo_filter.handle(working, params)

    assert result["total_points"] == 139
    assert len(result["visits"]) == 3
    assert result["time_spent_minutes"] == 136.0


def test_list_handlers_expose_truncation_metadata(nav_df: pd.DataFrame) -> None:
    result = time_slice.handle(nav_df, {"period": "all_day", "start_hour": 0, "end_hour": 24})

    assert result["total_available"] == result["total_points"]
    assert result["limit"] == 1000
    assert result["truncated"] is False


def test_braking_respects_bag_id_boundary(nav_df: pd.DataFrame) -> None:
    result = braking.handle(nav_df, {"threshold": -0.04})
    event_starts = [event["start_timestamp"] for event in result["events"]]
    first_rows = nav_df.sort_values(["_bag_id", "_timestamp"]).groupby("_bag_id").head(1)
    first_timestamps = {
        timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in first_rows["datetime_moscow"]
    }
    assert not first_timestamps.intersection(event_starts)
