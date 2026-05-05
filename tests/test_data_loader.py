"""Tests for CSV loading timestamp normalization."""

from __future__ import annotations

import pandas as pd

from app.data_loader import compute_derived_metrics


def test_compute_derived_metrics_accepts_nanosecond_timestamps() -> None:
    base_seconds = 1_699_948_939
    df = pd.DataFrame(
        [
            {
                "_id": "row_0",
                "_bag_id": "bag",
                "_timestamp": base_seconds * 1_000_000_000,
                "_topic": "/gnss/novatel/oem7/inspvax",
                "_type": "novatel_oem7_msgs/msg/INSPVAX",
                "header__stamp__sec": base_seconds,
                "header__stamp__nanosec": 0,
                "header__frame_id": "gps",
                "nov_header__message_name": "INSPVAX",
                "nov_header__message_id": 1465,
                "nov_header__message_type": 0,
                "nov_header__sequence_number": 0,
                "nov_header__time_status": 180,
                "nov_header__gps_week_number": 2288,
                "nov_header__gps_week_milliseconds": 201757000,
                "nov_header__receiver_status": 33554432,
                "nov_header__idle_time": 27.0,
                "latitude": 56.0,
                "longitude": 30.0,
                "height": 20.0,
                "north_velocity": 5.0,
                "east_velocity": 0.0,
                "up_velocity": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "azimuth": 0.0,
                "pos_type__type": 56,
            },
            {
                "_id": "row_1",
                "_bag_id": "bag",
                "_timestamp": (base_seconds + 1) * 1_000_000_000,
                "_topic": "/gnss/novatel/oem7/inspvax",
                "_type": "novatel_oem7_msgs/msg/INSPVAX",
                "header__stamp__sec": base_seconds + 1,
                "header__stamp__nanosec": 0,
                "header__frame_id": "gps",
                "nov_header__message_name": "INSPVAX",
                "nov_header__message_id": 1465,
                "nov_header__message_type": 0,
                "nov_header__sequence_number": 1,
                "nov_header__time_status": 180,
                "nov_header__gps_week_number": 2288,
                "nov_header__gps_week_milliseconds": 201758000,
                "nov_header__receiver_status": 33554432,
                "nov_header__idle_time": 27.0,
                "latitude": 56.0,
                "longitude": 30.0,
                "height": 20.0,
                "north_velocity": 7.5,
                "east_velocity": 0.0,
                "up_velocity": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "azimuth": 0.0,
                "pos_type__type": 56,
            },
        ]
    )

    result = compute_derived_metrics(df)

    assert result["_timestamp"].tolist() == [float(base_seconds), float(base_seconds + 1)]
    assert result["datetime_moscow"].dt.year.tolist() == [2023, 2023]
    assert result["acceleration"].tolist() == [0.0, 2.5]
