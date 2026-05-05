"""Pytest fixtures for synthetic navigation data."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest


BASE_TIMESTAMP = 1_700_053_200
ROWS_PER_BAG = 100


@pytest.fixture()
def nav_df() -> pd.DataFrame:
    """Create a 200-row dataframe with derived fields matching DataStore."""
    rows = []
    for bag_index, bag_id in enumerate(("bag_a", "bag_b")):
        for i in range(ROWS_PER_BAG):
            timestamp = BASE_TIMESTAMP + bag_index * 40_000 + i * 60
            if bag_id == "bag_a" and 42 <= i <= 46:
                speed = 30.0 - (i - 41) * 3.0
            elif bag_id == "bag_b" and 15 <= i <= 17:
                speed = 28.0 - (i - 14) * 2.5
            else:
                speed = 5.0 + bag_index * 2.0 + i * 0.1

            if bag_id == "bag_b" and i == 0:
                speed = 8.0

            in_m11 = i < 70
            rows.append(
                {
                    "_id": f"{bag_id}_{i}",
                    "_bag_id": bag_id,
                    "_timestamp": timestamp,
                    "_topic": "/novatel/odom",
                    "_type": "novatel_msgs/msg/Inspva",
                    "header__stamp__sec": timestamp,
                    "header__stamp__nanosec": 0,
                    "header__frame_id": "map",
                    "nov_header__message_name": "INSPVA",
                    "nov_header__message_id": 508,
                    "nov_header__message_type": 0,
                    "nov_header__sequence_number": i,
                    "nov_header__time_status": 20,
                    "nov_header__gps_week_number": 2200,
                    "nov_header__gps_week_milliseconds": i * 1000,
                    "nov_header__receiver_status": 0,
                    "nov_header__idle_time": 0,
                    "latitude": 56.0 + i * 0.001 if in_m11 else 61.0 + i * 0.001,
                    "longitude": 31.0 + i * 0.001 if in_m11 else 40.0 + i * 0.001,
                    "height": 180.0,
                    "north_velocity": speed,
                    "east_velocity": 0.0,
                    "up_velocity": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "azimuth": float((i * 3) % 360),
                    "pos_type__type": 19 if i < 25 else 56,
                }
            )

    df = pd.DataFrame(rows).sort_values(["_bag_id", "_timestamp"]).reset_index(drop=True)
    df["horizontal_speed"] = np.sqrt(df["north_velocity"] ** 2 + df["east_velocity"] ** 2)
    df = pd.concat(
        [_compute_acceleration(group) for _, group in df.groupby("_bag_id", sort=False)],
        ignore_index=True,
    )
    df["datetime_moscow"] = pd.to_datetime(df["_timestamp"], unit="s") + timedelta(hours=3)
    df["hour_moscow"] = df["datetime_moscow"].dt.hour
    return df.reset_index(drop=True)


def _compute_acceleration(group: pd.DataFrame) -> pd.DataFrame:
    working = group.copy().sort_values("_timestamp")
    dt = working["_timestamp"].diff().fillna(1)
    dv = working["horizontal_speed"].diff().fillna(0)
    working["acceleration"] = (dv / dt).fillna(0.0)
    return working
