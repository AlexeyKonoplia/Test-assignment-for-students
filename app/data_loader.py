from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.domain import MOSCOW_UTC_OFFSET_HOURS


REQUIRED_COLUMNS = {
    "_id",
    "_bag_id",
    "_timestamp",
    "_topic",
    "_type",
    "header__stamp__sec",
    "header__stamp__nanosec",
    "header__frame_id",
    "nov_header__message_name",
    "nov_header__message_id",
    "nov_header__message_type",
    "nov_header__sequence_number",
    "nov_header__time_status",
    "nov_header__gps_week_number",
    "nov_header__gps_week_milliseconds",
    "nov_header__receiver_status",
    "nov_header__idle_time",
    "latitude",
    "longitude",
    "height",
    "north_velocity",
    "east_velocity",
    "up_velocity",
    "roll",
    "pitch",
    "azimuth",
    "pos_type__type",
}


class NavigationDataStore:
    """Instance-based storage for navigation records loaded at application startup."""

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None

    def load(self, data_dir: str) -> None:
        """Load all CSV files from *data_dir* and compute derived metrics once."""
        path = Path(data_dir)
        if not path.exists() or not path.is_dir():
            raise RuntimeError(f"Data directory does not exist or is not a directory: {data_dir}")

        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"Data directory is empty or contains no CSV files: {data_dir}")

        frames: list[pd.DataFrame] = []
        for csv_file in csv_files:
            try:
                frame = pd.read_csv(csv_file)
            except pd.errors.ParserError as exc:
                raise RuntimeError(f"Failed to parse CSV file {csv_file}: {exc}") from exc

            self._validate_schema(frame, str(csv_file))
            frames.append(frame)

        df = pd.concat(frames, ignore_index=True)
        self._validate_schema(df, data_dir)
        self._df = compute_derived_metrics(df)

    def get(self) -> pd.DataFrame:
        """Return the loaded navigation dataframe."""
        if self._df is None:
            raise RuntimeError("DataStore has not been loaded. Call DataStore.load() at startup.")
        return self._df

    @staticmethod
    def _validate_schema(df: pd.DataFrame, source: str) -> None:
        missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f"CSV schema validation failed for {source}. Missing columns: {missing}")


DataStore = NavigationDataStore()


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with speed, acceleration, and Moscow time fields."""
    working = df.copy()
    timestamp_seconds, datetime_utc = _normalize_timestamps(working)
    working["_timestamp"] = timestamp_seconds
    working["_datetime_utc"] = datetime_utc
    working["horizontal_speed"] = np.sqrt(
        working["north_velocity"] ** 2 + working["east_velocity"] ** 2
    )
    working = working.sort_values(["_bag_id", "_timestamp"]).reset_index(drop=True)
    working = pd.concat(
        [_compute_acceleration(group) for _, group in working.groupby("_bag_id", sort=False)],
        ignore_index=True,
    )
    working["datetime_moscow"] = working.pop("_datetime_utc") + timedelta(
        hours=MOSCOW_UTC_OFFSET_HOURS
    )
    working["hour_moscow"] = working["datetime_moscow"].dt.hour
    return working.reset_index(drop=True)


def _normalize_timestamps(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Convert supported epoch timestamp units to seconds and UTC datetimes."""
    numeric = pd.to_numeric(df["_timestamp"], errors="coerce")
    if numeric.isna().any():
        raise ValueError("_timestamp contains non-numeric values")

    median_abs = numeric.abs().median()
    if median_abs >= 1e17:
        unit = "ns"
        scale = 1_000_000_000.0
    elif median_abs >= 1e14:
        unit = "us"
        scale = 1_000_000.0
    elif median_abs >= 1e11:
        unit = "ms"
        scale = 1_000.0
    else:
        unit = "s"
        scale = 1.0

    datetime_utc = pd.to_datetime(numeric, unit=unit, errors="coerce")
    if datetime_utc.isna().any() and {"header__stamp__sec", "header__stamp__nanosec"}.issubset(df.columns):
        header_seconds = pd.to_numeric(df["header__stamp__sec"], errors="coerce")
        header_nanoseconds = pd.to_numeric(df["header__stamp__nanosec"], errors="coerce").fillna(0)
        fallback = pd.to_datetime(header_seconds, unit="s", errors="coerce") + pd.to_timedelta(
            header_nanoseconds, unit="ns"
        )
        datetime_utc = datetime_utc.fillna(fallback)

    if datetime_utc.isna().any():
        bad_count = int(datetime_utc.isna().sum())
        raise ValueError(f"_timestamp contains {bad_count} out-of-bounds timestamp value(s)")

    return numeric / scale, datetime_utc


def _compute_acceleration(group: pd.DataFrame) -> pd.DataFrame:
    """Compute speed delta over timestamp delta inside one bag."""
    working = group.copy().sort_values("_timestamp")
    dt = working["_timestamp"].diff().fillna(1)
    dt = dt.replace(0, np.nan)
    dv = working["horizontal_speed"].diff().fillna(0)
    working["acceleration"] = (dv / dt).fillna(0.0)
    return working
