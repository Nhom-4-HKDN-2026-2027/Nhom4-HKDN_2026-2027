"""Adapt source-specific samples to the shared in-memory feature contract."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
FEATURE_COLUMNS = [
    "station_id",
    "event_time",
    "pm25_ug_m3",
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_m_s",
    "wind_direction_deg",
    "precipitation_mm",
]


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"station_id", "event_time"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing key columns: {sorted(missing)}")
    result = frame.copy()
    result["station_id"] = result["station_id"].astype("string").str.strip()
    if result.station_id.isna().any() or result.station_id.eq("").any():
        raise ValueError("station_id must be a nonempty string")
    # Do not silently interpret an unqualified local timestamp as UTC.
    for value in result.event_time:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("event_time must contain timezone-aware timestamps")
    result["event_time"] = pd.to_datetime(result.event_time, utc=True, errors="raise")
    if not result.event_time.equals(result.event_time.dt.floor("h")):
        raise ValueError("event_time must align to whole hours")
    if result.duplicated(["station_id", "event_time"]).any():
        raise ValueError("Duplicate station_id/event_time keys")
    return result


def normalize_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return eight contract columns with string IDs, UTC timestamps and float values.

    Missing numeric values remain NaN. This is a schema adapter, not an outlier
    filter or imputer. Extra source metadata is omitted from the feature view.
    """
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    result = _normalize_keys(frame[FEATURE_COLUMNS])
    for col in FEATURE_COLUMNS[2:]:
        result[col] = pd.to_numeric(result[col], errors="raise").astype(float)
        if np.isinf(result[col]).any():
            raise ValueError(f"Nonfinite values in {col}; review source quality")
    return result.sort_values(["station_id", "event_time"]).reset_index(drop=True)


def build_vietnam_features(pm25: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Left-join Vietnam PM2.5/weather using normalized keys, preserving PM2.5 rows."""
    required_weather = ["station_id", "event_time", *FEATURE_COLUMNS[3:]]
    for label, frame, required in [
        ("PM2.5", pm25, FEATURE_COLUMNS[:3]),
        ("weather", weather, required_weather),
    ]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
    pm = _normalize_keys(pm25[FEATURE_COLUMNS[:3]])
    wx = _normalize_keys(weather[required_weather])
    joined = pm.merge(
        wx, on=["station_id", "event_time"], how="left", validate="one_to_one", indicator=True
    )
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        logger.warning("%d PM2.5 rows have no matching weather row; values remain NaN", unmatched)
    return normalize_feature_table(joined)
