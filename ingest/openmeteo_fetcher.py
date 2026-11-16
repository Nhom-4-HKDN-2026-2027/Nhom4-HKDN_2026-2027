"""Open-Meteo Archive API hourly weather fetcher for Vietnam PM2.5 station coordinates."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from ingest.openaq_fetcher import _get_with_retry

logger = logging.getLogger(__name__)

OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_WEATHER_MODEL = "era5"
EXPECTED_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "wind_speed_10m": "m/s",
    "wind_direction_10m": "°",
    "precipitation": "mm",
}
HOURLY_VARIABLES_LIST: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]
HOURLY_VARIABLES = ",".join(HOURLY_VARIABLES_LIST)

WEATHER_COLS: list[str] = [
    "city",
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "event_time",
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_m_s",
    "wind_direction_deg",
    "precipitation_mm",
]


def _expected_utc_hourly_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Build the expected inclusive hourly UTC DatetimeIndex from start_date 00:00 to end_date 23:00."""
    start_ts = pd.Timestamp(start_date, tz="UTC").floor("D")
    end_ts = pd.Timestamp(end_date, tz="UTC").floor("D") + pd.Timedelta(hours=23)
    if end_ts < start_ts:
        raise ValueError(f"end_date ({end_date}) must not precede start_date ({start_date})")
    return pd.date_range(start=start_ts, end=end_ts, freq="1h", tz="UTC")


def fetch_station_hourly_weather(
    station_id: int,
    station_name: str,
    city: str,
    latitude: float,
    longitude: float,
    start_date: str = "2024-11-14",
    end_date: str = "2026-11-13",
    models: str | None = None,
    allow_nan: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch and validate hourly meteorological series for a station coordinate in UTC and m/s."""
    expected_idx = _expected_utc_hourly_index(start_date, end_date)
    expected_hours = len(expected_idx)
    model = models or DEFAULT_WEATHER_MODEL
    request_end = expected_idx[-1].floor("D")
    if model in {"era5", "era5_land"}:
        available_end = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=5)
        if request_end > available_end:
            logger.warning("%s archive has a 5-day delay; unavailable hours remain NaN", model)
            request_end = available_end
    if request_end < expected_idx[0]:
        if not allow_nan:
            raise ValueError("Requested window is entirely inside the historical archive delay")
        result = pd.DataFrame(
            {
                "city": city,
                "station_id": int(station_id),
                "station_name": str(station_name).strip(),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "event_time": expected_idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
                **{col: np.nan for col in WEATHER_COLS[6:]},
            },
            columns=WEATHER_COLS,
        )
        result.attrs.update(
            model=model,
            missing_values=len(expected_idx) * 5,
            fetch_status="unavailable_archive_delay",
        )
        return result

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": request_end.strftime("%Y-%m-%d"),
        "hourly": HOURLY_VARIABLES,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "models": model,
    }

    resp = _get_with_retry(
        url=OPENMETEO_ARCHIVE_URL,
        headers={"Accept": "application/json"},
        params=params,
        timeout=timeout,
    )
    payload = resp.json()
    units = payload.get("hourly_units", {})
    if any(units.get(var) != unit for var, unit in EXPECTED_UNITS.items()):
        raise ValueError("Open-Meteo hourly_units missing or incompatible with requested units")
    if payload.get("utc_offset_seconds", 0) != 0:
        raise ValueError("Open-Meteo returned a non-UTC offset")
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times and not allow_nan:
        raise ValueError(
            f"Open-Meteo returned empty hourly time series for station_id={station_id} "
            f"({start_date} -> {end_date})"
        )

    n_times = len(times)
    if n_times != expected_hours and not allow_nan:
        raise ValueError(
            f"Open-Meteo returned {n_times} hours, expected {expected_hours} hours "
            f"for station_id={station_id} ({start_date} -> {end_date})"
        )

    for var_name in HOURLY_VARIABLES_LIST:
        arr = hourly.get(var_name)
        if arr is None or len(arr) != n_times:
            raise ValueError(
                f"Open-Meteo hourly array '{var_name}' missing or length mismatch "
                f"(expected {n_times}) for station_id={station_id}"
            )

    parsed_ts = pd.to_datetime(times, utc=True, errors="coerce")
    if parsed_ts.isna().any():
        raise ValueError(f"Open-Meteo returned unparseable timestamps for station_id={station_id}")

    floored_ts = parsed_ts.floor("h")
    if (
        not parsed_ts.equals(floored_ts)
        or parsed_ts.has_duplicates
        or not parsed_ts.is_monotonic_increasing
        or not parsed_ts.isin(expected_idx).all()
        or (not allow_nan and not parsed_ts.equals(expected_idx))
    ):
        raise ValueError(
            f"Open-Meteo timestamps do not match expected hourly UTC grid "
            f"[{expected_idx[0]} .. {expected_idx[-1]}] for station_id={station_id}"
        )

    df_out = pd.DataFrame(
        {
            "city": city,
            "station_id": int(station_id),
            "station_name": str(station_name).strip(),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "event_time": floored_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temperature_c": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
            "relative_humidity_pct": pd.to_numeric(hourly["relative_humidity_2m"], errors="coerce"),
            "wind_speed_m_s": pd.to_numeric(hourly["wind_speed_10m"], errors="coerce"),
            "wind_direction_deg": pd.to_numeric(hourly["wind_direction_10m"], errors="coerce"),
            "precipitation_mm": pd.to_numeric(hourly["precipitation"], errors="coerce"),
        },
        columns=WEATHER_COLS,
    )

    wx_cols = WEATHER_COLS[6:]
    df_out[wx_cols] = df_out[wx_cols].replace([np.inf, -np.inf], np.nan)
    # Reindex only after rejecting duplicates, offsets and ordering defects.
    df_out.index = parsed_ts
    df_out = df_out.reindex(expected_idx)
    for col, value in {
        "city": city,
        "station_id": int(station_id),
        "station_name": str(station_name).strip(),
        "latitude": float(latitude),
        "longitude": float(longitude),
    }.items():
        df_out[col] = value
    df_out["event_time"] = expected_idx.strftime("%Y-%m-%dT%H:%M:%SZ")
    nan_total = int(df_out[wx_cols].isna().sum().sum())
    if nan_total:
        if not allow_nan:
            raise ValueError(
                f"Open-Meteo returned {nan_total} NaN values across meteorological columns "
                f"for station_id={station_id} (check historical archive delay)"
            )

        logger.warning("Station %d: %d missing weather values retained", station_id, nan_total)
    df_out = df_out.reset_index(drop=True)
    df_out.attrs.update(model=model, missing_values=nan_total, units=units)
    return df_out


def fetch_usable_stations_weather_2yr(
    stations_meta: list[dict[str, Any]],
    start_date: str = "2024-11-14",
    end_date: str = "2026-11-13",
    models: str | None = None,
    allow_nan: bool = False,
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    """Fetch 2-year hourly meteorological data for all unique usable stations."""
    if not stations_meta:
        return pd.DataFrame(columns=WEATHER_COLS)

    frames: list[pd.DataFrame] = []
    seen_station_ids: set[int] = set()

    for st in stations_meta:
        st_id = int(st["station_id"])
        if st_id in seen_station_ids:
            continue
        seen_station_ids.add(st_id)

        logger.info("Fetching 2-year Open-Meteo weather for station %d...", st_id)
        df_st = fetch_station_hourly_weather(
            station_id=st_id,
            station_name=str(st["station_name"]),
            city=str(st["city"]),
            latitude=float(st["latitude"]),
            longitude=float(st["longitude"]),
            start_date=start_date,
            end_date=end_date,
            models=models,
            allow_nan=allow_nan,
        )
        frames.append(df_st)
        time.sleep(sleep_sec)

    if not frames:
        return pd.DataFrame(columns=WEATHER_COLS)
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["station_id", "event_time"], keep="first")
        .reset_index(drop=True)
    )
