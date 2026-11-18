"""OpenAQ API v3 data fetcher and 2-year completeness/gap calculator for Hanoi and HCMC."""

from __future__ import annotations

import logging
import time
from email.utils import parsedate_to_datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
VIETNAM_COUNTRY_ISO = "VN"
VIETNAM_COUNTRY_ID = 56
PM25_PARAMETER_ID = 2

CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "Hanoi": (105.2, 20.5, 106.1, 21.5),
    "HCMC": (106.3, 10.3, 107.1, 11.2),
}

LIFECYCLE_CATEGORIES: list[str] = [
    "not_fetched (activity unknown)",
    "zero_valid_hours (records present)",
    "inactive (no valid observations since cutoff)",
    "active (valid observations since cutoff)",
    "no_data_in_window (fetch completed)",
    "fetch_failed (activity unknown)",
]

STATIONS_METADATA_COLS: list[str] = [
    "city",
    "location_id",
    "name",
    "locality",
    "provider",
    "owner",
    "is_monitor",
    "is_mobile",
    "sensor_id",
    "lat",
    "lon",
    "datetime_first",
    "datetime_last",
]

PM25_HOURLY_COLS: list[str] = [
    "city",
    "station_id",
    "station_name",
    "sensor_id",
    "provider",
    "is_monitor",
    "latitude",
    "longitude",
    "event_time",
    "pm25_ug_m3",
    "percent_coverage",
]

STATION_SUMMARY_COLS: list[str] = [
    "city",
    "station_id",
    "sensor_id",
    "station_name",
    "is_monitor",
    "alive_nov2026",
    "usable_demo",
    "api_rows",
    "recorded_hours",
    "unrecorded_2yr_h",
    "nan_rows",
    "neg_rows",
    "valid_hours",
    "completeness_2yr_pct",
    "missing_2yr_pct",
    "active_span_h",
    "missing_active_h",
    "missing_in_active_pct",
    "num_gaps",
    "num_gaps_ge_24h",
    "max_gap_hours",
    "leading_gap_h",
    "trailing_gap_h",
    "gt500_count",
    "mean_pm25_raw",
    "mean_pm25_le500",
    "max_pm25",
    "first_valid_utc",
    "last_valid_utc",
]

GAP_COLS: list[str] = [
    "city",
    "station_id",
    "station_name",
    "gap_start_utc",
    "gap_end_utc",
    "gap_hours",
    "gap_days",
]


def _get_with_retry(
    url: str,
    headers: dict[str, str],
    params: dict[str, Any],
    timeout: int = 30,
    max_retries: int = 5,
) -> requests.Response:
    """Execute HTTP GET with bounded retry and exponential backoff on 429 and 5xx errors."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            backoff = 2**attempt
            logger.warning(
                "Request error on %s (%s, attempt %d/%d). Retrying in %ds...",
                url,
                type(exc).__name__,
                attempt,
                max_retries,
                backoff,
            )
            time.sleep(backoff)
            continue

        if resp.status_code == 429:
            if attempt == max_retries:
                resp.raise_for_status()
            raw_reset = resp.headers.get("Retry-After")
            if raw_reset is None and url.startswith(OPENAQ_BASE_URL + "/"):
                # OpenAQ's documented example and SDK define this as seconds until reset.
                raw_reset = resp.headers.get("x-ratelimit-reset")
            try:
                wait_sec = max(1.0, float(raw_reset or 60))
            except ValueError:
                try:
                    wait_sec = max(1.0, parsedate_to_datetime(raw_reset).timestamp() - time.time())
                except (TypeError, ValueError, OverflowError):
                    wait_sec = 60
            logger.warning(
                "Rate limited (HTTP 429) on %s (attempt %d/%d). Sleeping %ds...",
                url,
                attempt,
                max_retries,
                wait_sec,
            )
            time.sleep(wait_sec)
            continue

        if resp.status_code in {500, 502, 503, 504} and attempt < max_retries:
            backoff = 2**attempt
            logger.warning(
                "Server error HTTP %d on %s (attempt %d/%d). Retrying in %ds...",
                resp.status_code,
                url,
                attempt,
                max_retries,
                backoff,
            )
            time.sleep(backoff)
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError(f"Exhausted {max_retries} retries for {url}") from last_exc


def _infer_city(lat: float | None, lon: float | None, locality: str | None) -> str | None:
    """Assign a station to Hanoi or HCMC based on coordinates or locality string."""
    if lat is not None and lon is not None and not (pd.isna(lat) or pd.isna(lon)):
        for city, (min_lon, min_lat, max_lon, max_lat) in CITY_BBOXES.items():
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                return city
    loc_text = (locality or "").lower()
    if "hanoi" in loc_text or "hà nội" in loc_text:
        return "Hanoi"
    if "ho chi minh" in loc_text or "hồ chí minh" in loc_text or "hcmc" in loc_text:
        return "HCMC"
    return None


def _select_primary_pm25_sensor(
    pm25_sensors: list[dict[str, Any]],
    location_id: int | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> int:
    """Select an inventory default by recency, NOT by window completeness.

    The full-window fetch compares candidates using actual valid hourly observations.
    """
    if not pm25_sensors:
        raise ValueError("pm25_sensors list cannot be empty")
    if len(pm25_sensors) == 1:
        return int(pm25_sensors[0]["id"])

    candidate_sensors = pm25_sensors
    has_detailed_meta = any(
        (s.get("datetimeLast") or {}).get("utc") is not None
        or (s.get("coverage") or {}).get("observedCount") is not None
        for s in pm25_sensors
    )

    if not has_detailed_meta and location_id is not None and headers is not None:
        try:
            resp = _get_with_retry(
                url=f"{OPENAQ_BASE_URL}/locations/{location_id}/sensors",
                headers=headers,
                params={},
                timeout=timeout,
            )
            detailed = [
                s
                for s in resp.json().get("results", [])
                if (s.get("parameter") or {}).get("id") == PM25_PARAMETER_ID
                or (s.get("parameter") or {}).get("name") == "pm25"
            ]
            if detailed:
                candidate_sensors = detailed
            time.sleep(0.95)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Could not fetch detailed sensor metadata for location %s: %s",
                location_id,
                type(exc).__name__,
            )

    def _sort_key(s: dict[str, Any]) -> tuple[pd.Timestamp, int, int]:
        dt_last_str = (s.get("datetimeLast") or {}).get("utc")
        dt_last = pd.to_datetime(dt_last_str, utc=True, errors="coerce")
        if pd.isna(dt_last):
            dt_last = pd.Timestamp.min.tz_localize("UTC")
        raw_count = pd.to_numeric((s.get("coverage") or {}).get("observedCount"), errors="coerce")
        obs_count = int(raw_count) if pd.notna(raw_count) and np.isfinite(raw_count) else 0
        return (dt_last, obs_count, int(s.get("id", 0)))

    sorted_sensors = sorted(candidate_sensors, key=_sort_key, reverse=True)
    return int(sorted_sensors[0]["id"])


def fetch_vietnam_pm25_stations(api_key: str, timeout: int = 30) -> pd.DataFrame:
    """Fetch all PM2.5 monitoring locations in Hanoi and HCMC from OpenAQ API v3."""
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    rows: list[dict[str, Any]] = []
    sensor_candidates: dict[int, list[int]] = {}
    seen_location_ids: set[int] = set()
    page = 1

    while True:
        resp = _get_with_retry(
            url=f"{OPENAQ_BASE_URL}/locations",
            headers=headers,
            params={
                "iso": VIETNAM_COUNTRY_ISO,
                "parameters_id": PM25_PARAMETER_ID,
                "limit": 1000,
                "page": page,
            },
            timeout=timeout,
        )
        results = resp.json().get("results")
        if not isinstance(results, list):
            raise ValueError("OpenAQ inventory response missing results array")
        if not results:
            break

        for loc in results:
            loc_id = int(loc["id"])
            if loc_id in seen_location_ids:
                continue

            coords = loc.get("coordinates") or {}
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            city = _infer_city(lat, lon, loc.get("locality"))
            if city is None:
                continue

            pm25_sensors = [
                s
                for s in (loc.get("sensors") or [])
                if (s.get("parameter") or {}).get("id") == PM25_PARAMETER_ID
                or (s.get("parameter") or {}).get("name") == "pm25"
            ]
            if not pm25_sensors:
                continue

            seen_location_ids.add(loc_id)
            sensor_candidates[loc_id] = sorted({int(s["id"]) for s in pm25_sensors})
            sensor_id = _select_primary_pm25_sensor(
                pm25_sensors,
                timeout=timeout,
            )
            dt_first = (loc.get("datetimeFirst") or {}).get("utc")
            dt_last = (loc.get("datetimeLast") or {}).get("utc")

            rows.append(
                {
                    "city": city,
                    "location_id": loc_id,
                    "name": str(loc.get("name") or "").strip(),
                    "locality": loc.get("locality"),
                    "provider": (loc.get("provider") or {}).get("name"),
                    "owner": (loc.get("owner") or {}).get("name"),
                    "is_monitor": bool(loc.get("isMonitor", False)),
                    "is_mobile": bool(loc.get("isMobile", False)),
                    "sensor_id": sensor_id,
                    "lat": float(lat) if lat is not None else np.nan,
                    "lon": float(lon) if lon is not None else np.nan,
                    "datetime_first": dt_first,
                    "datetime_last": dt_last,
                }
            )

        if len(results) < 1000:
            break
        page += 1
        time.sleep(0.95)

    if not rows:
        return pd.DataFrame(columns=STATIONS_METADATA_COLS)
    result = (
        pd.DataFrame(rows, columns=STATIONS_METADATA_COLS)
        .sort_values(["city", "location_id"])
        .reset_index(drop=True)
    )
    result.attrs["sensor_candidates"] = sensor_candidates
    return result


def fetch_sensor_hourly_pm25(
    sensor_id: int,
    api_key: str,
    datetime_from: str = "2024-11-14T00:00:00Z",
    datetime_to: str = "2026-11-14T00:00:00Z",
    sleep_sec: float = 0.95,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Fetch paginated hourly PM2.5 records for a single sensor within [datetime_from, datetime_to)."""
    sensor_id = int(sensor_id)
    headers = {"X-API-Key": str(api_key), "Accept": "application/json"}
    w_start, w_end = _validate_window(datetime_from, datetime_to)
    records: list[dict[str, Any]] = []
    page = 1

    while True:
        resp = _get_with_retry(
            url=f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours",
            headers=headers,
            params={
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": 1000,
                "page": page,
            },
            timeout=timeout,
        )
        results = resp.json().get("results")
        if not isinstance(results, list):
            raise ValueError("OpenAQ hourly response missing results array")
        if not results:
            break

        for item in results:
            period = item.get("period") or {}
            dt_from = (period.get("datetimeFrom") or {}).get("utc")
            if not dt_from:
                raise ValueError("OpenAQ hourly record missing period.datetimeFrom.utc")
            ts = pd.to_datetime(dt_from, utc=True, errors="coerce")
            if pd.isna(ts):
                raise ValueError("OpenAQ hourly record contains invalid timestamp")
            if ts < w_start or ts >= w_end:
                continue
            parameter = item.get("parameter") or {}
            if parameter and (
                parameter.get("id") != PM25_PARAMETER_ID
                or parameter.get("units") not in {"µg/m³", "μg/m³", "ug/m3"}
            ):
                raise ValueError("OpenAQ hourly parameter/units incompatible with PM2.5 µg/m³")
            coverage = item.get("coverage") or {}
            records.append(
                {
                    "event_time": ts.floor("h").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "pm25_ug_m3": item.get("value"),
                    "percent_coverage": coverage.get("percentComplete"),
                }
            )

        if len(results) < 1000:
            break
        page += 1
        time.sleep(sleep_sec)

    return records


def fetch_all_active_stations_2yr(
    stations_df: pd.DataFrame,
    api_key: str,
    window_start_utc: str = "2024-11-14T00:00:00Z",
    window_end_utc: str = "2026-11-14T00:00:00Z",
) -> pd.DataFrame:
    """Fetch 2-year hourly PM2.5 series for all stations active during the evaluation window."""
    if stations_df.empty:
        return pd.DataFrame(columns=PM25_HOURLY_COLS)

    w_start, w_end = _validate_window(window_start_utc, window_end_utc)

    all_rows: list[dict[str, Any]] = []
    statuses: dict[int, str] = {}
    selected: dict[int, int] = {}
    scores: dict[int, dict[int, int]] = {}
    for _, st in stations_df.iterrows():
        loc_id = int(st["location_id"])
        dt_last = pd.to_datetime(st.get("datetime_last"), utc=True, errors="coerce")
        if pd.notna(dt_last) and dt_last < w_start:
            statuses[loc_id] = "inactive"
            continue

        candidates = stations_df.attrs.get("sensor_candidates", {}).get(loc_id)
        if candidates is None:
            candidates = [] if pd.isna(st.get("sensor_id")) else [int(st["sensor_id"])]
        if not candidates:
            statuses[loc_id] = "not_fetched"
            continue
        fetched: dict[int, list[dict[str, Any]]] = {}
        scores[loc_id] = {}
        try:
            for candidate in sorted(set(candidates)):
                logger.info("Fetching station %d, sensor %d", loc_id, candidate)
                items = fetch_sensor_hourly_pm25(
                    sensor_id=candidate,
                    api_key=api_key,
                    datetime_from=window_start_utc,
                    datetime_to=window_end_utc,
                )
                fetched[candidate] = items
                frame = pd.DataFrame(items, columns=["event_time", "pm25_ug_m3"])
                ts = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
                values = pd.to_numeric(frame["pm25_ug_m3"], errors="coerce")
                valid = np.isfinite(values) & (values >= 0) & (ts >= w_start) & (ts < w_end)
                scores[loc_id][candidate] = int(ts[valid].dt.floor("h").nunique())
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            # Incomplete candidate comparisons must not claim a best sensor.
            statuses[loc_id] = "fetch_failed"
            logger.warning("Station %d fetch failed: %s", loc_id, type(exc).__name__)
            continue
        sensor_id = max(scores[loc_id], key=lambda sid: (scores[loc_id][sid], -sid))
        selected[loc_id] = sensor_id
        hourly_items = fetched[sensor_id]
        statuses[loc_id] = "fetched" if hourly_items else "no_data_in_window"
        for rec in hourly_items:
            all_rows.append(
                {
                    "city": st["city"],
                    "station_id": loc_id,
                    "station_name": str(st["name"]).strip(),
                    "sensor_id": sensor_id,
                    "provider": st.get("provider"),
                    "is_monitor": bool(st["is_monitor"]),
                    "latitude": float(st["lat"]),
                    "longitude": float(st["lon"]),
                    "event_time": rec["event_time"],
                    "pm25_ug_m3": rec["pm25_ug_m3"],
                    "percent_coverage": rec["percent_coverage"],
                }
            )

    result = pd.DataFrame(all_rows, columns=PM25_HOURLY_COLS)
    result.attrs.update(fetch_status=statuses, selected_sensors=selected, sensor_scores=scores)
    return result


def classify_station_lifecycle(
    stations_df: pd.DataFrame,
    pm25_df: pd.DataFrame,
    window_start_utc: str = "2024-11-14T00:00:00Z",
    window_end_utc: str = "2026-11-14T00:00:00Z",
    alive_cutoff_utc: str = "2026-11-01T00:00:00Z",
) -> pd.DataFrame:
    """Classify observed activity, preserving unknown/failed fetches separately.

    Inactive means no recent valid observations, not proof of physical closure.
    Fetch provenance is carried in pm25_df.attrs and persisted by the pipeline manifest.
    """
    w_start = pd.Timestamp(window_start_utc)
    w_end = pd.Timestamp(window_end_utc)
    cutoff_ts = pd.Timestamp(alive_cutoff_utc)

    # Normalize station catalog from stations_df and union any station_ids only present in pm25_df
    catalog_frames: list[pd.DataFrame] = []
    if not stations_df.empty:
        st_col = "location_id" if "location_id" in stations_df.columns else "station_id"
        sub_st = stations_df[["city", st_col]].rename(columns={st_col: "station_id"}).copy()
        sub_st["station_id"] = sub_st["station_id"].astype(int)
        catalog_frames.append(sub_st)

    if not pm25_df.empty and {"city", "station_id"}.issubset(pm25_df.columns):
        sub_pm = pm25_df[["city", "station_id"]].copy()
        sub_pm["station_id"] = sub_pm["station_id"].astype(int)
        catalog_frames.append(sub_pm)

    if not catalog_frames:
        return pd.DataFrame(
            {
                "lifecycle_status": LIFECYCLE_CATEGORIES,
                "HCMC": [0] * len(LIFECYCLE_CATEGORIES),
                "Hanoi": [0] * len(LIFECYCLE_CATEGORIES),
                "Total": [0] * len(LIFECYCLE_CATEGORIES),
            }
        )

    st_catalog = (
        pd.concat(catalog_frames, ignore_index=True)
        .drop_duplicates(subset=["station_id"], keep="first")
        .reset_index(drop=True)
    )

    if pm25_df.empty:
        stations_with_api_rows: set[int] = set()
        stations_with_valid_rows: set[int] = set()
        stations_alive_end: set[int] = set()
    else:
        df_raw = pm25_df.copy()
        df_raw["event_time_dt"] = pd.to_datetime(
            df_raw["event_time"], utc=True, errors="coerce"
        ).dt.floor("h")
        df_raw = df_raw[
            df_raw["event_time_dt"].notna()
            & (df_raw["event_time_dt"] >= w_start)
            & (df_raw["event_time_dt"] < w_end)
        ]
        pm_vals = pd.to_numeric(df_raw["pm25_ug_m3"], errors="coerce")
        valid_raw = df_raw[np.isfinite(pm_vals) & (pm_vals >= 0)]

        stations_with_api_rows = set(df_raw["station_id"].astype(int).unique())
        stations_with_valid_rows = set(valid_raw["station_id"].astype(int).unique())
        last_valid_per_st = valid_raw.groupby(valid_raw["station_id"].astype(int))[
            "event_time_dt"
        ].max()
        stations_alive_end = set(last_valid_per_st[last_valid_per_st >= cutoff_ts].index)

    def _category(loc_id: int) -> str:
        status = pm25_df.attrs.get("fetch_status", {}).get(loc_id)
        if status == "fetch_failed":
            return LIFECYCLE_CATEGORIES[5]
        if loc_id not in stations_with_api_rows:
            if status == "no_data_in_window":
                return LIFECYCLE_CATEGORIES[4]
            if status == "inactive":
                return LIFECYCLE_CATEGORIES[2]
            return LIFECYCLE_CATEGORIES[0]
        if loc_id not in stations_with_valid_rows:
            return LIFECYCLE_CATEGORIES[1]
        if loc_id not in stations_alive_end:
            return LIFECYCLE_CATEGORIES[2]
        return LIFECYCLE_CATEGORIES[3]

    st_catalog["lifecycle_status"] = pd.Categorical(
        st_catalog["station_id"].astype(int).apply(_category),
        categories=LIFECYCLE_CATEGORIES,
        ordered=True,
    )
    summary = (
        st_catalog.groupby(["lifecycle_status", "city"], observed=False)["station_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(LIFECYCLE_CATEGORIES, fill_value=0)
        .reset_index()
    )
    for col in ["HCMC", "Hanoi"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["Total"] = summary["HCMC"] + summary["Hanoi"]
    return summary[["lifecycle_status", "HCMC", "Hanoi", "Total"]]


def _validate_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    bounds = [pd.Timestamp(start), pd.Timestamp(end)]
    if any(t.tzinfo is None or t != t.floor("h") for t in bounds):
        raise ValueError("Window bounds must be timezone-aware whole hours")
    if bounds[1] <= bounds[0]:
        raise ValueError("window_end_utc must be strictly greater than window_start_utc")
    return bounds[0].tz_convert("UTC"), bounds[1].tz_convert("UTC")


def compute_completeness_and_gaps(
    pm25_df: pd.DataFrame,
    window_start_utc: str = "2024-11-14T00:00:00Z",
    window_end_utc: str = "2026-11-14T00:00:00Z",
    alive_cutoff_utc: str = "2026-11-01T00:00:00Z",
    max_missing_2yr_pct: float = 60.0,
    max_gt500_count: int = 0,
    include_boundary_gaps: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-station 2-year completeness metrics and continuous outage gaps from raw data."""
    w_start, w_end = _validate_window(window_start_utc, window_end_utc)

    total_window_hours = int((w_end - w_start).total_seconds() // 3600)
    w_last_hour = w_end - pd.Timedelta(hours=1)
    cutoff_ts = pd.Timestamp(alive_cutoff_utc)

    if pm25_df.empty:
        return (
            pd.DataFrame(columns=STATION_SUMMARY_COLS),
            pd.DataFrame(columns=GAP_COLS),
        )

    df = pm25_df.copy()
    df["station_name"] = df["station_name"].astype(str).str.strip()
    df["pm25_ug_m3"] = pd.to_numeric(df["pm25_ug_m3"], errors="coerce")
    df["event_time_dt"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce").dt.floor("h")

    # Strictly filter records to the half-open evaluation window [w_start, w_end)
    df = df[
        df["event_time_dt"].notna()
        & (df["event_time_dt"] >= w_start)
        & (df["event_time_dt"] < w_end)
    ].copy()

    if df.empty:
        return (
            pd.DataFrame(columns=STATION_SUMMARY_COLS),
            pd.DataFrame(columns=GAP_COLS),
        )

    # Mark valid measurements so duplicate timestamps prioritize valid values over NaN/negative
    df["_is_valid"] = np.isfinite(df["pm25_ug_m3"]) & (df["pm25_ug_m3"] >= 0)

    station_rows: list[dict[str, Any]] = []
    all_gaps: list[dict[str, Any]] = []

    for (city, st_id), grp in df.groupby(["city", "station_id"], dropna=False):
        if "sensor_id" in grp and grp["sensor_id"].nunique() > 1:
            raise ValueError(
                f"Multiple sensors for station {st_id}; select one before computing DQ"
            )
        api_rows = len(grp)
        grp_sorted = grp.sort_values(
            ["event_time_dt", "_is_valid"], ascending=[True, False], kind="stable"
        )
        valid_subset = grp_sorted[grp_sorted["_is_valid"]]
        ref_row = valid_subset.iloc[-1] if not valid_subset.empty else grp_sorted.iloc[-1]

        st_name = str(ref_row["station_name"])
        sensor_id = int(ref_row["sensor_id"]) if "sensor_id" in ref_row else int(st_id)
        is_monitor = bool(ref_row["is_monitor"]) if "is_monitor" in ref_row else False

        grp_unique = grp_sorted.drop_duplicates(subset=["event_time_dt"], keep="first").sort_values(
            "event_time_dt"
        )
        recorded_hours = len(grp_unique)
        unrecorded_2yr_hours = total_window_hours - recorded_hours
        nan_rows = int(grp_unique["pm25_ug_m3"].isna().sum())
        neg_rows = int((grp_unique["pm25_ug_m3"] < 0).sum())

        valid = grp_unique[grp_unique["_is_valid"]].copy()
        valid_hours = len(valid)
        gt500_count = int((valid["pm25_ug_m3"] > 500).sum())

        missing_2yr_hours = total_window_hours - valid_hours
        missing_2yr_pct = round(missing_2yr_hours / total_window_hours * 100, 2)
        completeness_2yr_pct = round(valid_hours / total_window_hours * 100, 2)

        station_gaps: list[dict[str, Any]] = []

        if valid_hours > 0:
            first_t = valid["event_time_dt"].min()
            last_t = valid["event_time_dt"].max()
            active_span_hours = int((last_t - first_t).total_seconds() // 3600) + 1
            missing_active_hours = active_span_hours - valid_hours
            missing_in_active_pct = round(missing_active_hours / active_span_hours * 100, 2)

            leading_gap_h = int((first_t - w_start).total_seconds() // 3600)
            trailing_gap_h = int((w_last_hour - last_t).total_seconds() // 3600)

            if include_boundary_gaps and leading_gap_h > 0:
                station_gaps.append(
                    {
                        "city": city,
                        "station_id": int(st_id),
                        "station_name": st_name,
                        "gap_start_utc": w_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "gap_end_utc": (first_t - pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "gap_hours": leading_gap_h,
                        "gap_days": round(float(leading_gap_h) / 24.0, 1),
                    }
                )

            valid["prev_valid_t"] = valid["event_time_dt"].shift(1)
            valid["diff_hours"] = (
                valid["event_time_dt"] - valid["prev_valid_t"]
            ).dt.total_seconds() // 3600
            gap_df = valid[valid["diff_hours"] > 1].copy()
            gap_df["gap_hours"] = gap_df["diff_hours"].astype(int) - 1

            for _, g_row in gap_df.iterrows():
                station_gaps.append(
                    {
                        "city": city,
                        "station_id": int(st_id),
                        "station_name": st_name,
                        "gap_start_utc": (g_row["prev_valid_t"] + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "gap_end_utc": (g_row["event_time_dt"] - pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "gap_hours": int(g_row["gap_hours"]),
                        "gap_days": round(float(g_row["gap_hours"]) / 24.0, 1),
                    }
                )

            if include_boundary_gaps and trailing_gap_h > 0:
                station_gaps.append(
                    {
                        "city": city,
                        "station_id": int(st_id),
                        "station_name": st_name,
                        "gap_start_utc": (last_t + pd.Timedelta(hours=1)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "gap_end_utc": w_last_hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "gap_hours": trailing_gap_h,
                        "gap_days": round(float(trailing_gap_h) / 24.0, 1),
                    }
                )

            mean_raw = round(float(valid["pm25_ug_m3"].mean()), 2)
            valid_clean = valid[valid["pm25_ug_m3"] <= 500]["pm25_ug_m3"]
            mean_le500 = round(float(valid_clean.mean()), 2) if len(valid_clean) > 0 else np.nan
            max_pm25 = round(float(valid["pm25_ug_m3"].max()), 2)
            alive_end = bool(last_t >= cutoff_ts)
        else:
            # Station has records in the window, but 0 valid hours (all NaN or < 0)
            first_t = pd.NaT
            last_t = pd.NaT
            active_span_hours = 0
            missing_active_hours = 0
            missing_in_active_pct = 100.0
            # Invariant: leading_gap_h + active_span_h + trailing_gap_h == total_window_hours
            leading_gap_h = total_window_hours
            trailing_gap_h = 0
            if include_boundary_gaps:
                station_gaps.append(
                    {
                        "city": city,
                        "station_id": int(st_id),
                        "station_name": st_name,
                        "gap_start_utc": w_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "gap_end_utc": w_last_hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "gap_hours": total_window_hours,
                        "gap_days": round(float(total_window_hours) / 24.0, 1),
                    }
                )
            mean_raw = np.nan
            mean_le500 = np.nan
            max_pm25 = np.nan
            alive_end = False

        num_gaps = len(station_gaps)
        num_gaps_ge_24h = sum(1 for g in station_gaps if g["gap_hours"] >= 24)
        max_gap_hours = max((g["gap_hours"] for g in station_gaps), default=0)
        all_gaps.extend(station_gaps)

        usable_demo = bool(
            alive_end
            and (missing_2yr_pct < max_missing_2yr_pct)
            and (gt500_count <= max_gt500_count)
        )

        station_rows.append(
            {
                "city": city,
                "station_id": int(st_id),
                "sensor_id": sensor_id,
                "station_name": st_name,
                "is_monitor": is_monitor,
                "alive_nov2026": alive_end,
                "usable_demo": usable_demo,
                "api_rows": api_rows,
                "recorded_hours": recorded_hours,
                "unrecorded_2yr_h": unrecorded_2yr_hours,
                "nan_rows": nan_rows,
                "neg_rows": neg_rows,
                "valid_hours": valid_hours,
                "completeness_2yr_pct": completeness_2yr_pct,
                "missing_2yr_pct": missing_2yr_pct,
                "active_span_h": active_span_hours,
                "missing_active_h": missing_active_hours,
                "missing_in_active_pct": missing_in_active_pct,
                "num_gaps": num_gaps,
                "num_gaps_ge_24h": num_gaps_ge_24h,
                "max_gap_hours": max_gap_hours,
                "leading_gap_h": leading_gap_h,
                "trailing_gap_h": trailing_gap_h,
                "gt500_count": gt500_count,
                "mean_pm25_raw": mean_raw,
                "mean_pm25_le500": mean_le500,
                "max_pm25": max_pm25,
                "first_valid_utc": (
                    first_t.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(first_t) else None
                ),
                "last_valid_utc": (
                    last_t.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(last_t) else None
                ),
            }
        )

    df_st = (
        pd.DataFrame(station_rows, columns=STATION_SUMMARY_COLS)
        .sort_values(["usable_demo", "valid_hours"], ascending=[False, False])
        .reset_index(drop=True)
    )
    if all_gaps:
        df_gp = (
            pd.DataFrame(all_gaps, columns=GAP_COLS)
            .sort_values("gap_hours", ascending=False)
            .reset_index(drop=True)
        )
    else:
        df_gp = pd.DataFrame(columns=GAP_COLS)
    return df_st, df_gp
