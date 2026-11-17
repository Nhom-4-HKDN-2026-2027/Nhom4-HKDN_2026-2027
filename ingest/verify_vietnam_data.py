"""CLI and pipeline runner to fetch, verify, and sample Hanoi/HCMC OpenAQ & Open-Meteo data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from ingest.fetch_openaq import (
    PM25_HOURLY_COLS,
    STATIONS_METADATA_COLS,
    classify_station_lifecycle,
    compute_completeness_and_gaps,
    fetch_all_active_stations_2yr,
    fetch_vietnam_pm25_stations,
)
from ingest.fetch_openmeteo import (
    WEATHER_COLS,
    fetch_usable_stations_weather_2yr,
)
from ingest.pipeline_config import DEFAULT_CONFIG, PipelineConfig, load_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

WINDOW_START_UTC = DEFAULT_CONFIG.window_start_utc
WINDOW_END_UTC = DEFAULT_CONFIG.window_end_utc
SAMPLE_START_UTC = DEFAULT_CONFIG.sample_start_utc
SAMPLE_END_UTC = DEFAULT_CONFIG.sample_end_utc

USABLE_STATION_IDS: list[int] = [3276359, 4946812, 2161292, 4946813, 6123215]
# Historical reference only. Pass usable_ids explicitly to reproduce an old station subset.
INCLUDE_BOUNDARY_GAPS = DEFAULT_CONFIG.include_boundary_gaps
CACHE_FILES = [
    "stations_hn_hcmc_all_62.csv",  # Legacy filename, not a claimed inventory size.
    "openaq_pm25_full_2yr.csv",
    "openmeteo_weather_usable_stations_2yr.csv",
]


def validate_cached_datasets(
    stations, pm, weather, config: PipelineConfig = DEFAULT_CONFIG
) -> None:
    """Validate structural compatibility; sparse PM data alone cannot prove fetch coverage."""
    for label, frame, cols in [
        ("stations", stations, STATIONS_METADATA_COLS),
        ("PM2.5", pm, PM25_HOURLY_COLS),
        ("weather", weather, WEATHER_COLS),
    ]:
        missing = set(cols) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} cache missing columns: {sorted(missing)}")
    if stations.location_id.isna().any() or stations.location_id.duplicated().any():
        raise ValueError("Invalid or duplicate inventory location_id")
    inventory = stations.set_index("location_id")
    for label, frame in [("PM2.5", pm), ("weather", weather)]:
        if frame.station_id.isna().any() or not set(frame.station_id) <= set(inventory.index):
            raise ValueError(f"{label} cache has unknown station_id")
        ts = pd.to_datetime(frame.event_time, utc=True, errors="coerce")
        if (
            ts.isna().any()
            or (ts < pd.Timestamp(config.window_start_utc)).any()
            or (ts >= pd.Timestamp(config.window_end_utc)).any()
            or not ts.equals(ts.dt.floor("h"))
        ):
            raise ValueError(f"{label} timestamps incompatible with evaluation window")
        for station_id, group in frame.groupby("station_id"):
            st = inventory.loc[station_id]
            if not (group.city == st.city).all():
                raise ValueError(f"{label} city mismatch for station {station_id}")
            for col, meta_col in [("latitude", "lat"), ("longitude", "lon")]:
                coords = pd.to_numeric(group[col], errors="coerce")
                if not np.isclose(coords, float(st[meta_col]), atol=1e-6, rtol=0).all():
                    raise ValueError(f"{label} coordinate mismatch for station {station_id}")
            if label == "PM2.5" and not (group.sensor_id == st.sensor_id).all():
                raise ValueError(f"PM2.5 sensor mismatch for station {station_id}")
        numeric_cols = ["pm25_ug_m3", "percent_coverage"] if label == "PM2.5" else WEATHER_COLS[6:]
        for col in numeric_cols:
            converted = pd.to_numeric(frame[col], errors="coerce")
            if (frame[col].notna() & converted.isna()).any():
                raise ValueError(f"{label} nonnumeric values in {col}")
    comp, _ = compute_completeness_and_gaps(pm, **config.metrics_kwargs())
    usable_ids = set(select_usable_station_ids(comp))
    if set(weather.station_id) != usable_ids:
        raise ValueError("Weather station_id set differs from usable PM2.5 stations")
    expected = pd.date_range(
        config.window_start_utc, config.window_end_utc, inclusive="left", freq="h"
    )
    for station_id, group in weather.groupby("station_id"):
        ts = pd.DatetimeIndex(pd.to_datetime(group.event_time, utc=True))
        if not ts.equals(expected):
            raise ValueError(f"Weather hourly grid incomplete/duplicate/unsorted: {station_id}")
    if weather[WEATHER_COLS[6:]].isna().any().any():
        logger.warning("Weather cache includes missing values; retained for quality reporting")


def _check_new_paths(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing outputs: {existing}. "
            "Choose a new --raw-dir/--sample-dir, or use --no-write."
        )


def select_usable_station_ids(df_completeness: pd.DataFrame) -> list[int]:
    """Select usable station IDs dynamically from the computed quality criteria."""
    if df_completeness.empty or "usable_demo" not in df_completeness.columns:
        return []
    mask = df_completeness["usable_demo"].map(
        lambda value: (
            value is True
            or isinstance(value, np.bool_)
            and bool(value)
            or isinstance(value, str)
            and value.lower() == "true"
        )
    )
    return [int(st_id) for st_id in df_completeness.loc[mask, "station_id"].tolist()]


def extract_one_week_samples(
    df_pm25_2yr: pd.DataFrame,
    df_wx_2yr: pd.DataFrame,
    usable_ids: list[int] | None = None,
    sample_start_utc: str = SAMPLE_START_UTC,
    sample_end_utc: str = SAMPLE_END_UTC,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract 1-week PM2.5 and meteorological sample DataFrames for the usable stations."""
    if usable_ids is None:
        df_comp, _ = compute_completeness_and_gaps(
            df_pm25_2yr,
            **config.metrics_kwargs(),
        )
        target_ids = select_usable_station_ids(df_comp)
    else:
        target_ids = usable_ids

    s_start = pd.Timestamp(sample_start_utc)
    s_end = pd.Timestamp(sample_end_utc)

    if df_pm25_2yr.empty:
        pm_sample = pd.DataFrame(columns=PM25_HOURLY_COLS)
    else:
        pm = df_pm25_2yr.copy()
        pm["station_name"] = pm["station_name"].astype(str).str.strip()
        pm["event_time_dt"] = pd.to_datetime(pm["event_time"], utc=True, errors="coerce")
        pm_vals = pd.to_numeric(pm["pm25_ug_m3"], errors="coerce")
        pm_sample = (
            pm[
                pm["station_id"].isin(target_ids)
                & pm["event_time_dt"].notna()
                & (pm["event_time_dt"] >= s_start)
                & (pm["event_time_dt"] < s_end)
                & np.isfinite(pm_vals)
                & (pm_vals >= 0)
            ]
            .sort_values(["city", "station_id", "event_time_dt"])
            .drop_duplicates(subset=["station_id", "event_time_dt"], keep="first")
            .drop(columns=["event_time_dt"])
            .reset_index(drop=True)
        )

    if df_wx_2yr.empty:
        wx_sample = pd.DataFrame(columns=WEATHER_COLS)
    else:
        wx = df_wx_2yr.copy()
        wx["event_time_dt"] = pd.to_datetime(wx["event_time"], utc=True, errors="coerce")
        wx_sample = (
            wx[
                wx["station_id"].isin(target_ids)
                & wx["event_time_dt"].notna()
                & (wx["event_time_dt"] >= s_start)
                & (wx["event_time_dt"] < s_end)
            ]
            .sort_values(["city", "station_id", "event_time_dt"])
            .drop(columns=["event_time_dt"])
            .reset_index(drop=True)
        )
    return pm_sample, wx_sample


def build_summary_and_samples(
    df_pm25_2yr: pd.DataFrame,
    df_wx_2yr: pd.DataFrame,
    write_outputs: bool = True,
    raw_dir: Path = RAW_DIR,
    sample_dir: Path = SAMPLE_DIR,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute 2-year completeness/gaps and generate 1-week samples from raw DataFrames."""
    df_completeness, df_gaps = compute_completeness_and_gaps(
        df_pm25_2yr,
        **config.metrics_kwargs(),
    )
    usable_ids = select_usable_station_ids(df_completeness)
    pm_sample, wx_sample = extract_one_week_samples(
        df_pm25_2yr,
        df_wx_2yr,
        usable_ids=usable_ids,
        sample_start_utc=config.sample_start_utc,
        sample_end_utc=config.sample_end_utc,
        config=config,
    )

    if write_outputs:
        _check_new_paths(
            [
                raw_dir / "completeness_2yr_summary.csv",
                raw_dir / "gaps_2yr.csv",
                sample_dir / "openaq_pm25_hanoi_hcmc.csv",
                sample_dir / "openmeteo_weather_hanoi_hcmc.csv",
            ]
        )
        raw_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        df_completeness.to_csv(raw_dir / "completeness_2yr_summary.csv", index=False)
        df_gaps.to_csv(raw_dir / "gaps_2yr.csv", index=False)
        pm_sample.to_csv(sample_dir / "openaq_pm25_hanoi_hcmc.csv", index=False)
        wx_sample.to_csv(sample_dir / "openmeteo_weather_hanoi_hcmc.csv", index=False)

    return df_completeness, df_gaps, pm_sample, wx_sample


def load_or_fetch_vn_datasets(
    force_fetch: bool = False,
    raw_dir: Path = RAW_DIR,
    write_outputs: bool = True,
    allow_legacy_cache: bool = False,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 2-year raw CSVs from raw_dir if present, otherwise fetch via OpenAQ & Open-Meteo APIs."""
    stations_csv, pm25_2yr_csv, wx_2yr_csv = [raw_dir / name for name in CACHE_FILES]
    manifest_path = raw_dir / "fetch_manifest.json"

    if (
        not force_fetch
        and stations_csv.is_file()
        and pm25_2yr_csv.is_file()
        and wx_2yr_csv.is_file()
    ):
        df_stations = pd.read_csv(stations_csv)
        df_pm25_2yr = pd.read_csv(pm25_2yr_csv)
        df_wx_2yr = pd.read_csv(wx_2yr_csv)
        try:
            validate_cached_datasets(df_stations, df_pm25_2yr, df_wx_2yr, config=config)
        except ValueError as exc:
            raise ValueError(
                f"Incompatible cache: {exc}. Fetch again with --fetch into a new --raw-dir."
            ) from exc
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            expected = {
                "version": 2,
                "window_start_utc": config.window_start_utc,
                "window_end_utc": config.window_end_utc,
                "weather_model": config.weather_model,
                "include_boundary_gaps": config.include_boundary_gaps,
                "config": config.provenance(),
                "sensor_selection": "max_valid_hours_in_window",
            }
            if any(manifest.get(key) != val for key, val in expected.items()):
                raise ValueError("Cache configuration incompatible; fetch into a new --raw-dir")
            for path in [stations_csv, pm25_2yr_csv, wx_2yr_csv]:
                if (
                    manifest.get("sha256", {}).get(path.name)
                    != hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    raise ValueError(
                        f"Cache checksum mismatch: {path.name}; fetch into a new --raw-dir"
                    )
            statuses = {int(k): v for k, v in manifest.get("fetch_status", {}).items()}
            if set(statuses) != set(df_stations.location_id):
                raise ValueError("Cache fetch_status inventory mismatch")
            if not set(statuses.values()) <= {
                "fetched",
                "not_fetched",
                "fetch_failed",
                "no_data_in_window",
                "inactive",
            }:
                raise ValueError("Cache contains invalid fetch statuses")
            with_rows = set(df_pm25_2yr.station_id)
            if with_rows != {sid for sid, status in statuses.items() if status == "fetched"}:
                raise ValueError("Cache fetch_status disagrees with PM2.5 station rows")
            if any(status in {"not_fetched", "fetch_failed"} for status in statuses.values()):
                logger.warning(
                    "Cache contains incomplete station fetches; review lifecycle before use"
                )
            df_pm25_2yr.attrs["fetch_status"] = statuses
        elif not allow_legacy_cache:
            raise ValueError(
                "Legacy cache has no fetch_manifest.json: window/model/fetch coverage unverified. "
                "Use --allow-legacy-cache for explicit local audit, or --fetch with a new --raw-dir."
            )
        else:
            logger.warning("Legacy cache: source window/model/fetch provenance UNVERIFIED")
        return df_stations, df_pm25_2yr, df_wx_2yr

    logger.info("Cache missing or --fetch requested; fetching source data")
    if write_outputs:
        _check_new_paths([stations_csv, pm25_2yr_csv, wx_2yr_csv, manifest_path])
    api_key = os.environ.get("OPENAQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAQ_API_KEY environment variable. "
            "Register a free key at https://explore.openaq.org/register to fetch 2-year raw data."
        )

    df_stations = fetch_vietnam_pm25_stations(api_key=api_key)

    df_pm25_2yr = fetch_all_active_stations_2yr(
        stations_df=df_stations,
        api_key=api_key,
        window_start_utc=config.window_start_utc,
        window_end_utc=config.window_end_utc,
    )
    selected = df_pm25_2yr.attrs.get("selected_sensors", {})
    for loc_id, sensor_id in selected.items():
        df_stations.loc[df_stations.location_id == loc_id, "sensor_id"] = sensor_id

    df_comp, _ = compute_completeness_and_gaps(
        df_pm25_2yr,
        **config.metrics_kwargs(),
    )
    usable_ids = select_usable_station_ids(df_comp)
    if not usable_ids:
        logger.warning("No usable stations: weather and samples will be empty")

    usable_meta = (
        df_pm25_2yr[df_pm25_2yr["station_id"].isin(usable_ids)][
            ["city", "station_id", "station_name", "latitude", "longitude"]
        ]
        .drop_duplicates(subset=["station_id"])
        .to_dict(orient="records")
    )
    df_wx_2yr = fetch_usable_stations_weather_2yr(
        stations_meta=usable_meta,
        start_date=pd.Timestamp(config.window_start_utc).strftime("%Y-%m-%d"),
        end_date=(pd.Timestamp(config.window_end_utc) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        models=config.weather_model,
        allow_nan=True,
    )
    validate_cached_datasets(df_stations, df_pm25_2yr, df_wx_2yr, config=config)
    if write_outputs:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for frame, path in [
            (df_stations, stations_csv),
            (df_pm25_2yr, pm25_2yr_csv),
            (df_wx_2yr, wx_2yr_csv),
        ]:
            frame.to_csv(path, index=False, mode="x")
        manifest = {
            "version": 2,
            "window_start_utc": config.window_start_utc,
            "window_end_utc": config.window_end_utc,
            "weather_model": config.weather_model,
            "include_boundary_gaps": config.include_boundary_gaps,
            "config": config.provenance(),
            "sensor_selection": "max_valid_hours_in_window",
            "python_version": platform.python_version(),
            "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "fetch_status": df_pm25_2yr.attrs.get("fetch_status", {}),
            "sensor_scores": df_pm25_2yr.attrs.get("sensor_scores", {}),
            "sensor_candidates": df_stations.attrs.get("sensor_candidates", {}),
            "weather_missing_values": int(df_wx_2yr[WEATHER_COLS[6:]].isna().sum().sum()),
            "sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [stations_csv, pm25_2yr_csv, wx_2yr_csv]
            },
        }
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, indent=2)

    return df_stations, df_pm25_2yr, df_wx_2yr


def main() -> None:
    """Run the data verification and sample generation pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Fetch and verify 2-year OpenAQ PM2.5 and Open-Meteo data for Hanoi and HCMC."
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Force re-fetching 2-year data from OpenAQ and Open-Meteo APIs into data/raw/.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Compute summaries in memory without overwriting CSVs on disk.",
    )
    parser.add_argument("--config", type=Path, help="Non-secret pipeline TOML")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument(
        "--allow-legacy-cache",
        action="store_true",
        help="Audit old CSVs without asserting source fetch provenance.",
    )
    args = parser.parse_args()
    config = load_config(args.config) if args.config else DEFAULT_CONFIG
    raw_dir = args.raw_dir or (config.path("raw_dir") if args.config else RAW_DIR)
    sample_dir = args.sample_dir or (config.path("sample_dir") if args.config else SAMPLE_DIR)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.no_write:
        # Check every output before spending requests or persisting a partial run.
        _check_new_paths(
            [
                raw_dir / "completeness_2yr_summary.csv",
                raw_dir / "gaps_2yr.csv",
                sample_dir / "openaq_pm25_hanoi_hcmc.csv",
                sample_dir / "openmeteo_weather_hanoi_hcmc.csv",
            ]
        )
        if args.fetch or not all((raw_dir / name).is_file() for name in CACHE_FILES):
            _check_new_paths(
                [raw_dir / name for name in CACHE_FILES] + [raw_dir / "fetch_manifest.json"]
            )

    df_stations, df_pm25_2yr, df_wx_2yr = load_or_fetch_vn_datasets(
        force_fetch=args.fetch,
        raw_dir=raw_dir,
        write_outputs=not args.no_write,
        allow_legacy_cache=args.allow_legacy_cache,
        config=config,
    )
    df_lifecycle = classify_station_lifecycle(
        df_stations,
        df_pm25_2yr,
        window_start_utc=config.window_start_utc,
        window_end_utc=config.window_end_utc,
        alive_cutoff_utc=config.alive_cutoff_utc,
    )
    df_completeness, df_gaps, pm_sample, wx_sample = build_summary_and_samples(
        df_pm25_2yr,
        df_wx_2yr,
        write_outputs=not args.no_write,
        raw_dir=raw_dir,
        sample_dir=sample_dir,
        config=config,
    )

    print(f"=== {len(df_stations)}-Station Lifecycle Summary ===")
    print(df_lifecycle.to_string(index=False))
    print("\n=== Top Stations by 2-Year Completeness ===")
    cols = [
        "city",
        "station_id",
        "station_name",
        "usable_demo",
        "valid_hours",
        "missing_2yr_pct",
        "active_span_h",
        "missing_in_active_pct",
        "num_gaps",
        "num_gaps_ge_24h",
        "max_gap_hours",
    ]
    print(df_completeness[cols].head(8).to_string(index=False))
    print(f"\nTotal 2-year PM2.5 rows: {len(df_pm25_2yr):,}")
    print(f"Total 2-year Weather rows: {len(df_wx_2yr):,}")
    print(f"Total continuous gaps recorded: {len(df_gaps):,}")
    print(f"1-week sample rows: PM2.5={len(pm_sample):,}, Weather={len(wx_sample):,}")


if __name__ == "__main__":
    main()
