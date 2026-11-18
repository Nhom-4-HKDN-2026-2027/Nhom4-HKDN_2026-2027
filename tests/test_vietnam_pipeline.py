"""Offline pipeline, cache provenance, sample handoff and CLI tests."""

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest import fetch_openaq as aq  # noqa: E402
from ingest import fetch_openmeteo as wx  # noqa: E402
from ingest import verify_vietnam_data as pipeline  # noqa: E402
from ingest.fetch_openaq import (  # noqa: E402
    PM25_HOURLY_COLS,
    STATIONS_METADATA_COLS,
    compute_completeness_and_gaps,
)
from ingest.fetch_openmeteo import WEATHER_COLS  # noqa: E402
from ingest.pipeline_config import PipelineConfig  # noqa: E402
from ingest.verify_vietnam_data import (  # noqa: E402
    USABLE_STATION_IDS,
    build_summary_and_samples,
    select_usable_station_ids,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_USABLE_STATIONS = set(USABLE_STATION_IDS)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(
        aq.requests, "get", Mock(side_effect=AssertionError("Unexpected network call"))
    )


def pm_frame(values, times=None):
    if times is None:
        times = pd.date_range("2026-11-13", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "city": "Hanoi",
            "station_id": 1,
            "station_name": "Test",
            "sensor_id": 10,
            "provider": "mock",
            "is_monitor": True,
            "latitude": 21.0,
            "longitude": 105.8,
            "event_time": times,
            "pm25_ug_m3": values,
            "percent_coverage": 100.0,
        }
    )


def station_frame():
    return pd.DataFrame(
        [
            dict(
                zip(
                    aq.STATIONS_METADATA_COLS,
                    [
                        "Hanoi",
                        1,
                        "Test",
                        "Hanoi",
                        "mock",
                        "mock",
                        True,
                        False,
                        10,
                        21.0,
                        105.8,
                        None,
                        None,
                    ],
                    strict=True,
                )
            )
        ]
    )


def cached_frames():
    pm = pm_frame([20.0] * 17520, pd.date_range(pipeline.WINDOW_START_UTC, periods=17520, freq="h"))
    weather = pm[
        ["city", "station_id", "station_name", "latitude", "longitude", "event_time"]
    ].copy()
    for col in wx.WEATHER_COLS[6:]:
        weather[col] = 1.0
    return station_frame(), pm, weather


def test_sample_files_exist_and_follow_contracts():
    """Verify 1-week sample CSV files for the 5 usable stations follow CONTRACTS.md."""
    sample_dir = PROJECT_ROOT / "data" / "sample"
    openaq_csv = sample_dir / "openaq_pm25_hanoi_hcmc.csv"
    weather_csv = sample_dir / "openmeteo_weather_hanoi_hcmc.csv"

    assert openaq_csv.is_file(), f"Missing handoff sample: {openaq_csv}"
    assert weather_csv.is_file(), f"Missing handoff sample: {weather_csv}"

    df_pm25 = pd.read_csv(openaq_csv)
    df_weather = pd.read_csv(weather_csv)

    assert not df_pm25.empty, "OpenAQ sample CSV is empty"
    assert not df_weather.empty, "Open-Meteo sample CSV is empty"

    for col in ["station_id", "event_time", "pm25_ug_m3", "latitude", "longitude"]:
        assert col in df_pm25.columns, f"Missing column {col} in OpenAQ sample"

    for col in [
        "station_id",
        "event_time",
        "temperature_c",
        "relative_humidity_pct",
        "wind_speed_m_s",
        "wind_direction_deg",
        "precipitation_mm",
    ]:
        assert col in df_weather.columns, f"Missing column {col} in Open-Meteo sample"

    assert set(df_pm25["station_id"].unique()) == EXPECTED_USABLE_STATIONS
    assert set(df_weather["station_id"].unique()) == EXPECTED_USABLE_STATIONS
    assert len(df_weather) == 5 * 168


def test_deliverable_artifacts_exist():
    """Verify that the notebook, ingest modules, and 1-page data quality report exist."""
    notebook = PROJECT_ROOT / "notebooks" / "01_verify_vn_data.ipynb"
    report = PROJECT_ROOT / "docs" / "data_quality_report.md"
    openaq_mod = PROJECT_ROOT / "ingest" / "fetch_openaq.py"
    openmeteo_mod = PROJECT_ROOT / "ingest" / "fetch_openmeteo.py"
    verify_cli = PROJECT_ROOT / "ingest" / "verify_vietnam_data.py"

    assert notebook.is_file(), "Missing notebooks/01_verify_vn_data.ipynb"
    assert report.is_file(), "Missing docs/data_quality_report.md"
    assert openaq_mod.is_file(), "Missing ingest/fetch_openaq.py"
    assert openmeteo_mod.is_file(), "Missing ingest/fetch_openmeteo.py"
    assert verify_cli.is_file(), "Missing ingest/verify_vietnam_data.py"
    assert report.read_text(encoding="utf-8").strip(), "Data quality report is empty"


def test_separate_cache_summary_and_dynamic_usable_selection(tmp_path: Path):
    """Verify Checklist #5 and #6: build_summary_and_samples writes outputs & dynamically selects usable IDs."""
    pm25_df = pd.DataFrame(
        [
            {
                "city": "HCMC",
                "station_id": 3276359,
                "station_name": "CMT8",
                "sensor_id": 11357424,
                "provider": "AirGradient",
                "is_monitor": False,
                "latitude": 10.78,
                "longitude": 106.67,
                "event_time": "2025-11-16T00:00:00Z",
                "pm25_ug_m3": 20.0,
                "percent_coverage": 100.0,
            },
            {
                "city": "HCMC",
                "station_id": 3276359,
                "station_name": "CMT8",
                "sensor_id": 11357424,
                "provider": "AirGradient",
                "is_monitor": False,
                "latitude": 10.78,
                "longitude": 106.67,
                "event_time": "2026-11-13T23:00:00Z",
                "pm25_ug_m3": 25.0,
                "percent_coverage": 100.0,
            },
        ]
    )
    wx_df = pd.DataFrame(
        [
            {
                "city": "HCMC",
                "station_id": 3276359,
                "station_name": "CMT8",
                "latitude": 10.78,
                "longitude": 106.67,
                "event_time": "2025-11-16T00:00:00Z",
                "temperature_c": 28.0,
                "relative_humidity_pct": 75,
                "wind_speed_m_s": 2.1,
                "wind_direction_deg": 180,
                "precipitation_mm": 0.0,
            }
        ]
    )
    comp, _ = compute_completeness_and_gaps(pm25_df, max_missing_2yr_pct=100.0)
    assert select_usable_station_ids(comp) == [3276359]

    raw_tmp = tmp_path / "raw"
    sample_tmp = tmp_path / "sample"
    build_summary_and_samples(
        pm25_df,
        wx_df,
        write_outputs=True,
        raw_dir=raw_tmp,
        sample_dir=sample_tmp,
    )
    assert (raw_tmp / "completeness_2yr_summary.csv").is_file()
    assert (sample_tmp / "openaq_pm25_hanoi_hcmc.csv").is_file()
    assert (sample_tmp / "openmeteo_weather_hanoi_hcmc.csv").is_file()


def test_no_silent_hardcoded_sample_fallback():
    pm = pm_frame([20], ["2025-11-16T00:00:00Z"])
    pm["station_id"] = pipeline.USABLE_STATION_IDS[0]
    sample, _ = pipeline.extract_one_week_samples(pm, pd.DataFrame())
    assert sample.empty


def test_no_write_does_not_create_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "dummy")
    monkeypatch.setattr(pipeline, "fetch_vietnam_pm25_stations", Mock(return_value=station_frame()))
    monkeypatch.setattr(
        pipeline, "fetch_all_active_stations_2yr", Mock(return_value=pm_frame([20]))
    )
    monkeypatch.setattr(
        pipeline,
        "fetch_usable_stations_weather_2yr",
        Mock(return_value=pd.DataFrame(columns=wx.WEATHER_COLS)),
    )
    target = tmp_path / "new"
    pipeline.load_or_fetch_vn_datasets(force_fetch=True, raw_dir=target, write_outputs=False)
    assert not target.exists()


def test_valid_cache_and_reproducible_outputs(tmp_path):
    stations, pm, weather = cached_frames()
    pipeline.validate_cached_datasets(stations, pm, weather)
    first = pipeline.build_summary_and_samples(
        pm, weather, raw_dir=tmp_path / "raw", sample_dir=tmp_path / "sample"
    )
    second = pipeline.build_summary_and_samples(pm, weather, write_outputs=False)
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)
    assert len(first[2]) == len(first[3]) == 168
    joined = first[2].merge(first[3], on=["station_id", "event_time"], validate="one_to_one")
    assert len(joined) == 168


@pytest.mark.parametrize(
    "defect",
    ["schema", "window", "station", "sensor", "coords", "weather_hours", "weather_stations"],
)
def test_incompatible_cache(defect):
    stations, pm, weather = cached_frames()
    if defect == "schema":
        pm = pm.drop(columns="pm25_ug_m3")
    elif defect == "window":
        pm.loc[0, "event_time"] = pd.Timestamp("2000-01-01", tz="UTC")
    elif defect == "station":
        pm.loc[0, "station_id"] = 99
    elif defect == "sensor":
        pm.loc[0, "sensor_id"] = 99
    elif defect == "coords":
        weather.loc[0, "latitude"] = 0
    elif defect == "weather_hours":
        weather = weather.iloc[1:]
    else:
        weather = weather.iloc[:0]
    with pytest.raises(ValueError):
        pipeline.validate_cached_datasets(stations, pm, weather)


def test_legacy_cache_explicit_opt_in(tmp_path):
    for frame, name in zip(cached_frames(), pipeline.CACHE_FILES, strict=True):
        frame.to_csv(tmp_path / name, index=False)
    with pytest.raises(ValueError, match="Legacy cache"):
        pipeline.load_or_fetch_vn_datasets(raw_dir=tmp_path)
    result = pipeline.load_or_fetch_vn_datasets(raw_dir=tmp_path, allow_legacy_cache=True)
    assert len(result[1]) == 17520


def test_fetch_write_reload_manifest_and_no_overwrite(tmp_path, monkeypatch):
    stations, pm, weather = cached_frames()
    pm.attrs["fetch_status"] = {1: "fetched"}
    monkeypatch.setenv("OPENAQ_API_KEY", "dummy")
    monkeypatch.setattr(pipeline, "fetch_vietnam_pm25_stations", Mock(return_value=stations))
    monkeypatch.setattr(pipeline, "fetch_all_active_stations_2yr", Mock(return_value=pm))
    monkeypatch.setattr(pipeline, "fetch_usable_stations_weather_2yr", Mock(return_value=weather))
    pipeline.load_or_fetch_vn_datasets(force_fetch=True, raw_dir=tmp_path)
    loaded = pipeline.load_or_fetch_vn_datasets(raw_dir=tmp_path)
    assert loaded[1].attrs["fetch_status"] == {1: "fetched"}
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        pipeline.load_or_fetch_vn_datasets(force_fetch=True, raw_dir=tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_boolean_strings_do_not_select_false_or_nan():
    summary = pd.DataFrame(
        {"station_id": [1, 2, 3, 4], "usable_demo": ["False", np.nan, "True", False]}
    )
    assert pipeline.select_usable_station_ids(summary) == [3]


def test_cli_no_write_preserves_existing_files(tmp_path, monkeypatch):
    stations, pm, weather = cached_frames()
    for frame, name in zip([stations, pm, weather], pipeline.CACHE_FILES, strict=True):
        frame.to_csv(tmp_path / name, index=False)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(
        sys, "argv", ["verify", "--no-write", "--allow-legacy-cache", "--raw-dir", str(tmp_path)]
    )
    pipeline.main()
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_missing_partial_cache_never_overwrites_or_fetches(tmp_path):
    sentinel = tmp_path / pipeline.CACHE_FILES[0]
    sentinel.write_text("existing raw data")
    with pytest.raises(FileExistsError):
        pipeline.load_or_fetch_vn_datasets(raw_dir=tmp_path)
    assert sentinel.read_text() == "existing raw data"


def test_empty_usable_stations_keep_empty_output_schemas():
    summary, gaps, pm, weather = pipeline.build_summary_and_samples(
        pd.DataFrame(columns=aq.PM25_HOURLY_COLS),
        pd.DataFrame(columns=wx.WEATHER_COLS),
        write_outputs=False,
    )
    assert summary.empty and gaps.empty and pm.empty and weather.empty
    assert list(weather) == wx.WEATHER_COLS


def test_cli_conflicting_sample_fails_before_fetch(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "openaq_pm25_hanoi_hcmc.csv").write_text("existing")
    fetch = Mock(side_effect=AssertionError("must not reach fetch"))
    monkeypatch.setattr(pipeline, "load_or_fetch_vn_datasets", fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--fetch", "--raw-dir", str(tmp_path / "raw"), "--sample-dir", str(samples)],
    )
    with pytest.raises(FileExistsError):
        pipeline.main()
    fetch.assert_not_called()
    assert not (tmp_path / "raw").exists()


def test_custom_window_fetch_manifest_reload_and_samples(tmp_path, monkeypatch):
    config = PipelineConfig(
        window_start_utc="2025-01-01T00:00:00Z",
        window_end_utc="2025-01-02T00:00:00Z",
        alive_cutoff_utc="2025-01-01T12:00:00Z",
        sample_start_utc="2025-01-01T00:00:00Z",
        sample_end_utc="2025-01-02T00:00:00Z",
        include_boundary_gaps=True,
    )
    stations = pd.DataFrame(
        [["Hanoi", 1, "Test", "Hanoi", "mock", "mock", True, False, 10, 21, 105.8, None, None]],
        columns=STATIONS_METADATA_COLS,
    )
    times = pd.date_range(config.window_start_utc, periods=24, freq="h").strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    pm = pd.DataFrame(
        {
            "city": "Hanoi",
            "station_id": 1,
            "station_name": "Test",
            "sensor_id": 10,
            "provider": "mock",
            "is_monitor": True,
            "latitude": 21,
            "longitude": 105.8,
            "event_time": times,
            "pm25_ug_m3": 20,
            "percent_coverage": 100,
        },
        columns=PM25_HOURLY_COLS,
    )
    pm.attrs["fetch_status"] = {1: "fetched"}
    weather = pm[WEATHER_COLS[:6]].copy()
    for col in WEATHER_COLS[6:]:
        weather[col] = 1.0
    monkeypatch.setenv("OPENAQ_API_KEY", "dummy")
    monkeypatch.setattr(pipeline, "fetch_vietnam_pm25_stations", Mock(return_value=stations))
    monkeypatch.setattr(pipeline, "fetch_all_active_stations_2yr", Mock(return_value=pm))
    fetch_weather = Mock(return_value=weather)
    monkeypatch.setattr(pipeline, "fetch_usable_stations_weather_2yr", fetch_weather)
    raw = tmp_path / "raw"
    pipeline.load_or_fetch_vn_datasets(force_fetch=True, raw_dir=raw, config=config)
    loaded = pipeline.load_or_fetch_vn_datasets(raw_dir=raw, config=config)
    summary, _, ps, ws = pipeline.build_summary_and_samples(
        loaded[1],
        loaded[2],
        config=config,
        raw_dir=raw,
        sample_dir=tmp_path / "sample",
    )
    assert summary.iloc[0].valid_hours == 24
    assert len(ps) == len(ws) == 24
    assert fetch_weather.call_args.kwargs["start_date"] == "2025-01-01"
    assert fetch_weather.call_args.kwargs["end_date"] == "2025-01-01"
    with pytest.raises(ValueError):
        pipeline.load_or_fetch_vn_datasets(raw_dir=raw, config=replace(config, max_gt500_count=2))
