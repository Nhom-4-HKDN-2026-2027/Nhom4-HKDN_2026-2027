"""Configuration validation and custom-window pipeline integration."""

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest import verify_vn_data as pipeline  # noqa: E402
from ingest.config import DEFAULT_CONFIG, PipelineConfig, load_config  # noqa: E402
from ingest.openaq_fetcher import PM25_HOURLY_COLS, STATIONS_METADATA_COLS  # noqa: E402
from ingest.openmeteo_fetcher import WEATHER_COLS  # noqa: E402


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_end_utc": "2030-01-01T00:00:00Z"},
        {"alive_cutoff_utc": "2026-11-14T00:00:00Z"},
        {"window_start_utc": "2024-11-14"},
        {"max_missing_2yr_pct": 101},
        {"weather_model": "typo"},
    ],
)
def test_reject_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG, **changes)


def test_unknown_config_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[pipeline]\nweather_modle="era5"\n')
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


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
