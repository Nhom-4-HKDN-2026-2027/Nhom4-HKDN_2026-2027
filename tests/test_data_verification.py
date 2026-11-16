import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest.openaq_fetcher import (  # noqa: E402
    GAP_COLS,
    LIFECYCLE_CATEGORIES,
    PM25_HOURLY_COLS,
    STATION_SUMMARY_COLS,
    STATIONS_METADATA_COLS,
    _select_primary_pm25_sensor,
    classify_station_lifecycle,
    compute_completeness_and_gaps,
    fetch_all_active_stations_2yr,
    fetch_vietnam_pm25_stations,
)
from ingest.openmeteo_fetcher import (  # noqa: E402
    EXPECTED_UNITS,
    WEATHER_COLS,
    fetch_station_hourly_weather,
    fetch_usable_stations_weather_2yr,
)
from ingest.verify_vn_data import (  # noqa: E402
    USABLE_STATION_IDS,
    build_summary_and_samples,
    extract_one_week_samples,
    select_usable_station_ids,
)

EXPECTED_USABLE_STATIONS = set(USABLE_STATION_IDS)


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
    report = PROJECT_ROOT / "docs" / "data_quality_report_vn.md"
    openaq_mod = PROJECT_ROOT / "ingest" / "openaq_fetcher.py"
    openmeteo_mod = PROJECT_ROOT / "ingest" / "openmeteo_fetcher.py"
    verify_cli = PROJECT_ROOT / "ingest" / "verify_vn_data.py"

    assert notebook.is_file(), "Missing notebooks/01_verify_vn_data.ipynb"
    assert report.is_file(), "Missing docs/data_quality_report_vn.md"
    assert openaq_mod.is_file(), "Missing ingest/openaq_fetcher.py"
    assert openmeteo_mod.is_file(), "Missing ingest/openmeteo_fetcher.py"
    assert verify_cli.is_file(), "Missing ingest/verify_vn_data.py"
    assert report.read_text(encoding="utf-8").strip(), "Data quality report is empty"


def test_empty_dataframes_always_preserve_schema():
    """Verify Checklist #4: all functions returning empty DataFrames preserve exact column schemas."""
    with patch("ingest.openaq_fetcher._get_with_retry") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_get.return_value = mock_resp
        df_st_empty = fetch_vietnam_pm25_stations(api_key="dummy")
        assert df_st_empty.empty
        assert list(df_st_empty.columns) == STATIONS_METADATA_COLS

    df_pm_empty = fetch_all_active_stations_2yr(
        pd.DataFrame(columns=STATIONS_METADATA_COLS), api_key="dummy"
    )
    assert df_pm_empty.empty
    assert list(df_pm_empty.columns) == PM25_HOURLY_COLS

    comp_empty, gaps_empty = compute_completeness_and_gaps(df_pm_empty)
    assert comp_empty.empty
    assert list(comp_empty.columns) == STATION_SUMMARY_COLS
    assert gaps_empty.empty
    assert list(gaps_empty.columns) == GAP_COLS

    wx_empty = fetch_usable_stations_weather_2yr([])
    assert wx_empty.empty
    assert list(wx_empty.columns) == WEATHER_COLS


def test_primary_sensor_selection_from_detailed_metadata():
    """Verify Checklist #1: primary sensor selection queries /locations/{id}/sensors when needed."""
    summary_sensors = [
        {"id": 100, "parameter": {"id": 2, "name": "pm25"}},
        {"id": 205, "parameter": {"id": 2, "name": "pm25"}},
    ]
    with patch("ingest.openaq_fetcher._get_with_retry") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "id": 100,
                    "parameter": {"id": 2, "name": "pm25"},
                    "datetimeLast": {"utc": "2024-06-01T00:00:00Z"},
                    "coverage": {"observedCount": 5000},
                },
                {
                    "id": 205,
                    "parameter": {"id": 2, "name": "pm25"},
                    "datetimeLast": {"utc": "2026-11-13T23:00:00Z"},
                    "coverage": {"observedCount": 12000},
                },
            ]
        }
        mock_get.return_value = mock_resp
        selected = _select_primary_pm25_sensor(
            summary_sensors,
            location_id=4946812,
            headers={"X-API-Key": "dummy"},
        )
        assert selected == 205
        mock_get.assert_called_once()


def test_lifecycle_for_skipped_nodata_invalid_and_active_stations():
    """Verify Checklist #2: lifecycle accurately classifies skipped, no-data, invalid/NaN, and active."""
    stations_df = pd.DataFrame(
        [
            {"city": "Hanoi", "location_id": 1, "name": "Skipped Old Station"},
            {"city": "Hanoi", "location_id": 2, "name": "Out-of-window Station"},
            {"city": "HCMC", "location_id": 7440, "name": "All-NaN & Negative Station"},
            {"city": "Hanoi", "location_id": 4946811, "name": "Stopped Mid-Window"},
            {"city": "HCMC", "location_id": 3276359, "name": "Active End"},
        ]
    )
    pm25_df = pd.DataFrame(
        [
            {
                "city": "Hanoi",
                "station_id": 2,
                "station_name": "Out-of-window Station",
                "event_time": "2023-05-01T00:00:00Z",
                "pm25_ug_m3": 30.0,
            },
            {
                "city": "HCMC",
                "station_id": 7440,
                "station_name": "All-NaN & Negative Station",
                "event_time": "2025-01-01T00:00:00Z",
                "pm25_ug_m3": float("nan"),
            },
            {
                "city": "HCMC",
                "station_id": 7440,
                "station_name": "All-NaN & Negative Station",
                "event_time": "2025-01-01T01:00:00Z",
                "pm25_ug_m3": -5.0,
            },
            {
                "city": "Hanoi",
                "station_id": 4946811,
                "station_name": "Stopped Mid-Window",
                "event_time": "2026-07-15T10:00:00Z",
                "pm25_ug_m3": 25.0,
            },
            {
                "city": "HCMC",
                "station_id": 3276359,
                "station_name": "Active End",
                "event_time": "2026-11-13T23:00:00Z",
                "pm25_ug_m3": 18.0,
            },
        ]
    )
    lifecycle = classify_station_lifecycle(stations_df, pm25_df)
    assert list(lifecycle["lifecycle_status"]) == LIFECYCLE_CATEGORIES
    totals = dict(zip(lifecycle["lifecycle_status"], lifecycle["Total"], strict=True))
    assert totals[LIFECYCLE_CATEGORIES[0]] == 2  # Station 1 (no rows) & Station 2 (out of window)
    assert totals[LIFECYCLE_CATEGORIES[1]] == 1  # Station 7440 (NaN and < 0 only)
    assert totals[LIFECYCLE_CATEGORIES[2]] == 1  # Station 4946811 (stopped 07/2026)
    assert totals[LIFECYCLE_CATEGORIES[3]] == 1  # Station 3276359 (active 11/2026)


def test_boundary_gaps_and_zero_valid_hours_invariant():
    """Verify Checklist #3: leading_gap_h + active_span_h + trailing_gap_h == total_window_hours."""
    df_test = pd.DataFrame(
        [
            {
                "city": "HCMC",
                "station_id": 7440,
                "station_name": "Zero Valid Station",
                "sensor_id": 99,
                "is_monitor": True,
                "event_time": "2026-11-10T01:00:00Z",
                "pm25_ug_m3": float("nan"),
            },
            {
                "city": "Hanoi",
                "station_id": 1001,
                "station_name": "Partial Active Station",
                "sensor_id": 2001,
                "is_monitor": True,
                "event_time": "2026-11-10T02:00:00Z",
                "pm25_ug_m3": 35.0,
            },
            {
                "city": "Hanoi",
                "station_id": 1001,
                "station_name": "Partial Active Station",
                "sensor_id": 2001,
                "is_monitor": True,
                "event_time": "2026-11-10T04:00:00Z",
                "pm25_ug_m3": 40.0,
            },
        ]
    )
    comp, gaps_internal = compute_completeness_and_gaps(
        df_test,
        window_start_utc="2026-11-10T00:00:00Z",
        window_end_utc="2026-11-10T06:00:00Z",
        include_boundary_gaps=False,
    )
    for _, r in comp.iterrows():
        assert r["leading_gap_h"] + r["active_span_h"] + r["trailing_gap_h"] == 6
        assert (
            r["leading_gap_h"] + r["missing_active_h"] + r["trailing_gap_h"] == 6 - r["valid_hours"]
        )

    r_zero = comp[comp["station_id"] == 7440].iloc[0]
    assert r_zero["valid_hours"] == 0
    assert r_zero["leading_gap_h"] == 6
    assert r_zero["trailing_gap_h"] == 0
    assert r_zero["num_gaps"] == 0

    comp_b, gaps_boundary = compute_completeness_and_gaps(
        df_test,
        window_start_utc="2026-11-10T00:00:00Z",
        window_end_utc="2026-11-10T06:00:00Z",
        include_boundary_gaps=True,
    )
    r_part_b = comp_b[comp_b["station_id"] == 1001].iloc[0]
    assert r_part_b["num_gaps"] == 3  # 2h leading + 1h internal + 1h trailing
    assert len(gaps_boundary) == 4  # 3 for station 1001 + 1 full-window 6h gap for station 7440


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


def test_openmeteo_expected_hours_and_timestamp_validation():
    """Verify Checklist #7: Open-Meteo validates expected hour count, UTC grid, and NaN values."""
    with patch("ingest.openmeteo_fetcher._get_with_retry") as mock_get:
        # Case 1: Valid 24-hour payload for 2025-11-15
        times_24 = (
            pd.date_range("2025-11-15T00:00:00Z", "2025-11-15T23:00:00Z", freq="1h", tz="UTC")
            .strftime("%Y-%m-%dT%H:%M")
            .tolist()
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hourly_units": EXPECTED_UNITS,
            "hourly": {
                "time": times_24,
                "temperature_2m": [25.0] * 24,
                "relative_humidity_2m": [70] * 24,
                "wind_speed_10m": [2.5] * 24,
                "wind_direction_10m": [180] * 24,
                "precipitation": [0.0] * 24,
            },
        }
        mock_get.return_value = mock_resp

        df_wx = fetch_station_hourly_weather(
            station_id=3276359,
            station_name="CMT8",
            city="HCMC",
            latitude=10.78,
            longitude=106.67,
            start_date="2025-11-15",
            end_date="2025-11-15",
        )
        assert len(df_wx) == 24
        assert df_wx.iloc[0]["event_time"] == "2025-11-15T00:00:00Z"
        assert df_wx.iloc[-1]["event_time"] == "2025-11-15T23:00:00Z"

        # Case 2: Mismatched hour count (23 instead of 24) raises ValueError
        mock_resp.json.return_value["hourly"]["time"] = times_24[:23]
        with pytest.raises(ValueError, match="expected 24 hours"):
            fetch_station_hourly_weather(
                station_id=3276359,
                station_name="CMT8",
                city="HCMC",
                latitude=10.78,
                longitude=106.67,
                start_date="2025-11-15",
                end_date="2025-11-15",
            )

        # Case 3: NaN value in weather array raises ValueError when allow_nan=False
        mock_resp.json.return_value["hourly"]["time"] = times_24
        mock_resp.json.return_value["hourly"]["temperature_2m"] = [25.0] * 23 + [None]
        with pytest.raises(ValueError, match="NaN values"):
            fetch_station_hourly_weather(
                station_id=3276359,
                station_name="CMT8",
                city="HCMC",
                latitude=10.78,
                longitude=106.67,
                start_date="2025-11-15",
                end_date="2025-11-15",
                allow_nan=False,
            )


def test_duplicate_timestamps_and_out_of_window_filtering():
    """Verify OQ-02 and OQ-03: valid row kept over NaN duplicate and out-of-window dropped."""
    df_test = pd.DataFrame(
        [
            {
                "city": "Hanoi",
                "station_id": 1001,
                "station_name": "Dup & Out Station",
                "sensor_id": 2001,
                "is_monitor": True,
                "event_time": "2020-01-01T00:00:00Z",
                "pm25_ug_m3": 99.0,
            },
            {
                "city": "Hanoi",
                "station_id": 1001,
                "station_name": "Dup & Out Station",
                "sensor_id": 2001,
                "is_monitor": True,
                "event_time": "2026-11-10T00:00:00Z",
                "pm25_ug_m3": float("nan"),
            },
            {
                "city": "Hanoi",
                "station_id": 1001,
                "station_name": "Dup & Out Station",
                "sensor_id": 2001,
                "is_monitor": True,
                "event_time": "2026-11-10T00:00:00Z",
                "pm25_ug_m3": 35.0,
            },
            {
                "city": "Hanoi",
                "station_id": 1001,
                "station_name": "Dup & Out Station",
                "sensor_id": 2001,
                "is_monitor": True,
                "event_time": "2026-11-11T01:00:00Z",
                "pm25_ug_m3": 40.0,
            },
        ]
    )
    comp, gaps = compute_completeness_and_gaps(
        df_test,
        window_start_utc="2026-11-10T00:00:00Z",
        window_end_utc="2026-11-12T00:00:00Z",
        alive_cutoff_utc="2026-11-01T00:00:00Z",
    )
    row = comp.iloc[0]
    assert row["valid_hours"] == 2
    assert row["nan_rows"] == 0
    assert row["unrecorded_2yr_h"] == 48 - 2
    assert row["num_gaps"] == 1
    assert row["num_gaps_ge_24h"] == 1
    assert row["max_gap_hours"] == 24
    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_hours"] == 24
    pm_s, wx_s = extract_one_week_samples(
        df_test, pd.DataFrame(columns=WEATHER_COLS), usable_ids=[1001]
    )
    assert pm_s.empty
    assert wx_s.empty
