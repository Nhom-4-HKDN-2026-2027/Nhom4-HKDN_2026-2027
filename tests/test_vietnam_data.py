"""Offline OpenAQ/Open-Meteo fetching, completeness, gaps and station lifecycle tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest import fetch_openaq as aq  # noqa: E402
from ingest import fetch_openmeteo as wx  # noqa: E402
from ingest.fetch_openaq import (  # noqa: E402
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
from ingest.fetch_openmeteo import (  # noqa: E402
    EXPECTED_UNITS,
    WEATHER_COLS,
    fetch_station_hourly_weather,
    fetch_usable_stations_weather_2yr,
)
from ingest.verify_vietnam_data import (  # noqa: E402
    extract_one_week_samples,
)


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


def test_empty_dataframes_always_preserve_schema():
    """Verify Checklist #4: all functions returning empty DataFrames preserve exact column schemas."""
    with patch("ingest.fetch_openaq._get_with_retry") as mock_get:
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
    with patch("ingest.fetch_openaq._get_with_retry") as mock_get:
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


def test_openmeteo_expected_hours_and_timestamp_validation():
    """Verify Checklist #7: Open-Meteo validates expected hour count, UTC grid, and NaN values."""
    with patch("ingest.fetch_openmeteo._get_with_retry") as mock_get:
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


@pytest.mark.parametrize("value", [np.nan, -1, np.inf, -np.inf])
def test_no_valid_hours(value):
    summary, gaps = aq.compute_completeness_and_gaps(pm_frame([value]), include_boundary_gaps=True)
    assert summary.iloc[0].valid_hours == 0
    assert summary.iloc[0].completeness_2yr_pct == 0
    assert gaps.gap_hours.sum() == 17520


def test_unknown_metadata_is_fetched(monkeypatch):
    fetch = Mock(return_value=[])
    monkeypatch.setattr(aq, "fetch_sensor_hourly_pm25", fetch)
    aq.fetch_all_active_stations_2yr(station_frame(), "dummy")
    fetch.assert_called_once()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_never_retry(monkeypatch, status):
    response = requests.Response()
    response.status_code = status
    get = Mock(return_value=response)
    monkeypatch.setattr(aq.requests, "get", get)
    with pytest.raises(requests.HTTPError):
        aq._get_with_retry("https://api.openaq.org/v3/test", {}, {})
    assert get.call_count == 1


@pytest.mark.parametrize("error", [requests.Timeout(), requests.ConnectionError(), 503, 429])
def test_transient_retry(monkeypatch, error):
    success = requests.Response()
    success.status_code = 200
    if isinstance(error, int):
        response = requests.Response()
        response.status_code = error
        response.headers["Retry-After"] = "3"
        error = response
    get = Mock(side_effect=[error, success])
    sleep = Mock()
    monkeypatch.setattr(aq.requests, "get", get)
    monkeypatch.setattr(aq.time, "sleep", sleep)
    assert aq._get_with_retry("https://api.openaq.org/v3/test", {}, {}) is success
    assert get.call_count == 2
    sleep.assert_called_once()


def test_multiple_sensors_uses_actual_window_hours_once(monkeypatch):
    stations = station_frame()
    stations.attrs["sensor_candidates"] = {1: [10, 20]}
    items = {
        10: [
            {"event_time": f"2025-01-01T0{i}:00:00Z", "pm25_ug_m3": 20, "percent_coverage": 100}
            for i in range(3)
        ],
        20: [{"event_time": "2026-11-13T23:00:00Z", "pm25_ug_m3": 20, "percent_coverage": 100}],
    }
    fetch = Mock(side_effect=lambda sensor_id, **kwargs: items[sensor_id])
    monkeypatch.setattr(aq, "fetch_sensor_hourly_pm25", fetch)
    result = aq.fetch_all_active_stations_2yr(stations, "dummy")
    assert set(result.sensor_id) == {10}
    assert result.attrs["sensor_scores"] == {1: {10: 3, 20: 1}}
    assert fetch.call_count == 2


def test_missing_sensor_and_failed_fetch_remain_unknown(monkeypatch):
    stations = station_frame()
    stations.loc[0, "sensor_id"] = np.nan
    result = aq.fetch_all_active_stations_2yr(stations, "dummy")
    assert result.attrs["fetch_status"] == {1: "not_fetched"}
    monkeypatch.setattr(aq, "fetch_sensor_hourly_pm25", Mock(side_effect=requests.Timeout()))
    result = aq.fetch_all_active_stations_2yr(station_frame(), "dummy")
    assert result.attrs["fetch_status"] == {1: "fetch_failed"}
    life = aq.classify_station_lifecycle(station_frame(), result).set_index("lifecycle_status")
    assert life.loc[aq.LIFECYCLE_CATEGORIES[5], "Hanoi"] == 1
    assert life.loc[aq.LIFECYCLE_CATEGORIES[2], "Total"] == 0


def test_empty_success_separate_from_not_fetched(monkeypatch):
    monkeypatch.setattr(aq, "fetch_sensor_hourly_pm25", Mock(return_value=[]))
    result = aq.fetch_all_active_stations_2yr(station_frame(), "dummy")
    life = aq.classify_station_lifecycle(station_frame(), result).set_index("lifecycle_status")
    assert life.loc[aq.LIFECYCLE_CATEGORIES[4], "Total"] == 1


def test_mixed_sensors_cannot_be_silently_combined():
    frame = pm_frame([10, 20])
    frame.loc[1, "sensor_id"] = 11
    with pytest.raises(ValueError, match="Multiple sensors"):
        aq.compute_completeness_and_gaps(frame)


def test_retry_exhaustion_and_openaq_reset(monkeypatch):
    response = requests.Response()
    response.status_code = 429
    response.headers["x-ratelimit-reset"] = "180"
    get, sleep = Mock(return_value=response), Mock()
    monkeypatch.setattr(aq.requests, "get", get)
    monkeypatch.setattr(aq.time, "sleep", sleep)
    with pytest.raises(requests.HTTPError):
        aq._get_with_retry(aq.OPENAQ_BASE_URL + "/locations", {}, {}, max_retries=2)
    assert get.call_count == 2
    sleep.assert_called_once_with(180)


def test_retry_after_http_date(monkeypatch):
    response, success = requests.Response(), requests.Response()
    response.status_code, success.status_code = 429, 200
    response.headers["Retry-After"] = "Thu, 01 Jan 1970 00:01:00 GMT"
    monkeypatch.setattr(aq.requests, "get", Mock(side_effect=[response, success]))
    monkeypatch.setattr(aq.time, "time", lambda: 10)
    sleep = Mock()
    monkeypatch.setattr(aq.time, "sleep", sleep)
    aq._get_with_retry(wx.OPENMETEO_ARCHIVE_URL, {}, {})
    sleep.assert_called_once_with(50)


def weather_payload():
    return {
        "hourly_units": dict(wx.EXPECTED_UNITS),
        "utc_offset_seconds": 0,
        "hourly": {
            "time": pd.date_range("2025-11-15", periods=24, freq="h")
            .strftime("%Y-%m-%dT%H:%M")
            .tolist(),
            **{name: [1.0] * 24 for name in wx.HOURLY_VARIABLES_LIST},
        },
    }


def fetch_weather(monkeypatch, payload, **kwargs):
    response = Mock()
    response.json.return_value = payload
    get = Mock(return_value=response)
    monkeypatch.setattr(wx, "_get_with_retry", get)
    return wx.fetch_station_hourly_weather(
        1,
        "Test",
        "Hanoi",
        21,
        105.8,
        start_date="2025-11-15",
        end_date="2025-11-15",
        **kwargs,
    ), get


@pytest.mark.parametrize(
    "defect", ["duplicate", "unsorted", "subhour", "units", "offset", "length"]
)
def test_weather_rejects_structural_defects(monkeypatch, defect):
    payload = weather_payload()
    times = payload["hourly"]["time"]
    if defect == "duplicate":
        times[1] = times[0]
    elif defect == "unsorted":
        times[0], times[1] = times[1], times[0]
    elif defect == "subhour":
        times[0] = "2025-11-15T00:15"
    elif defect == "units":
        payload["hourly_units"]["wind_speed_10m"] = "km/h"
    elif defect == "offset":
        payload["utc_offset_seconds"] = 25200
    else:
        payload["hourly"]["temperature_2m"].pop()
    with pytest.raises(ValueError):
        fetch_weather(monkeypatch, payload, allow_nan=True)


def test_weather_missing_hours_and_values_retained(monkeypatch):
    payload = weather_payload()
    for values in payload["hourly"].values():
        values.pop(3)
    payload["hourly"]["temperature_2m"][0] = None
    frame, get = fetch_weather(monkeypatch, payload, allow_nan=True)
    assert len(frame) == 24
    assert frame.loc[3, wx.WEATHER_COLS[6:]].isna().all()
    assert frame.temperature_c.isna().sum() == 2
    assert get.call_args.kwargs["params"]["models"] == "era5"
    assert frame.attrs["missing_values"] == 6


def test_inventory_success_keeps_candidates_without_extra_requests(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "results": [
            {
                "id": 1,
                "name": "Test",
                "locality": "Hanoi",
                "coordinates": {"latitude": 21, "longitude": 105.8},
                "sensors": [
                    {"id": sid, "parameter": {"id": 2, "name": "pm25"}} for sid in [10, 20]
                ],
            }
        ]
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(aq, "_get_with_retry", get)
    stations = aq.fetch_vietnam_pm25_stations("dummy")
    assert stations.attrs["sensor_candidates"] == {1: [10, 20]}
    assert len(stations) == 1
    assert get.call_count == 1
    assert get.call_args.kwargs["params"]["iso"] == "VN"


def test_hourly_api_pagination_and_half_open_window(monkeypatch):
    def item(time):
        return {
            "value": 25,
            "period": {"datetimeFrom": {"utc": time}},
            "parameter": {"id": 2, "units": "µg/m³"},
            "coverage": {"percentComplete": 100},
        }

    page1, page2 = Mock(), Mock()
    page1.json.return_value = {"results": [item("2024-11-14T00:00:00Z")] * 1000}
    page2.json.return_value = {"results": [item("2026-11-14T00:00:00Z")]}
    get = Mock(side_effect=[page1, page2])
    monkeypatch.setattr(aq, "_get_with_retry", get)
    monkeypatch.setattr(aq.time, "sleep", Mock())
    result = aq.fetch_sensor_hourly_pm25(10, "dummy")
    assert len(result) == 1000
    assert get.call_args_list[1].kwargs["params"]["page"] == 2
    assert result[0]["pm25_ug_m3"] == 25


@pytest.mark.parametrize("payload", [{}, {"results": None}, {"results": [{"period": None}]}])
def test_malformed_hourly_response_is_not_no_data(monkeypatch, payload):
    response = Mock()
    response.json.return_value = payload
    monkeypatch.setattr(aq, "_get_with_retry", Mock(return_value=response))
    with pytest.raises(ValueError):
        aq.fetch_sensor_hourly_pm25(10, "dummy")


def test_outlier_remains_observed_but_disqualifies_demo():
    summary, _ = aq.compute_completeness_and_gaps(pm_frame([501]), max_missing_2yr_pct=100)
    assert summary.iloc[0].valid_hours == 1
    assert summary.iloc[0].gt500_count == 1
    assert not summary.iloc[0].usable_demo


def test_weather_archive_delay_retains_unavailable_hours(monkeypatch):
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    fetch = Mock(side_effect=AssertionError("No API request for unavailable archive day"))
    monkeypatch.setattr(wx, "_get_with_retry", fetch)
    result = wx.fetch_station_hourly_weather(
        1, "Test", "Hanoi", 21, 105.8, start_date=today, end_date=today, allow_nan=True
    )
    assert len(result) == 24
    assert result[wx.WEATHER_COLS[6:]].isna().all().all()
    fetch.assert_not_called()


def test_empty_weather_response_reindexed_when_allowed(monkeypatch):
    payload = weather_payload()
    payload["hourly"] = {k: [] for k in payload["hourly"]}
    result, _ = fetch_weather(monkeypatch, payload, allow_nan=True)
    assert len(result) == 24
    assert result[wx.WEATHER_COLS[6:]].isna().all().all()
