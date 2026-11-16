"""Offline regressions for ingestion and reproducible data-quality outputs."""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest import openaq_fetcher as aq  # noqa: E402
from ingest import openmeteo_fetcher as wx  # noqa: E402
from ingest import verify_vn_data as pipeline  # noqa: E402


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


def test_no_silent_hardcoded_sample_fallback():
    pm = pm_frame([20], ["2025-11-16T00:00:00Z"])
    pm["station_id"] = pipeline.USABLE_STATION_IDS[0]
    sample, _ = pipeline.extract_one_week_samples(pm, pd.DataFrame())
    assert sample.empty


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


def cached_frames():
    pm = pm_frame([20.0] * 17520, pd.date_range(pipeline.WINDOW_START_UTC, periods=17520, freq="h"))
    weather = pm[
        ["city", "station_id", "station_name", "latitude", "longitude", "event_time"]
    ].copy()
    for col in wx.WEATHER_COLS[6:]:
        weather[col] = 1.0
    return station_frame(), pm, weather


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
