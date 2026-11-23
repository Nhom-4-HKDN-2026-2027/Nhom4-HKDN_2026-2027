"""Offline conversion, contract and source-archive tests for Beijing Multi-Site."""

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest import fetch_beijing as bj  # noqa: E402


def source_frame(periods=1, station="Aotizhongxin"):
    dates = pd.date_range("2015-11-15T08:00:00", periods=periods, freq="h")
    return pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "day": dates.day,
            "hour": dates.hour,
            "station": station,
            "PM2.5": 25,
            "TEMP": 20.0,
            "DEWP": 10.0,
            "RAIN": 0.5,
            "WSPM": 2.0,
            "wd": "NNW",
        }
    )


def make_archive(path, periods=168, outer=True):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for station in bj.STATIONS:
            frame = source_frame(periods, station)
            frame.loc[0, "PM2.5"] = np.nan
            archive.writestr(
                f"data/PRSA_Data_{station}_20130301-20170228.csv", frame.to_csv(index=False)
            )
    if outer:
        with ZipFile(path, "w") as archive:
            archive.writestr(bj.INNER_ARCHIVE, buffer.getvalue())
            archive.writestr("data.csv", "UNRELATED; must not be read")
    else:
        path.write_bytes(buffer.getvalue())


def test_contract_units_timezone_and_string_station_id():
    result = bj.normalize_beijing(source_frame())
    assert list(result) == bj.FEATURE_COLUMNS
    assert result.iloc[0].station_id == "beijing_Aotizhongxin"
    assert result.iloc[0].event_time == "2015-11-15T00:00:00Z"
    assert result.iloc[0].wind_direction_deg == 337.5
    assert result.iloc[0].wind_speed_m_s == 2.0
    assert result.iloc[0].precipitation_mm == 0.5
    assert result.iloc[0].relative_humidity_pct == pytest.approx(52.5413, abs=0.001)


def test_humidity_saturation_missing_invalid_and_negative_temperature():
    rh = bj.relative_humidity_from_dewpoint(
        pd.Series([20, 20, np.nan, 20, np.inf, -243.04, -10]),
        pd.Series([20, np.nan, 10, 21, 10, -243.04, -15]),
    )
    assert rh.iloc[0] == 100
    assert rh.iloc[1:6].isna().all()
    assert 0 < rh.iloc[6] < 100


def test_explicit_source_timezone_is_respected():
    result = bj.normalize_beijing(source_frame(), source_timezone="UTC")
    assert result.iloc[0].event_time == "2015-11-15T08:00:00Z"


@pytest.mark.parametrize("direction,degrees", list(bj.COMPASS_DEGREES.items()))
def test_all_compass_directions(direction, degrees):
    frame = source_frame()
    frame["wd"] = " " + direction.lower() + " "
    assert bj.normalize_beijing(frame).iloc[0].wind_direction_deg == degrees


def test_missing_weather_does_not_drop_rows_or_impute_pm25():
    source = source_frame(2)
    source.loc[0, ["PM2.5", "TEMP", "DEWP", "RAIN", "WSPM", "wd"]] = np.nan
    source.loc[1, "PM2.5"] = -1
    result = bj.normalize_beijing(source)
    assert len(result) == 2
    assert result.loc[0, bj.FEATURE_COLUMNS[2:]].isna().all()
    assert result.loc[1, "pm25_ug_m3"] == -1


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "old_dataset",
        "duplicate",
        "calendar",
        "hour",
        "fraction",
        "numeric",
        "direction",
        "station",
    ],
)
def test_reject_bad_source(defect):
    frame = source_frame()
    if defect == "schema":
        frame = frame.drop(columns="DEWP")
    elif defect == "old_dataset":
        frame = frame.rename(columns={"WSPM": "Iws", "wd": "cbwd", "PM2.5": "pm2.5"})
    elif defect == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    elif defect == "calendar":
        frame["month"] = 13
    elif defect == "hour":
        frame["hour"] = 24
    elif defect == "fraction":
        frame["hour"] = 8.5
    elif defect == "numeric":
        frame["TEMP"] = "broken"
    elif defect == "direction":
        frame["wd"] = "variable"
    else:
        frame["station"] = "Unknown"
    with pytest.raises(ValueError):
        bj.normalize_beijing(frame)


@pytest.mark.parametrize("outer", [True, False])
def test_archive_week_all_stations_and_missing_retained(tmp_path, outer):
    path = tmp_path / "source.zip"
    make_archive(path, outer=outer)
    result, metadata = bj.build_beijing_sample(path)
    assert len(result) == 12 * 168
    assert result.groupby("station_id").size().eq(168).all()
    assert result.pm25_ug_m3.isna().sum() == 12
    assert metadata["missing_cells"]["pm25_ug_m3"] == 12
    assert result.event_time.min() == bj.SAMPLE_START_UTC
    assert result.event_time.max() == "2015-11-21T23:00:00Z"


def test_missing_hour_and_nonweekly_window_rejected(tmp_path):
    path = tmp_path / "source.zip"
    make_archive(path, periods=167)
    with pytest.raises(ValueError, match="Missing/extra hours"):
        bj.build_beijing_sample(path)
    with pytest.raises(ValueError, match="seven days"):
        bj.build_beijing_sample(path, end_utc="2015-11-16T00:00:00Z")


def test_download_success_and_reuse_without_network(tmp_path, monkeypatch):
    source = tmp_path / "source.zip"
    make_archive(source)
    response = Mock(content=source.read_bytes())
    get = Mock(return_value=response)
    monkeypatch.setattr(bj, "_get_with_retry", get)
    destination = tmp_path / "raw" / "download.zip"
    bj.fetch_beijing_archive(destination)
    assert destination.read_bytes() == source.read_bytes()
    bj.fetch_beijing_archive(destination)
    get.assert_called_once()


def test_cli_existing_output_fails_before_download(tmp_path, monkeypatch):
    output = tmp_path / "sample.csv"
    output.write_text("existing")
    fetch = Mock(side_effect=AssertionError("network must not be called"))
    monkeypatch.setattr(bj, "fetch_beijing_archive", fetch)
    monkeypatch.setattr(sys, "argv", ["beijing", "--output", str(output)])
    with pytest.raises(FileExistsError):
        bj.main()
    fetch.assert_not_called()


def test_sample_handoff_matches_shared_contract():
    path = Path(__file__).resolve().parents[1] / "data/sample/beijing_multisite_features.csv"
    frame = pd.read_csv(path)
    assert list(frame) == bj.FEATURE_COLUMNS
    assert frame.station_id.nunique() == 12
    assert frame.groupby("station_id").size().eq(168).all()
    assert not frame.duplicated(["station_id", "event_time"]).any()
    assert frame.relative_humidity_pct.dropna().between(0, 100).all()
    assert frame.wind_direction_deg.dropna().isin(bj.COMPASS_DEGREES.values()).all()
