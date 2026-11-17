"""Shared downstream schema checks for Vietnam and Beijing samples."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingest.feature_contract import (  # noqa: E402
    FEATURE_COLUMNS,
    build_vietnam_features,
    normalize_feature_table,
)


def feature_row():
    return pd.DataFrame(
        {
            "station_id": [123],
            "event_time": ["2025-11-15T00:00:00Z"],
            **{col: [1] for col in FEATURE_COLUMNS[2:]},
        }
    )


def test_actual_sources_have_identical_downstream_schema():
    pm = pd.read_csv(ROOT / "data/sample/openaq_pm25_hanoi_hcmc.csv")
    weather = pd.read_csv(ROOT / "data/sample/openmeteo_weather_hanoi_hcmc.csv")
    beijing = pd.read_csv(ROOT / "data/sample/beijing_multisite_features.csv")
    vn = build_vietnam_features(pm, weather)
    bj = normalize_feature_table(beijing)
    assert list(vn) == list(bj) == FEATURE_COLUMNS
    assert vn.dtypes.astype(str).to_dict() == bj.dtypes.astype(str).to_dict()
    assert str(vn.station_id.dtype) == "string"
    assert str(vn.event_time.dt.tz) == str(bj.event_time.dt.tz) == "UTC"
    assert len(vn) == len(pm) == 815
    assert len(bj) == 2016
    assert bj.pm25_ug_m3.isna().sum() == beijing.pm25_ug_m3.isna().sum()


def test_join_mixed_id_types_and_missing_weather_preserves_pm25(caplog):
    frame = feature_row()
    weather = frame[["station_id", "event_time", *FEATURE_COLUMNS[3:]]].copy()
    weather["station_id"] = "123"
    joined = build_vietnam_features(frame[FEATURE_COLUMNS[:3]], weather)
    assert joined.iloc[0].temperature_c == 1.0
    joined = build_vietnam_features(frame[FEATURE_COLUMNS[:3]], weather.iloc[:0])
    assert len(joined) == 1
    assert joined[FEATURE_COLUMNS[3:]].isna().all().all()
    assert "no matching weather" in caplog.text


def test_missing_numeric_values_and_negative_source_values_are_preserved():
    frame = feature_row()
    frame["temperature_c"] = np.nan
    frame["pm25_ug_m3"] = -1
    result = normalize_feature_table(frame)
    assert pd.isna(result.iloc[0].temperature_c)
    assert result.iloc[0].pm25_ug_m3 == -1.0


@pytest.mark.parametrize(
    "defect",
    [
        "naive",
        "bad_time",
        "subhour",
        "missing_id",
        "empty_id",
        "duplicate",
        "schema",
        "numeric",
        "inf",
    ],
)
def test_contract_defects_are_rejected(defect):
    frame = feature_row()
    if defect == "naive":
        frame["event_time"] = "2025-11-15T00:00:00"
    elif defect == "bad_time":
        frame["event_time"] = "bad"
    elif defect == "subhour":
        frame["event_time"] = "2025-11-15T00:15:00Z"
    elif defect == "missing_id":
        frame["station_id"] = None
    elif defect == "empty_id":
        frame["station_id"] = " "
    elif defect == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    elif defect == "schema":
        frame = frame.drop(columns="pm25_ug_m3")
    elif defect == "numeric":
        frame["temperature_c"] = "bad"
    else:
        frame["wind_speed_m_s"] = np.inf
    with pytest.raises(ValueError):
        normalize_feature_table(frame)
