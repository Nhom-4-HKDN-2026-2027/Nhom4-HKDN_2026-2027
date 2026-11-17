"""Validation of pipeline configuration values and TOML keys."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.pipeline_config import DEFAULT_CONFIG, load_config  # noqa: E402


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
