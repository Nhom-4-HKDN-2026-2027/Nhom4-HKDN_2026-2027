from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_USABLE_STATIONS = {3276359, 4946812, 2161292, 4946813, 6123215}


def test_sample_files_exist_and_follow_contracts():
    """Verify 1-week sample CSV files for the 5 usable stations follow CONTRACTS.md."""
    openaq_csv = PROJECT_ROOT / "data" / "sample" / "openaq_pm25_hanoi_hcmc.csv"
    weather_csv = PROJECT_ROOT / "data" / "sample" / "openmeteo_weather_hanoi_hcmc.csv"

    assert openaq_csv.is_file(), "Missing data/sample/openaq_pm25_hanoi_hcmc.csv"
    assert weather_csv.is_file(), "Missing data/sample/openmeteo_weather_hanoi_hcmc.csv"

    df_pm25 = pd.read_csv(openaq_csv)
    df_weather = pd.read_csv(weather_csv)

    assert not df_pm25.empty, "OpenAQ sample CSV is empty"
    assert not df_weather.empty, "Open-Meteo sample CSV is empty"

    # Verify contract columns
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

    # Verify exact 5 usable stations and 1-week window (168 hours per station in weather)
    assert set(df_pm25["station_id"].unique()) == EXPECTED_USABLE_STATIONS
    assert set(df_weather["station_id"].unique()) == EXPECTED_USABLE_STATIONS
    assert len(df_weather) == 5 * 168


def test_deliverable_artifacts_exist():
    """Verify that the notebook and the 1-page data quality report exist."""
    notebook = PROJECT_ROOT / "notebooks" / "01_verify_vn_data.ipynb"
    report = PROJECT_ROOT / "docs" / "data_quality_report_vn.md"

    assert notebook.is_file(), "Missing notebooks/01_verify_vn_data.ipynb"
    assert report.is_file(), "Missing docs/data_quality_report_vn.md"
    assert report.read_text(encoding="utf-8").strip(), "Data quality report is empty"
