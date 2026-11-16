"""Download UCI Beijing Multi-Site and derive a contract-aligned weekly sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ingest.config import PROJECT_ROOT
from ingest.feature_schema import FEATURE_COLUMNS
from ingest.openaq_fetcher import _get_with_retry, _validate_window

logger = logging.getLogger(__name__)
SOURCE_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
SOURCE_DOI = "10.24432/C5RK5G"
INNER_ARCHIVE = "PRSA2017_Data_20130301-20170228.zip"
STATIONS = (
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
)
SOURCE_COLUMNS = ["year", "month", "day", "hour", "station", "PM2.5", "TEMP", "DEWP", "WSPM", "wd"]
COMPASS_DEGREES = {
    name: i * 22.5
    for i, name in enumerate("N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split())
}
SAMPLE_START_UTC = "2015-11-15T00:00:00Z"
SAMPLE_END_UTC = "2015-11-22T00:00:00Z"


def relative_humidity_from_dewpoint(temperature: pd.Series, dewpoint: pd.Series) -> pd.Series:
    """Magnus approximation over liquid water; T and Td in Celsius, RH in percent.

    RH = 100 exp(17.625 Td/(243.04+Td) - 17.625 T/(243.04+T)).
    Missing/nonfinite inputs, singular denominators and Td > T yield NaN, not clipping.
    See docs/beijing_multisite.md for the formula reference and approximation limits.
    """
    t = pd.to_numeric(temperature, errors="coerce").astype(float)
    td = pd.to_numeric(dewpoint, errors="coerce").astype(float)
    valid = np.isfinite(t) & np.isfinite(td) & (t > -243.04) & (td > -243.04) & (td <= t)
    result = pd.Series(np.nan, index=t.index, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result.loc[valid] = 100 * np.exp(
            17.625 * td[valid] / (243.04 + td[valid]) - 17.625 * t[valid] / (243.04 + t[valid])
        )
    return result.where(np.isfinite(result) & result.between(0, 100))


def normalize_beijing(frame: pd.DataFrame, source_timezone: str = "Asia/Shanghai") -> pd.DataFrame:
    """Map source rows to the seven shared contract columns, preserving missing rows."""
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Beijing source missing columns: {sorted(missing)}")
    ZoneInfo(source_timezone)
    dates = frame[["year", "month", "day", "hour"]].apply(pd.to_numeric, errors="coerce")
    if (
        dates.isna().any().any()
        or not np.isfinite(dates).all().all()
        or (dates % 1 != 0).any().any()
    ):
        raise ValueError("Calendar fields must be finite integers")
    if not dates["hour"].between(0, 23).all():
        raise ValueError("Source hour must be in 0..23")
    timestamps = (
        pd.to_datetime(dates, errors="raise")
        .dt.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
        .dt.tz_convert("UTC")
    )
    station = frame["station"].astype("string").str.strip()
    if station.isna().any() or not station.isin(STATIONS).all():
        raise ValueError("Unknown or missing Beijing Multi-Site station")
    numeric = frame[["PM2.5", "TEMP", "DEWP", "WSPM"]].apply(pd.to_numeric, errors="coerce")
    invalid_tokens = frame[numeric.columns].notna() & numeric.isna()
    if invalid_tokens.any().any():
        raise ValueError("Nonnumeric source value; only explicit missing values are allowed")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).astype(float)
    directions = frame["wd"].astype("string").str.strip().str.upper()
    degrees = directions.map(COMPASS_DEGREES).astype(float)
    unknown = directions.notna() & ~directions.isin(COMPASS_DEGREES)
    if unknown.any():
        raise ValueError(f"Unknown wind directions: {sorted(directions[unknown].unique())}")
    result = pd.DataFrame(
        {
            "station_id": "beijing_" + station,
            "event_time": timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pm25_ug_m3": numeric["PM2.5"],
            "temperature_c": numeric["TEMP"],
            "relative_humidity_pct": relative_humidity_from_dewpoint(
                numeric["TEMP"], numeric["DEWP"]
            ),
            "wind_speed_m_s": numeric["WSPM"],
            "wind_direction_deg": degrees,
        },
        columns=FEATURE_COLUMNS,
    )
    if result.duplicated(["station_id", "event_time"]).any():
        raise ValueError("Duplicate station/time keys; source rows must be reviewed")
    # Preserve negative source measurements for auditing; do not silently drop or impute them.
    return result.sort_values(["station_id", "event_time"]).reset_index(drop=True)


def build_beijing_sample(
    archive_path: Path,
    start_utc: str = SAMPLE_START_UTC,
    end_utc: str = SAMPLE_END_UTC,
    source_timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, dict]:
    """Read only the 12 official station members, never extract archive paths to disk."""
    start, end = _validate_window(start_utc, end_utc)
    if end - start != pd.Timedelta(days=7):
        raise ValueError("The handoff sample must span exactly seven days")
    frames, source_counts, source_hashes = [], {}, {}
    with ZipFile(archive_path) as outer:
        # Support either the UCI outer download or its original PRSA2017 nested archive.
        inner_bytes = (
            outer.read(INNER_ARCHIVE)
            if INNER_ARCHIVE in outer.namelist()
            else archive_path.read_bytes()
        )
    with ZipFile(BytesIO(inner_bytes)) as archive:
        for station in STATIONS:
            basename = f"PRSA_Data_{station}_20130301-20170228.csv"
            members = [
                info for info in archive.infolist() if PurePosixPath(info.filename).name == basename
            ]
            if len(members) != 1:
                raise ValueError(f"Expected one member for station {station}")
            content = archive.read(members[0])
            source = pd.read_csv(BytesIO(content), na_values=["NA"])
            if not source["station"].eq(station).all():
                raise ValueError(f"Station identity mismatch in {basename}")
            source_counts[station] = len(source)
            source_hashes[basename] = hashlib.sha256(content).hexdigest()
            normalized = normalize_beijing(source, source_timezone)
            times = pd.to_datetime(normalized.event_time, utc=True)
            sample = normalized[(times >= start) & (times < end)]
            actual = pd.DatetimeIndex(pd.to_datetime(sample.event_time, utc=True))
            expected = pd.date_range(start, end, inclusive="left", freq="h")
            if not actual.equals(expected):
                raise ValueError(f"Missing/extra hours for {station} in the selected week")
            frames.append(sample)
    result = pd.concat(frames, ignore_index=True)
    metadata = {
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "doi": SOURCE_DOI,
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "source_member_sha256": source_hashes,
        "source_rows_per_station": source_counts,
        "source_timezone": source_timezone,
        "timezone_status": "Assumption: source calendar fields are local Beijing time; UCI page omits timezone",
        "sample_start_utc": start.isoformat(),
        "sample_end_utc": end.isoformat(),
        "sample_rows": len(result),
        "stations": list(STATIONS),
        "columns": FEATURE_COLUMNS,
        "rh_formula": "100*exp(17.625*Td/(243.04+Td)-17.625*T/(243.04+T)); Celsius; liquid water",
        "rh_invalid_policy": "Missing/nonfinite inputs, Td>T, or denominators<=0 produce NaN; no clipping",
        "wind_direction_mapping": COMPASS_DEGREES,
        "wind_direction_convention": "from, clockwise from north; N=0",
        "missing_cells": {col: int(result[col].isna().sum()) for col in FEATURE_COLUMNS},
        "negative_pm25_rows": int((result.pm25_ug_m3 < 0).sum()),
        "negative_wind_speed_rows": int((result.wind_speed_m_s < 0).sum()),
    }
    return result, metadata


def fetch_beijing_archive(destination: Path) -> Path:
    """Download the public UCI archive; reuse an existing archive without overwriting."""
    if destination.is_file():
        return destination
    response = _get_with_retry(SOURCE_URL, {}, {}, timeout=90)
    # Validate before persisting an HTTP-success HTML/error response.
    with ZipFile(BytesIO(response.content)) as archive:
        if INNER_ARCHIVE not in archive.namelist():
            raise ValueError("UCI download missing the PRSA2017 Multi-Site archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(response.content)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=PROJECT_ROOT / "data/raw/beijing_multisite/uci_501.zip"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/sample/beijing_multisite_features.csv"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data/raw/beijing_multisite/sample_manifest.json",
    )
    parser.add_argument("--start", default=SAMPLE_START_UTC)
    parser.add_argument("--end", default=SAMPLE_END_UTC)
    parser.add_argument("--source-timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    for path in [args.output, args.metadata]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite {path}; choose new --output and --metadata paths"
            )
    if args.output.resolve() == args.metadata.resolve() or args.archive.resolve() in {
        args.output.resolve(),
        args.metadata.resolve(),
    }:
        raise ValueError("Archive, output and metadata paths must be distinct")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    archive = fetch_beijing_archive(args.archive)
    sample, metadata = build_beijing_sample(archive, args.start, args.end, args.source_timezone)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False, mode="x", float_format="%.8f")
    metadata["sample_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    with args.metadata.open("x") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info(
        "Saved %d rows, %d stations: %s", len(sample), sample.station_id.nunique(), args.output
    )
    logger.info("Missing cells retained: %s", metadata["missing_cells"])


if __name__ == "__main__":
    main()
