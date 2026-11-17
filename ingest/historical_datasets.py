"""Prepare historical benchmarks without changing their observation order."""

import argparse
import hashlib
import io
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ingest.minio_storage import create_client, ensure_buckets, upload_historical_datasets

BEIJING_COLUMNS = [
    "station_id",
    "event_time",
    "pm25_ug_m3",
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_m_s",
    "wind_direction_deg",
]
BEIJING_SOURCE = "https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set"
BEIJING_UPSTREAM = "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
BEIJING_DOI = "10.24432/C5RK5G"
BEIJING_DOWNLOAD = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "sid321axn/beijing-multisite-airquality-data-set?datasetVersionNumber=1"
)
COMPASS_DEGREES = {
    name: i * 22.5
    for i, name in enumerate("N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split())
}


def write_manifest(directory: Path, frame: pd.DataFrame, source_url: str, **metadata) -> None:
    """Record schema and checksums of each Parquet object."""
    manifest = {
        **metadata,
        "dataset_name": directory.name,
        "source_url": source_url,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "row_count": len(frame),
        "columns": list(frame.columns),
        "schema_dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "sha256_checksum": {
            path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*.parquet"))
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def download_beijing(destination: Path) -> Path:
    """Cache the pinned public Kaggle archive, validating before publishing it locally."""
    if destination.is_file():
        with ZipFile(destination) as archive:
            if len([name for name in archive.namelist() if name.endswith(".csv")]) != 12:
                raise ValueError("Expected 12 Beijing station CSV files")
        return destination
    with urlopen(BEIJING_DOWNLOAD, timeout=90) as response:
        content = response.read(64 * 1024 * 1024 + 1)
    if len(content) > 64 * 1024 * 1024:
        raise ValueError("Beijing download exceeds the expected archive size")
    with ZipFile(io.BytesIO(content)) as archive:
        if len([name for name in archive.namelist() if name.endswith(".csv")]) != 12:
            raise ValueError("Expected 12 Beijing station CSV files")
        if archive.testzip() is not None:
            raise ValueError("Corrupted Beijing archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def normalize_beijing(frame: pd.DataFrame, source_timezone: str) -> pd.DataFrame:
    """Normalize raw Beijing observations to the seven-column feature contract."""
    dates = frame[["year", "month", "day", "hour"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(dates).all().all() or (dates % 1 != 0).any().any():
        raise ValueError("Calendar fields must be finite integers")
    if not dates.hour.between(0, 23).all():
        raise ValueError("Source hour must be in 0..23")
    timestamps = pd.to_datetime(dates).dt.tz_localize(source_timezone).dt.tz_convert("UTC")
    numeric = frame[["PM2.5", "TEMP", "DEWP", "WSPM"]].apply(pd.to_numeric, errors="raise")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).astype("float64")
    temperature, dewpoint = numeric.TEMP, numeric.DEWP
    valid = (temperature > -243.04) & (dewpoint > -243.04) & (dewpoint <= temperature)
    # Missing or physically invalid inputs remain missing; do not impute or clip RH.
    humidity = pd.Series(np.nan, index=frame.index, dtype="float64")
    humidity.loc[valid] = 100 * np.exp(
        17.625 * dewpoint[valid] / (243.04 + dewpoint[valid])
        - 17.625 * temperature[valid] / (243.04 + temperature[valid])
    )
    directions = frame.wd.astype("string").str.strip().str.upper()
    if (directions.notna() & ~directions.isin(COMPASS_DEGREES)).any():
        raise ValueError("Unknown wind direction")
    return pd.DataFrame(
        {
            "station_id": "beijing_" + frame.station.astype("string"),
            "event_time": timestamps,
            "pm25_ug_m3": numeric["PM2.5"],
            "temperature_c": temperature,
            "relative_humidity_pct": humidity.where(
                np.isfinite(humidity) & humidity.between(0, 100)
            ),
            "wind_speed_m_s": numeric.WSPM,
            "wind_direction_deg": directions.map(COMPASS_DEGREES).astype("float64"),
        }
    )


def prepare_beijing(source: Path, output: Path, source_timezone: str = "Asia/Shanghai") -> Path:
    """Convert a full Kaggle ZIP or a contract CSV into UTC year/station partitions."""
    source, output = Path(source), Path(output)
    metadata = {}
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            members = sorted(name for name in archive.namelist() if name.endswith(".csv"))
            if not members:
                raise ValueError("Beijing archive contains no station CSV files")
            frames = []
            for name in members:
                # Read archive members directly; never extract untrusted member paths.
                with archive.open(name) as member:
                    frames.append(normalize_beijing(pd.read_csv(member), source_timezone))
            frame = pd.concat(frames, ignore_index=True)
        source_url = BEIJING_SOURCE
        metadata = {
            "source_version": 1,
            "source_page": BEIJING_UPSTREAM,
            "source_doi": BEIJING_DOI,
            "source_archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_timezone": source_timezone,
            "timezone_note": "Assumes source calendar fields use local Beijing time; the source omits timezone",
        }
    else:
        frame = pd.read_csv(source, dtype={"station_id": "string"})
        source_url = "data/sample/beijing_multisite_features.csv (contract CSV override)"
    if list(frame.columns) != BEIJING_COLUMNS or frame.empty:
        raise ValueError(
            "Beijing input must contain the seven contract columns and at least one row"
        )
    if (
        frame.station_id.isna().any()
        or not frame.station_id.map(
            lambda value: bool(re.fullmatch(r"[A-Za-z0-9_-]+", value))
        ).all()
    ):
        raise ValueError("Invalid station_id")
    if any(pd.isna(value) or pd.Timestamp(value).tzinfo is None for value in frame.event_time):
        raise ValueError("event_time must contain timezone-aware timestamps")
    frame["event_time"] = pd.to_datetime(frame.event_time, utc=True, format="mixed")
    if frame.duplicated(["station_id", "event_time"]).any():
        raise ValueError("Duplicate station/time observations")
    frame[BEIJING_COLUMNS[2:]] = frame[BEIJING_COLUMNS[2:]].astype("float64")
    frame = frame.sort_values(["station_id", "event_time"]).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    directory = output / "beijing_pm25"
    if source.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Source must be outside the generated dataset directory")
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        staged = Path(temporary) / directory.name
        for (year, station), rows in frame.groupby([frame.event_time.dt.year, "station_id"]):
            path = staged / f"year={year}" / f"station_id={station}" / "data.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(rows, preserve_index=False)
            # Hive readers infer int32 dictionary partition keys; match that encoding.
            schema = table.schema.set(
                0, pa.field("station_id", pa.dictionary(pa.int32(), pa.string()))
            )
            pq.write_table(table.cast(schema), path, compression="snappy")
        write_manifest(
            staged,
            frame,
            source_url,
            **metadata,
            years=sorted(int(year) for year in frame.event_time.dt.year.unique()),
            station_count=int(frame.station_id.nunique()),
            time_range_utc={
                "start": frame.event_time.min().isoformat(),
                "end": frame.event_time.max().isoformat(),
            },
        )
        # Replace only generated local output, after every new partition is prepared.
        if directory.exists():
            shutil.rmtree(directory)
        staged.replace(directory)
    return directory


def prepare_elec2(output: Path, samples=None) -> Path:
    """Preserve River's normalized period values and prequential sample order."""
    if samples is None:
        from river.datasets import Elec2

        samples = Elec2()
    frame = pd.DataFrame([{**features, "target": target} for features, target in samples])
    if frame.empty:
        raise ValueError("Elec2 input must not be empty")
    for column in frame.columns:
        frame[column] = frame[column].astype(
            "bool" if column == "target" else "int64" if column == "day" else "float64"
        )
    frame.insert(0, "sample_idx", pd.Series(range(len(frame)), dtype="int64"))
    directory = Path(output) / "elec2"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "elec2.parquet", index=False, compression="snappy")
    write_manifest(directory, frame, "https://riverml.xyz/0.22.0/api/datasets/Elec2/")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Prepare and upload both datasets")
    parser.add_argument("--to-parquet", action="store_true", help="Prepare local Parquet files")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the pinned Kaggle archive and prepare both datasets",
    )
    parser.add_argument("--upload-minio", action="store_true", help="Upload prepared files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing remote objects")
    parser.add_argument(
        "--beijing-csv", type=Path, help="Optional contract CSV override for samples/tests"
    )
    parser.add_argument("--beijing-archive", type=Path, help="Use an existing Kaggle ZIP")
    parser.add_argument("--source-timezone", default="Asia/Shanghai")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if not any((args.all, args.to_parquet, args.download, args.upload_minio)):
        parser.error("Choose --all, --to-parquet, --download or --upload-minio")
    if args.all or args.to_parquet or args.download:
        if args.beijing_csv and args.beijing_archive:
            parser.error("Choose only one of --beijing-csv or --beijing-archive")
        source = (
            args.beijing_csv
            or args.beijing_archive
            or download_beijing(args.output_dir / "downloads" / "beijing_multisite_v1.zip")
        )
        prepare_beijing(source, args.output_dir, args.source_timezone)
        prepare_elec2(args.output_dir)
    if args.all or args.upload_minio:
        directories = [args.output_dir / name for name in ("beijing_pm25", "elec2")]
        if not all((directory / "manifest.json").is_file() for directory in directories):
            parser.error("Prepare both datasets with --to-parquet before uploading")
        client = create_client()
        ensure_buckets(client)
        uploaded = upload_historical_datasets(client, directories, force=args.force)
        print(f"Uploaded {len(uploaded)} objects to raw; existing objects skipped unless --force")


if __name__ == "__main__":
    main()
