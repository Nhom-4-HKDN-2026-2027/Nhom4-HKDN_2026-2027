"""Prepare historical benchmarks without changing their observation order."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

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
BEIJING_SOURCE = "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"


def write_manifest(directory, frame, source_url):
    """Record schema and checksums of each Parquet object."""
    manifest = {
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


def prepare_beijing(source, output):
    """Convert the PR #6 contract sample to one UTC Parquet file per station."""
    frame = pd.read_csv(source, dtype={"station_id": "string"})
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
    frame["event_time"] = pd.to_datetime(frame.event_time, utc=True, format="mixed")
    if frame.event_time.isna().any() or frame.duplicated(["station_id", "event_time"]).any():
        raise ValueError("Missing timestamps or duplicate station/time observations")
    frame[BEIJING_COLUMNS[2:]] = frame[BEIJING_COLUMNS[2:]].astype("float64")
    directory = Path(output) / "beijing_pm25"
    for station, rows in frame.groupby("station_id", sort=False):
        path = directory / f"station_id={station}" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path, index=False, compression="snappy")
    write_manifest(directory, frame, BEIJING_SOURCE)
    return directory


def prepare_elec2(output, samples=None):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Prepare and upload both datasets")
    parser.add_argument("--to-parquet", action="store_true", help="Prepare local Parquet files")
    parser.add_argument(
        "--download", action="store_true", help="Fetch Elec2 via River and prepare data"
    )
    parser.add_argument("--upload-minio", action="store_true", help="Upload prepared files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing remote objects")
    parser.add_argument(
        "--beijing-csv", type=Path, default=Path("data/sample/beijing_multisite_features.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if not any((args.all, args.to_parquet, args.download, args.upload_minio)):
        parser.error("Choose --all, --to-parquet, --download or --upload-minio")
    if args.all or args.to_parquet or args.download:
        prepare_beijing(args.beijing_csv, args.output_dir)
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
