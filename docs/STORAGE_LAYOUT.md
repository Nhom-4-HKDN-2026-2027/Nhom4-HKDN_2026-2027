# Historical datasets and MinIO storage

## Quick start

Run from the repository root:

```bash
cp .env.example .env
docker compose up -d
docker compose ps -a
docker compose logs minio-init
```

MinIO Console: http://localhost:9001. S3 API: http://localhost:9000
(containers use http://minio:9000). Credentials in `.env.example` are for
local development only. Keep AWS credentials aligned with MinIO credentials.

The one-shot `minio-init` creates five private buckets and exits with code 0.
Wait for it to finish before running the Python upload command below.
Dataset preparation runs in the local `.venv`; there is no seed container or
custom Docker build in the shared Compose file. The first preparation needs
internet for River's Elec2. The `miniodata` volume persists objects;
`docker compose down -v` deletes it.

Local Python workflow uses Python 3.11, pip and `.venv`, without uv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
set -a
. ./.env
set +a
python -m ingest.historical_datasets --all
```

`--to-parquet` prepares local data; `--download` also prepares data and lets
River fetch Elec2. `--upload-minio` uploads prepared files. `--force` replaces
existing objects. Override paths with `--beijing-csv` and `--output-dir`.

## Layout and naming

| Bucket | Producer → consumer | Contents |
|---|---|---|
| raw | ingest → dbt, AI | Historical sources and future API observations |
| staging | dbt → dbt | Cleaned intermediate observations |
| features | dbt → models, drift, serving | Feature tables and actual labels |
| models | models → serving | Model packages and training metadata |
| mlflow | MLflow → serving | MLOps artifacts |

```text
raw/
  beijing_pm25/
    station_id=beijing_Aotizhongxin/data.parquet
    station_id=beijing_Changping/data.parquet
    ... (12 stations)
    manifest.json
  elec2/
    elec2.parquet
    manifest.json
```

File names use lowercase snake_case; station identifiers retain source case.
Files use Snappy compression, one file per station and one file for Elec2.
The station_id column is retained within each file. Read individual files with
`pandas.read_parquet`; disable Hive partition inference if a dataset reader
reports conflicting types for the stored station column.

Reserved prefixes: `raw/openaq/`, `raw/openmeteo/`. Proposed downstream paths:
`features/pm25_hourly_features/station_id=<id>/data.parquet` and
`features/pm25_actual_labels/station_id=<id>/data.parquet`. These proposals
require agreement under CONTRACTS.md §3.6; this seed writes only raw data.

## Schema and provenance

Beijing is the exact PR #6 sample from `feat/verify-vn-air-quality-data`:
2,016 rows, 12 stations, 168 hours per station. It is a one-week sample, not
the full multi-year dataset. Original source:
https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.
The optional full UCI download is not implemented.

The seven columns follow CONTRACTS.md §3.2:

| Column | Stored type |
|---|---|
| station_id | string |
| event_time | timestamp, UTC |
| pm25_ug_m3 | float64 |
| temperature_c | float64 |
| relative_humidity_pct | float64 |
| wind_speed_m_s | float64 |
| wind_direction_deg | float64 |

Missing numeric observations are retained. Missing timestamps, invalid station
identifiers and duplicate station/time observations fail validation.

Elec2 uses `river.datasets.Elec2`, preserving the 45,312-observation order:

| Column | Stored type |
|---|---|
| sample_idx | int64, zero-based observation order |
| date | float64, original normalized value |
| day | int64 |
| period | float64, original normalized value |
| nswprice, nswdemand, vicprice, vicdemand, transfer | float64 |
| target | bool |

Elec2 is a separate classification benchmark, outside the PM2.5 contract.
Its fractional period values must be preserved for prequential evaluation.

Each manifest records `dataset_name`, `source_url`, `created_at_utc`,
`row_count`, `columns`, `schema_dtypes`, and `sha256_checksum` (a mapping from
dataset-relative Parquet paths to SHA-256). The Beijing manifest describes the
normalized frame before partitioning.

## Repeat runs and checks

Uploads check `head_object` and skip existing keys unless `--force` is set.
Permission and service errors propagate. Manifests upload after data files.
Uploads are not transactional. For changed input, use a clean local output
directory and deliberate remote versioning or replacement; skip mode does not
reconcile changed local data with existing remote objects.

```bash
ruff check .
ruff format --check .
pytest -v
python -m ingest.historical_datasets --all
```

A repeat seed reports zero uploaded objects. Inspect raw/beijing_pm25 and
raw/elec2 in the Console. Generated data/raw files and Parquet are Git-ignored.

## Shared Compose and local overrides

The shared MinIO service matches `feat/Nguyet-mlflow-serving`: image
`minio/minio`, ports 9000/9001, volume `miniodata`, credentials and `mc ready`
healthcheck. The initializer retains `mlflow` and additionally creates `raw`,
`staging`, `features` and `models`. This branch does not add the MLOps Postgres,
MLflow or serving services. Grafana retains the team's host port 3000.

On the implementation workstation, pulling `minio/minio` failed. An ignored
`docker-compose.override.yml` selects the previously built upstream-source
image `nhom4-minio:RELEASE.2025-10-15T17-29-55Z` and its HTTP readiness check.
This is a machine-specific fallback, not an image published for teammates.
The shared image's availability on a fresh machine remains a team infrastructure
issue to resolve with MLOps. Do not retag the local image as the official image.

Docker Compose loads the local override automatically. Inspect the shared
configuration alone using `docker compose -f docker-compose.yml config`.
Keep machine-specific settings out of the shared file. No Dockerfile is needed
for the dataset conversion and upload commands in this branch.

The ingestion code stays flat: `historical_datasets.py` owns conversion,
manifests and its CLI; `minio_storage.py` owns reusable S3 operations. Introduce
source-specific subpackages only when additional ingestion pipelines need them.
