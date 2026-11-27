# Historical datasets and MinIO storage

## Start MinIO and populate `raw` (step-by-step order)

Compose starts MinIO and initializes its five buckets (`raw`, `staging`,
`features`, `models`, `mlflow`), but it **does not** download, convert or
upload datasets automatically. A new MinIO instance has an empty `raw` bucket
until you run the manual ingestion command in **Step 3** below. Existing objects
persist in the `miniodata` Docker volume.

Run the following steps in order from the repository root:

### Step 1: Start MinIO and verify bucket initialization

```bash
docker compose up -d
docker compose logs minio-init
docker compose ps -a
```

The one-shot `minio-init` container exits with code `0` after creating any
missing buckets.

### Step 2: Create the Python 3.11 virtual environment and install dependencies

**Linux / macOS (Bash/Zsh):**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### Step 3: Load environment variables and run manual data ingestion into `raw`

`ingest/minio_storage.py` reads MinIO credentials and `MINIO_ENDPOINT` from the
process environment (`.env` is not loaded automatically). Copy `.env.example`
to `.env`, load it into your shell session, and run `ingest.historical_datasets --all`:

**Linux / macOS (Bash/Zsh):**

```bash
cp -n .env.example .env
set -a
. ./.env
set +a
python -m ingest.historical_datasets --all
```

**Windows (PowerShell):**

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
}
.\.venv\Scripts\python.exe -m ingest.historical_datasets --all
```

When complete, the CLI prints `Uploaded 63 objects to raw; existing objects skipped unless --force`
(on a fresh bucket) and populates both `raw/beijing_pm25/` and `raw/elec2/`.

MinIO Console: http://localhost:9001 (`127.0.0.1:9001`). S3 API:
http://localhost:9000 (`127.0.0.1:9000`; containers use `http://minio:9000`).
Local development credentials are documented in `.env.example`; use `.env` to
override them. Keep AWS credentials aligned with MinIO credentials. Host ports
for MinIO (`127.0.0.1:9000`, `127.0.0.1:9001`), Prometheus (`127.0.0.1:9090`)
and Grafana (`127.0.0.1:3000`) are bound to localhost.

Compose pins the publicly pullable `chainguard/minio` image by SHA-256 digest
(`chainguard/minio@sha256:e9cf6b99c226e9539ed0f14b6f00e37b07da153c7ba1f4f8709c807574b471ec`,
packaging MinIO `RELEASE.2026-11-16T00-11-40Z`) and pins `minio-init` to
`public.ecr.aws/aws-cli/aws-cli:2.41.2`. The original `minio/minio` Docker Hub
image could not be pulled on a clean setup. MinIO runs as UID:GID `100:101` to
retain access to data created by the previous image. Ports 9000/9001, the
`miniodata` volume, and the `mc ready local` healthcheck remain unchanged.
Bucket `mlflow` is preserved; the initializer adds `raw`, `staging`, `features`
and `models`. There is no custom image build or dataset container.

## Buckets and partitions

| Bucket | Producer → consumer | Contents |
|---|---|---|
| raw | ingest → dbt, AI | Historical source observations |
| staging | dbt → dbt | Cleaned intermediate observations |
| features | dbt → models, drift, serving | Feature tables and actual labels |
| models | models → serving | Model packages and metadata |
| mlflow | MLflow → serving | Existing MLOps artifacts |

```text
raw/
  beijing_pm25/
    year=2013/
      station_id=beijing_Aotizhongxin/data.parquet
      station_id=beijing_Changping/data.parquet
      ... (12 stations)
    year=2014/...
    year=2015/...
    year=2016/...
    year=2017/...
    manifest.json
  elec2/
    elec2.parquet
    manifest.json
```

Beijing has 60 Snappy-compressed files: one per UTC year/station pair. Elec2 has
one file. There are 63 active objects including the two manifests. Paths are
stable, so repeat runs do not generate new keys. File names use lowercase
snake_case; station identifiers retain their source case.

Reserved prefixes `raw/openaq/` and `raw/openmeteo/` are not populated. Proposed
`features/pm25_hourly_features/` and `features/pm25_actual_labels/` locations still
require team agreement; the uploader writes only the two historical raw datasets.

## Source and schemas

Beijing uses version 1 of
[Kaggle Beijing Multi-Site Air-Quality](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set):
420,768 observations, 12 stations, March 1, 2013 through February 28, 2017 in
source calendar time. The 12 CSV files were verified byte-for-byte against the
nested `PRSA2017_Data_20130301-20170228.zip` in
[UCI dataset 501](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data),
DOI `10.24432/C5RK5G`. The distribution archive checksums differ because the
packaging differs; the CSV checksums match.

Source calendar fields are assumed to be Beijing local time (`Asia/Shanghai`),
consistent with the PR #6 sample; the source description does not specify a
formal timezone. `--source-timezone` can override this assumption. UTC coverage
is February 28, 2013 16:00 through February 28, 2017 15:00. Partition years are
computed from UTC, so local January 1 early hours can belong to the preceding year.

Beijing's logical schema follows CONTRACTS.md §1.2 (base observation schema):

| Column | Type / derivation |
|---|---|
| station_id | string, `beijing_` + source station |
| event_time | timestamp, UTC |
| pm25_ug_m3 | float64, source PM2.5 |
| temperature_c | float64, source TEMP |
| relative_humidity_pct | float64, derived from TEMP and DEWP |
| wind_speed_m_s | float64, source WSPM |
| wind_direction_deg | float64, 16-point compass mapping; N=0, clockwise |
| precipitation_mm | float64, source RAIN |

Relative humidity uses the Magnus approximation over liquid water:
`100*exp(17.625*Td/(243.04+Td) - 17.625*T/(243.04+T))` (Celsius).
Missing/nonfinite inputs, invalid denominators and dewpoint above temperature
remain missing. Missing numeric observations and negative source measurements
are retained; rows are not silently dropped, clipped or imputed.
Duplicate station/time observations and invalid station IDs fail validation.
Normalization is independent of PR #6; it does not import or modify that branch.
The existing one-week sample remains only for unit tests and explicit previews.

`station_id` is retained in the Parquet files with int32 dictionary encoding
compatible with Hive partition discovery. To load the **entire Beijing
Multi-Site dataset** (all 60 partitions, 420,768 rows across 2013–2017) for
multi-year training and a full 12-month baseline evaluation:

```python
from pathlib import Path
import pandas as pd
from ingest.historical_datasets import BEIJING_COLUMNS

paths = sorted(Path("data/raw/beijing_pm25").glob("year=*/station_id=*/data.parquet"))
beijing = pd.read_parquet(paths, columns=BEIJING_COLUMNS)
beijing["station_id"] = beijing["station_id"].astype("string")
beijing = beijing.sort_values(["station_id", "event_time"]).reset_index(drop=True)
```

Because `year=YYYY` directories are partitioned by **UTC** timestamps while
Beijing calendar months are evaluated in `Asia/Shanghai` (`UTC+8`) and use up to
24-hour lags, loading all `year=*` partitions preserves the contiguous 1-hour
grid per station across year boundaries (`2013-03-01 00:00` through
`2017-02-28 23:00` local time). You can also filter a single UTC year when
needed (`glob("year=2015/station_id=*/data.parquet")`).

Use a Parquet glob or manifest-listed paths rather than passing the dataset
root containing `manifest.json` to a Parquet-only reader. Hive readers may add
a virtual `year` column; passing `columns=BEIJING_COLUMNS` selects the exact
eight base contract columns when passing to AI.

Elec2 comes from `river.datasets.Elec2`, preserving 45,312 observations:
`sample_idx` and `day` are int64; `target` is bool; `date`, `period`, `nswprice`,
`nswdemand`, `vicprice`, `vicdemand`, and `transfer` are float64. Its normalized
`date`/`period` are not timestamps. Prequential sample order is retained.

Each manifest records dataset name, source URL, creation time, row count,
columns, normalized dtypes, and per-file SHA-256 checksums. Beijing additionally
records source version, UCI page/DOI, archive checksum, timezone assumption,
station count, UTC time range and partition years.

## Repeat runs and recovery

- The `miniodata` Docker volume persists objects across Compose restarts.
- Kaggle archives are cached under the local output directory's `downloads/`.
  River uses its normal local cache (or the `RIVER_DATA` environment variable).
- `--all` reuses source downloads, prepares Parquet and skips unchanged remote
  objects. It still performs conversion; it does not check MinIO before preparation.
- `--upload-minio` reuses already prepared local Parquet without downloading or
  converting sources again, and uploads only missing objects.
- Changed manifests require explicit `--force`; permission and service errors
  propagate rather than being treated as missing data.
- Files are uploaded before their manifest. Uploads are not transactional.

Do not run `docker compose down -v` unless you intend to delete MinIO data.
Local files under `data/raw/` are independent of Docker volumes and Git-ignored.

## Manual preparation and migration

For local Python use Python 3.11, pip and `.venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
set -a
. ./.env
set +a
python -m ingest.historical_datasets --all
```

`--to-parquet` (or `--download`) downloads/caches sources and prepares local
files without uploading. `--upload-minio` uploads existing local output.
`--beijing-archive` selects an existing Kaggle ZIP; `--output-dir` changes the
local output root. Generated `data/raw/` files are Git-ignored.

To migrate an existing sample dataset, first back up its manifest-listed files,
then run:

```bash
python -m ingest.historical_datasets --all --force
```

`--force` replaces current files and removes only superseded Parquet keys named
in the prior dataset manifest. Other datasets and buckets are not deleted.
A repeat normal upload skips unchanged data. A changed dataset requires force;
local checksum validation happens before upload. Local Beijing output is staged
before replacement to prevent stale station partitions from surviving reruns.

For a sample-only preview use a separate output directory:

```bash
python -m ingest.historical_datasets --to-parquet \
  --beijing-csv data/sample/beijing_multisite_features.csv \
  --output-dir data/raw/sample_preview
```

Do not force-upload a preview over the full dataset.

## Checks

```bash
ruff check .
ruff format --check .
pytest -v
docker compose -f docker-compose.yml config -q
```
