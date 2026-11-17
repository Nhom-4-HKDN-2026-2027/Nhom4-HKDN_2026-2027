# Running the Vietnam data pipeline on a new machine

Run commands from the repository root. Requirements: **Python 3.11**, Internet access and your own OpenAQ API key. Fetch the two-year raw dataset before running the notebook; cloning the repository only provides one-week CSV samples. Do not copy virtual environments between machines or commit full raw data.

Commands use Linux/macOS paths. Python/uv commands work in Bash, Zsh and fish without environment activation; choose the matching shell block when entering the key. Clean installation was verified on Linux x86_64/Python 3.11.18, not Windows/PowerShell or macOS.

## 1. Create the environment

Use a new environment path; do not overwrite an existing environment:

```sh
python3.11 -m venv .venv/py311
.venv/py311/bin/python -m pip install -r requirements-py311.lock
.venv/py311/bin/python --version
.venv/py311/bin/ruff check .
.venv/py311/bin/ruff format --check .
.venv/py311/bin/pytest -q
```

## 2. Configure the pipeline and API key

`ingest/vietnam_pipeline.toml` is the shared, non-secret CLI/notebook configuration: UTC window, activity cutoff, sample dates, quality thresholds, boundary-gap policy, weather model and output paths. Relative TOML paths resolve against the repository root. Use new raw/sample output paths for each backfill; existing files are never intentionally overwritten.

The key is read from `OPENAQ_API_KEY`; **`.env` is not loaded automatically**. Enter the key without echoing it, using **one** block in the terminal that will run the backfill.

**Bash:**

```bash
read -r -s -p "OpenAQ API key: " OPENAQ_API_KEY
export OPENAQ_API_KEY
```

**Zsh:**

```zsh
read -r -s "OPENAQ_API_KEY?OpenAQ API key: "
export OPENAQ_API_KEY
```

**fish:**

```fish
read --silent --prompt-str 'OpenAQ API key: ' --local openaq_key
set -gx OPENAQ_API_KEY $openaq_key
set -e openaq_key
```

Keep the key out of TOML, notebooks, reports and command history. Run the remaining steps in the same terminal.

## 3. Fetch the two-year dataset

The handoff samples are `data/sample/openaq_pm25_hanoi_hcmc.csv` (815 rows) and `data/sample/openmeteo_weather_hanoi_hcmc.csv` (840 rows), from the 2026-11-16 backfill. Preserve them when testing a new fetch. Use **new raw and sample output paths**, such as the following; choose different names if these already exist:

```sh
.venv/py311/bin/python -m ingest.verify_vietnam_data --config ingest/vietnam_pipeline.toml --fetch --raw-dir data/raw/new_run --sample-dir data/raw/new_run_samples
.venv/py311/bin/python -m ingest.verify_vietnam_data --config ingest/vietnam_pipeline.toml --raw-dir data/raw/new_run --no-write
```

The CLI checks all output paths before calling APIs. The second command validates the cache and recomputes results in memory without writing files. New samples are placed under `data/raw/` in this example to keep exploratory outputs separate from the handoff samples in Git.

Raw output includes three normalized CSVs, `fetch_manifest.json`, the completeness summary and gaps. PM2.5 raw output retains one selected sensor per location, not every candidate or the original API JSON. Weather is fetched for the quality-selected station set. One-week samples are extracted from these datasets without additional API calls.

Manifest v2 records configuration, fetch time/status per location, sensor candidates/scores, Python version and CSV checksums. A v1 cache requires a new backfill directory. If any `fetch_failed` or `not_fetched` statuses remain, do not claim complete collection even when the command exits successfully. Fetching again may change station selection or values as sources update or archive delays expire; it is not guaranteed to reproduce the report exactly. Recomputing from the same raw snapshot is reproducible.

## 4. Run the notebook

Open Jupyter with the raw directory just fetched:

```sh
.venv/py311/bin/python -m ipykernel install --prefix .venv/py311 --name vn-py311 --display-name 'VN pipeline Python 3.11'
env -u VN_PIPELINE_ALLOW_LEGACY_CACHE VN_PIPELINE_RAW_DIR="$PWD/data/raw/new_run" .venv/py311/bin/jupyter lab
```

Before running all notebook sections, download the Beijing source and reproduce its sample using the command in [the Beijing setup section below](#6-beijing-multi-site-source-and-reproduction). This requires no API key and uses new output paths to preserve the handoff CSV.

Select `VN pipeline Python 3.11`, open `notebooks/01_verify_vn_data.ipynb` and run all cells from the start. Check the raw path printed in the first cell. The notebook validates the manifest and recomputes tables, charts and samples in memory; it neither writes CSVs nor fetches missing raw data automatically.

To use a different TOML file, provide it when launching Jupyter (replace the example path):

```sh
env -u VN_PIPELINE_ALLOW_LEGACY_CACHE VN_PIPELINE_CONFIG=/path/to/config.toml VN_PIPELINE_RAW_DIR="$PWD/data/raw/new_run" .venv/py311/bin/jupyter lab
```

## 5. Optional setup and legacy-cache audit

### Install Python with uv

If Python 3.11 is unavailable and uv is already installed:

```sh
uv python install 3.11 --install-dir .venv/interpreters --no-bin
uv venv .venv/py311 --python .venv/interpreters/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11
uv pip install --python .venv/py311/bin/python -r requirements-py311.lock
```

The interpreter directory depends on the downloaded Python version and OS. Check the actual path before running `uv venv`; the path above is the Linux interpreter used for local validation.

### Audit old CSVs without a manifest

Use this only for an explicit local legacy audit; source/model provenance remains unverified:

```sh
env VN_PIPELINE_RAW_DIR="$PWD/data/raw" VN_PIPELINE_ALLOW_LEGACY_CACHE=1 .venv/py311/bin/jupyter lab
```

To return to `new_run`, stop the existing Jupyter server and relaunch with the **correct raw path**:

```sh
env -u VN_PIPELINE_ALLOW_LEGACY_CACHE VN_PIPELINE_RAW_DIR="$PWD/data/raw/new_run" .venv/py311/bin/jupyter lab
```

These `env` settings apply only to the launched process. Restart the server when switching data sources, select the Python 3.11 kernel and run all cells again. To use the TOML default (`data/raw/verified_2026-11-16`) instead, launch with `env -u VN_PIPELINE_RAW_DIR -u VN_PIPELINE_ALLOW_LEGACY_CACHE .venv/py311/bin/jupyter lab`; clearing the override does not select `new_run` automatically.


## 6. Beijing Multi-Site source and reproduction

### Dataset identity and attribution

Use **UCI dataset 501, Beijing Multi-Site Air Quality**, DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), licensed under **CC BY 4.0**. Attribution: Chen, S. (2017), Beijing Multi-Site Air Quality, UCI Machine Learning Repository. The [official dataset page](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) describes 12 sites and hourly observations from March 2013 through February 2017, including missing values. Air-quality and weather observations come from different monitoring networks; weather is matched to nearby stations.

The official download contains `PRSA2017_Data_20130301-20170228.zip` plus unrelated files. Ingestion reads only the 12 `PRSA_Data_<station>_20130301-20170228.csv` members of that nested archive. It does **not** use the older single-site Beijing PM2.5 dataset (UCI 381, 2010–2014). The name `PRSA` in a filename alone does not distinguish these datasets.

### Reproduce the sample

After completing the environment setup above, run from the repository root:

```sh
.venv/py311/bin/python -m ingest.fetch_beijing --output data/raw/beijing_multisite/reproduced_features.csv --metadata data/raw/beijing_multisite/reproduced_manifest.json
```

No API key is required. The CLI downloads the public archive to `data/raw/beijing_multisite/uci_501.zip` if absent, otherwise reuses it. Outputs must not already exist. Defaults select `[2015-11-15T00:00:00Z, 2015-11-22T00:00:00Z)` and all 12 sites. Pass `--archive`, `--start`, `--end`, `--source-timezone`, `--output` and `--metadata` to select another source file or seven-day window. Calendar bounds must be timezone-aware whole hours; exactly seven days are required for this sample workflow.

The handoff file is [data/sample/beijing_multisite_features.csv](../data/sample/beijing_multisite_features.csv). Only this small derived CSV and ingestion code/docs/tests belong in the PR; do not commit the complete archive. Local metadata records source/member hashes, row counts, conversions, assumptions, missing-value counts and sample checksum. Run the CLI against the same source to regenerate it; no shared contract file was changed.

### Contract mapping

The CSV uses exactly the seven feature columns in `docs/CONTRACTS.md` §3.2:

| Contract column | Source / transformation | Unit |
|---|---|---|
| `station_id` | `beijing_` + original station name | String |
| `event_time` | `year/month/day/hour`, interpreted in source timezone, converted to UTC | ISO 8601 with `Z` |
| `pm25_ug_m3` | `PM2.5` | µg/m³ |
| `temperature_c` | `TEMP` | °C |
| `relative_humidity_pct` | Derived from `TEMP` and `DEWP` | % |
| `wind_speed_m_s` | `WSPM`, unchanged | m/s |
| `wind_direction_deg` | `wd`, 16-point compass mapping | Degrees clockwise from north |

**Timezone assumption:** the UCI page does not explicitly state a timezone. The default interprets calendar fields as local Beijing time, `Asia/Shanghai` (UTC+8 for the dataset period), then converts to UTC. Thus source `2015-11-15 08:00` becomes `2015-11-15T00:00:00Z`. The default UTC sample corresponds to local `[2015-11-15 08:00, 2015-11-22 08:00)`. This assumption is recorded in metadata and must be confirmed before cross-source temporal comparisons; use `--source-timezone` if authoritative clarification differs.

Humidity uses the liquid-water Magnus approximation, as used in [this AMS research article, Eq. 5](https://journals.ametsoc.org/view/journals/apme/61/8/JAMC-D-21-0091.1.xml):

```text
RH = 100 × exp(17.625 × Td / (243.04 + Td) − 17.625 × T / (243.04 + T))
```

`T=TEMP` and `Td=DEWP` are in Celsius. This is a derived approximation, not directly observed RH or an ice-relative calculation below freezing. Missing/nonfinite inputs, `Td > T`, singular/nonpositive denominators or invalid results produce NaN; humidity is not clipped or imputed. The raw inputs remain in the source archive. The formula is intended for meteorological temperatures, not arbitrary extreme values.

Wind uses the standard [16-point compass bearings](https://www.weather.gov/media/mqt/FireWx/MQT%20Fire%20Wx%20AOP%202020%20-%20Final%20Version.pdf): N=0, NNE=22.5, NE=45, …, NNW=337.5. These represent the direction **from which** wind blows. Missing directions stay NaN; unknown nonmissing tokens raise an error. `RAIN` is not added because it is not in the seven-column feature contract; raw precipitation remains available in the archive.

### Shared downstream adapter

CSV timestamp values are serialized as ISO 8601 strings; they become UTC timestamps when loaded for downstream processing. Vietnam's existing split CSVs retain numeric OpenAQ IDs for compatibility. `ingest/feature_contract.py` provides the common in-memory interface:

```python
import pandas as pd
from ingest.feature_contract import build_vietnam_features, normalize_feature_table

vn_features = build_vietnam_features(
    pd.read_csv("data/sample/openaq_pm25_hanoi_hcmc.csv"),
    pd.read_csv("data/sample/openmeteo_weather_hanoi_hcmc.csv"),
)
beijing_features = normalize_feature_table(
    pd.read_csv("data/sample/beijing_multisite_features.csv")
)
```

Both return exactly seven columns: `station_id` as pandas string, `event_time` as timezone-aware UTC datetime, and five float columns. The Vietnam adapter performs a one-to-one left join on normalized station/time keys. Missing weather remains NaN with a warning; PM2.5 rows are preserved. Naive/invalid/non-hourly timestamps, missing IDs, duplicate keys, nonnumeric values and infinities are rejected. Numeric NaN and finite source values, including negatives, remain available for quality assessment; this adapter does not define experiment acceptance or imputation policies. Contract names and units are unchanged.

Verified source archive SHA-256:
`b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`

Handoff CSV SHA-256:
`525ba52fbde3001bd2f80f901911f77dd3d003f32a8d8a0d40698301545ed162`

## 7. File structure and responsibilities

### Ingestion and configuration

| File | Responsibility |
|---|---|
| `ingest/fetch_openaq.py` | Discover Vietnam PM2.5 locations, fetch hourly observations, select one sensor per location and calculate completeness, gaps and lifecycle counts. |
| `ingest/fetch_openmeteo.py` | Fetch ERA5 weather at selected station coordinates and validate hourly timestamps and units while retaining permitted missing values. |
| `ingest/fetch_beijing.py` | Download/read the UCI Multi-Site archive and extract a seven-day sample with derived humidity, wind bearings and UTC timestamps. |
| `ingest/verify_vietnam_data.py` | Run the Vietnam CLI: coordinate fetching, verify cached data and provenance, and produce quality summaries, gaps and samples. |
| `ingest/feature_contract.py` | Join Vietnam PM2.5/weather and normalize Vietnam and Beijing tables to the same seven-column downstream contract. |
| `ingest/pipeline_config.py` | Load and validate pipeline configuration and resolve repository-relative paths. |
| `ingest/vietnam_pipeline.toml` | Define Vietnam evaluation/sample windows, screening parameters, weather model and output paths; contains no API key. |
| `ingest/__init__.py` | Expose selected ingestion and quality functions as package imports. |

### Analysis, data and environment

| Path | Responsibility |
|---|---|
| `notebooks/01_verify_vn_data.ipynb` | Read local source data, recompute quality tables/charts and samples in memory, compare the Beijing sample with its source and verify the shared schema. Run ingestion first. |
| `data/raw/` | Local normalized Vietnam CSVs, manifests, quality outputs and the Beijing source archive. Excluded from Git; a clone alone cannot reproduce the full notebook analysis. |
| `data/sample/openaq_pm25_hanoi_hcmc.csv` | One-week Vietnam PM2.5 handoff sample extracted from the backfill. |
| `data/sample/openmeteo_weather_hanoi_hcmc.csv` | Weather for the same Vietnam sample window and selected stations. |
| `data/sample/beijing_multisite_features.csv` | One-week, twelve-site Beijing handoff sample in the shared feature schema. |
| `requirements-py311.lock` | Pinned Python dependencies for reproducing the tested Python 3.11 environment. |
| `docs/data_quality_report.md` | Findings, limitations and the demo/experiment decision. This setup guide documents how to reproduce them. |

### Tests

| File | Responsibility |
|---|---|
| `tests/test_vietnam_data.py` | Offline OpenAQ/Open-Meteo fetching, retries, sensor selection, completeness, gaps, lifecycle and weather validation. |
| `tests/test_vietnam_pipeline.py` | Cache validation, manifests, sample handoff, custom configuration, reproducible outputs and CLI protection against overwrites. |
| `tests/test_pipeline_config.py` | Reject invalid configuration values and unknown TOML keys. |
| `tests/test_fetch_beijing.py` | Beijing source/archive handling, conversions, missing values and sample checks. |
| `tests/test_feature_contract.py` | Shared schema/types, Vietnam joins, missing-value preservation and malformed-input rejection. |
| `tests/test_project_structure.py` | General repository structure checks, separate from data-quality calculations. |

### Execution flow

Vietnam: `vietnam_pipeline.toml` → `verify_vietnam_data` → OpenAQ observations and selected-station Open-Meteo weather → local raw CSVs and manifest → completeness/gap outputs and one-week samples.

Beijing: UCI Multi-Site archive → `fetch_beijing` → conversions and seven-day extraction → feature sample and local provenance metadata.

The notebook reads the local sources and recomputes the analysis; the report summarizes the verified findings. `feature_contract` provides the common downstream interface for both datasets. Missing intervals remain visible throughout these flows; they are not filled from other stations.
