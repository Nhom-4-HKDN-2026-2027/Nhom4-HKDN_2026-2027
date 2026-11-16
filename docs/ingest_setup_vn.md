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

`ingest/vn_pipeline.toml` is the shared, non-secret CLI/notebook configuration: UTC window, activity cutoff, sample dates, quality thresholds, boundary-gap policy, weather model and output paths. Relative TOML paths resolve against the repository root. Use new raw/sample output paths for each backfill; existing files are never intentionally overwritten.

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
.venv/py311/bin/python -m ingest.verify_vn_data --config ingest/vn_pipeline.toml --fetch --raw-dir data/raw/new_run --sample-dir data/raw/new_run_samples
.venv/py311/bin/python -m ingest.verify_vn_data --config ingest/vn_pipeline.toml --raw-dir data/raw/new_run --no-write
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

Before running all notebook sections, download the Beijing source and reproduce its sample using the command in [beijing_multisite.md](beijing_multisite.md). This requires no API key and uses new output paths to preserve the handoff CSV.

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
