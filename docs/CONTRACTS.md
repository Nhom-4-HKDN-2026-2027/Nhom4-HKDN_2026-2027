# INTER-TEAM TECHNICAL AGREEMENT

## PROJECT: Adaptive Machine Learning Model Updating

---

## General Information

**Objective:** Establish consistent data, model, deployment, and monitoring interfaces so that the three teams can develop in parallel without depending on one another's completion.

---

## PART 1. DATA PIPELINE → AI → MLOPS

### 1. Data Contract – Feature Table

*Responsible team: Data Pipeline | Reviewing teams: AI, MLOps*

#### 1.1. Data Source Information

| Item                       | Agreed Information                                                                                                                                                                                                                                                             |
| :------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Datasets**         | 1. **Historical Benchmark:** Beijing Multi-Site Air Quality Dataset and Elec2 Dataset<br>2. **Live / Vietnam Air Quality:** OpenAQ (PM2.5) + Open-Meteo ERA5 (Weather)                                                                                             |
| **Data sources**     | - Beijing Air Quality: Kaggle (`sid321axn/beijing-multisite-airquality-data-set`), UCI (`10.24432/C5RK5G`)<br>- Elec2: River ML datasets (`river.datasets.Elec2`)<br>- Vietnam PM2.5: OpenAQ API<br>- Vietnam Weather: Open-Meteo API (Historical / Forecast Weather model `era5`) |
| **Cities / regions** | - Beijing: 12 monitoring stations in Beijing<br>- Vietnam: Hanoi and Ho Chi Minh City (HCMC)<br>- Elec2: New South Wales, Australia                                                                                                                                        |
| **Data time range**  | - Beijing: 2013-03-01 – 2017-02-28 (hourly)<br>- Vietnam: 2024-11-14 – 2026-11-14 (2 years, according to `vietnam_pipeline.toml`); one-week sample: 2025-11-15 – 2025-11-22                                                                                               |
| **Data formats**     | Parquet (primary processing/storage format) / CSV (sample or exported data)                                                                                                                                                                                                    |
| **Storage location** | MinIO S3 object storage (buckets: `raw`, `staging`, `features`) at `http://minio:9000` (inside Docker) or `http://localhost:9000` (host); local storage at `data/raw/` and `data/sample/`                                                                        |
| **Update frequency** | Hourly                                                                                                                                                                                                                                                                         |
| **Owner**            | Data Pipeline Team (`@ngynqt`, `@HuynhThiCamGiang`, `@XuanHuong63`)                                                                                                                                                                                                            |

#### 1.2. Feature Table Schema

- **Table name:** `pm25_hourly_features`
- **Description:** An hourly feature table containing historical PM2.5 concentrations, lag features, rolling mean features, meteorological variables (including rainfall/precipitation), and time-based features for PM2.5 forecasting.
- **Base Observation Schema (8 columns produced by `ingest/feature_contract.py` & stored in `s3://raw/` / `data/sample/`):** `station_id`, `event_time`, `pm25_ug_m3`, `temperature_c`, `relative_humidity_pct`, `wind_speed_m_s`, `wind_direction_deg`, `precipitation_mm`.

| No. | Column name | Data type | Unit | Nullable? | Description |
|:---:|---|---|:---:|---|---|
| 1 | `station_id` | STRING | - | No | Monitoring station identifier |
| 2 | `event_time` | TIMESTAMP (ISO 8601 UTC) | UTC | No | Observation time, rounded to the hour |
| 3 | `pm25_ug_m3` | FLOAT | µg/m³ | Yes | Measured PM2.5 concentration at the current time |
| 4 | `pm25_lag_1h` | FLOAT | µg/m³ | Yes | PM2.5 concentration lagged by 1 hour |
| 5 | `pm25_lag_3h` | FLOAT | µg/m³ | Yes | PM2.5 concentration lagged by 3 hours |
| 6 | `pm25_lag_24h` | FLOAT | µg/m³ | Yes | PM2.5 concentration lagged by 24 hours |
| 7 | `pm25_rolling_mean_3h` | FLOAT | µg/m³ | Yes | Mean PM2.5 concentration over the previous 3 hours |
| 8 | `pm25_rolling_mean_24h` | FLOAT | µg/m³ | Yes | Mean PM2.5 concentration over the previous 24 hours |
| 9 | `temperature_c` | FLOAT | °C | Yes | Air temperature |
| 10 | `relative_humidity_pct` | FLOAT | % | Yes | Relative humidity |
| 11 | `wind_speed_m_s` | FLOAT | m/s | Yes | Wind speed |
| 12 | `wind_direction_deg` | FLOAT | Degree | Yes | Wind direction |
| 13 | `precipitation_mm` | FLOAT | mm | Yes | Hourly precipitation / rainfall |
| 14 | `hour` | INTEGER | 0–23 | No | Hour of day |
| 15 | `weekday` | INTEGER | 0–6 | No | Day of the week |

**Notes:**
- Lag and rolling mean features must be calculated separately for each monitoring station (`station_id`).
- Lag features represent historical observations and must not use future data.
- Rolling mean features must be calculated using historical data only to avoid data leakage.
- `hour` is extracted from `event_time`.
- `weekday` is extracted from `event_time`, where 0 represents Monday and 6 represents Sunday.
- Missing historical values may result in NULL values in lag and rolling mean features.
- For the Vietnam dbt feature mart (`dbt_pm25`), additional calendar/location columns (`city` STRING, `is_weekend` BOOLEAN, `is_holiday` BOOLEAN) may be included alongside the 15 core columns above.

#### 1.3. Data Rules

| Topic                                | Agreed Approach                                                                                                                                                                                                |
| :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp format**           | ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`)                                                                                                                                                                            |
| **Timezone**                   | UTC (timezone-aware timestamps are mandatory; local timestamps from source systems must be normalized to UTC)                                                                                                  |
| **Missing values**             | Preserve `NaN` for numeric fields. Do not impute or silently drop missing values in the schema adapter layer.                                                                                                |
| **Duplicate records**          | Identify records using the composite key `(station_id, event_time)`. Duplicate records are not allowed; raise an error if detected.                                                                          |
| **Outliers**                   | Reject non-finite values (`inf`, `-inf`) and physically invalid ranges (e.g., `pm25_ug_m3 < 0`, `relative_humidity_pct` outside `[0, 100]`, `wind_speed_m_s < 0`, `wind_direction_deg` outside `[0, 360]`, `precipitation_mm < 0`); preserve valid extreme pollution spikes for drift detection. |
| **Late-arriving data**         | Retain late-arriving records; do not silently discard them. Associate them with the corresponding station and `target_time`. The model should calculate errors only when the actual label becomes available. |
| **Data updates / corrections** | Idempotent ingestion keyed by `(station_id, event_time)`. MinIO Parquet uploads skip existing objects by default and overwrite only when explicitly run with `--force` (updating `manifest.json`).             |
| **Data validation**            | Validate schema, data types, primary keys, and hourly timestamp alignment through `ingest/feature_contract.py`; check data completeness and gaps using `verify_vietnam_data.py`.                           |

#### 1.4. Handoff Between Teams

- **Data Pipeline provides data at:** MinIO bucket `features` (`s3://features/`) or `data/sample/` (`openaq_pm25_hanoi_hcmc.csv`, `openmeteo_weather_hanoi_hcmc.csv`, `beijing_multisite_features.csv`); Beijing Parquet data at `s3://raw/beijing_pm25/`.
- **AI reads data using:** Pandas / PyArrow (`pd.read_parquet()`, `pd.read_csv()`).
- **MLOps accesses data using:** MinIO S3 API via Boto3 or a mounted shared volume.
- **Update frequency:** Hourly.
- **Confirmation:** Data Pipeline: Nguyễn Quốc Tường (`@ngynqt` – Confirmed) | AI: TO BE FILLED IN | MLOps: Nguyệt

---

## PART 2. AI → MLOPS

### 2. Model Interface Contract

*Responsible team: AI | Reviewing team: MLOps*

#### 2.1. Model Information

| Item                         | Agreed Approach                                                                                                                                                                                                                                                                                                                                                                     |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Problem type**       | PM2.5 Forecasting                                                                                                                                                                                                                                                                                                                                                                   |
| **Model algorithm**    | - Primary experiment: LightGBM Regressor (`LGBMRegressor`)<br>- Baseline for comparison: Persistence model (uses the current PM2.5 value as the forecast for the future)<br>- Current MLOps mock component: `RandomForestClassifier` trained on the Breast Cancer dataset                                                                                                       |
| **Framework**          | LightGBM / Scikit-Learn                                                                                                                                                                                                                                                                                                                                                             |
| **Language / version** | Python 3.11                                                                                                                                                                                                                                                                                                                                                                         |
| **Input features**     | Current baseline notebook experiments use: `pm25_ug_m3`, `hour`, `dayofweek`, PM2.5 lags `pm25_lag1`, `pm25_lag2`, `pm25_lag3`, `pm25_lag6`, `pm25_lag12`, `pm25_lag24`, and weather lags (`temperature_c_lag1`, `relative_humidity_pct_lag1`, `wind_speed_m_s_lag1`, `wind_direction_deg_lag1`). Official feature contract for handoff: TO BE FILLED IN. |
| **Prediction target**  | `predicted_pm25_ug_m3` (PM2.5 concentration at the forecast time)                                                                                                                                                                                                                                                                                                                 |
| **Forecast horizon**   | 1 hour (1-hour-ahead forecast: `target_time = event_time + 1h`)                                                                                                                                                                                                                                                                                                                   |
| **Output type**        | FLOAT (non-negative real number: `np.maximum(prediction, 0)`)                                                                                                                                                                                                                                                                                                                     |
| **Evaluation metrics** | MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)                                                                                                                                                                                                                                                                                                                        |
| **Training frequency** | TO BE FILLED IN                                                                                                                                                                                                                                                                                                                                                                     |

#### 2.2. Training Interface

- **Function name:** `train_model(train_data, config=None)` *(proposed in `docs/CONTRACTS.md`; not yet implemented as a separate code module)*

**Input:**

| Parameter      | Data type              | Description                                             |
| :------------- | :--------------------- | :------------------------------------------------------ |
| `train_data` | DataFrame              | Training dataset containing features and target         |
| `config`     | DICT / TO BE FILLED IN | Model training configuration (optional hyperparameters) |

**Output:**

| Field       | Data type              | Description                                          |
| :---------- | :--------------------- | :--------------------------------------------------- |
| `model`   | OBJECT                 | Trained model                                        |
| `metrics` | DICT                   | Evaluation results on the test set (MAE, RMSE, etc.) |
| `params`  | DICT / TO BE FILLED IN | Hyperparameters used                                 |

**Training conventions:**

- **Train/test split:** Time-based split (split chronologically; define a cutoff between training and test data; do not split randomly).
- **Shuffle:** Disabled (`shuffle=False`) to prevent time-series leakage.
- **Random seed:** `42`
- **Model storage:** MLflow Model format (`mlflow.sklearn.log_model` or the corresponding flavor).

#### 2.3. Prediction Interface

- **Function name:** `predict(model, input_data)` *(proposed in `docs/CONTRACTS.md`; not yet implemented as a separate module)*

| Item                               | Agreed Approach                                                                                                                                               |
| :--------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Input schema**             | TO BE FILLED IN                     |
| **Input format**             | JSON / DataFrame                                                                                                                                              |
| **Output schema**            | `station_id` (STRING), `target_time` (TIMESTAMP), `predicted_pm25_ug_m3` (FLOAT), `model_version` (STRING) *(as proposed in `docs/CONTRACTS.md`)* |
| **Output format**            | JSON                                                                                                                                                          |
| **Batch prediction**         | Supported                                                                                                                                                     |
| **Missing feature handling** | TO BE FILLED IN                                                                                                                                               |

**Example input:**

```json
{"pm25_ug_m3": 35.2, "pm25_lag_1h": 32.1}
```

**Example output:**

```json
{"prediction": 36.5}
```

#### 2.4. Model Artifact

| Item                          | Agreed Approach                                                                                                                                           |
| :---------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model format**        | MLflow Model format                                                                                                                                       |
| **Artifact storage**    | MinIO S3 (`s3://mlflow/` or `s3://models/`)                                                                                                           |
| **Model serialization** | MLflow flavors (cloudpickle / joblib through `mlflow.sklearn` or `mlflow.lightgbm`)                                                                   |
| **Model signature**     | Mandatory (inferred using `mlflow.models.infer_signature`)                                                                                              |
| **Dependencies**        | `requirements.txt` / `conda.yaml` generated automatically by MLflow, and the project lockfile `requirements-py311.lock`                             |
| **Required metadata**   | `model_name`, `model_version`, `training_timestamp`, `dataset_version`, `model_strategy`, `feature_schema_version`, `mae`, `rmse`, `r2` |

#### 2.5. Model Evaluation & Acceptance

| Criterion | Requirement |
| :--- | :--- |
| **Evaluation Metrics** | MAE (µg/m³), RMSE (µg/m³) |
| **Baseline Model** | Persistence Model |
| **Acceptance Condition** | The candidate model must outperform the Persistence baseline by achieving lower MAE and RMSE. |
| **Comparison with Champion** | 1. **Initial Deployment:** The candidate model becomes the `@champion` if it outperforms the Persistence baseline.<br>2. **Subsequent Updates:** The candidate must achieve at least 1–2% lower MAE than the current `@champion`, or demonstrate statistically significant improvement in prediction accuracy during concept drift events. |
| **Model Approver** | Thiện, Nguyệt |
**AI handoff to MLOps:**

- [X] Model artifact
- [X] Model signature
- [X] Training parameters
- [X] Evaluation metrics
- [X] Dataset version / reference
- [X] Source code / Git commit
- [X] Instructions for loading and predicting (`mlflow.pyfunc.load_model(...)` and FastAPI `POST /predict`)
- [X] Test input/output examples (13 PM2.5 features array -> list of float predictions)

*Confirmation: AI: TO BE FILLED IN | MLOps: Minh Nguyệt*

---

## PART 3. MLOPS

### 3. Model Registry Contract

*Responsible team: MLOps | Reviewing team: AI*

#### 3.1. MLflow Configuration

| Item                            | Agreed Approach                                                                                                   |
| :------------------------------ | :---------------------------------------------------------------------------------------------------------------- |
| **MLflow Tracking URI**   | `http://mlflow:5000` (inside Docker) or `http://localhost:5000` (host)                                        |
| **Backend Store**         | PostgreSQL (`postgresql+psycopg2://mlflow:mlflow123@postgres:5432/mlflow`)                                      |
| **Artifact Store**        | MinIO (`s3://mlflow/` with endpoint `http://minio:9000`)                                                      |
| **Experiment name**       | `pm25_forecasting` *(proposed in `docs/CONTRACTS.md`; current mock code uses `adaptive-model-updating`)*  |
| **Registered model name** | `pm25-forecasting` *(proposed in `docs/CONTRACTS.md`; current mock code uses `BreastCancerRandomForest`)* |
| **Model versioning**      | MLflow version number (auto-incremented: 1, 2, 3, ...)                                                            |
| **Model artifact format** | MLflow Model                                                                                                      |
| **Model signature**       | Mandatory                                                                                                         |

#### 3.2. Naming Conventions

| Object                     | Naming convention          | Example                                           |
| :------------------------- | :------------------------- | :------------------------------------------------ |
| **Experiment**       | snake_case                 | `pm25_forecasting`                              |
| **Registered model** | kebab-case                 | `pm25-forecasting`                              |
| **Model version**    | Auto-incrementing integer  | `1`, `2`                                      |
| **Run name**         | `<model_type>-<purpose>` | `lightgbm-baseline`, `random-forest-baseline` |
| **Artifact**         | MLflow Model standard      | `model/model.pkl`, `MLmodel`                  |

#### 3.3. Model Lifecycle

| State / Alias                                | Meaning                                                 | Transition condition                                                                                                                               |
| :------------------------------------------- | :------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Candidate**                          | A newly logged and registered model awaiting evaluation | Successfully registered in MLflow Registry                                                                                                         |
| **Champion** *(alias `@champion`)* | The model currently serving production traffic          | Meets evaluation criteria and outperforms the current model                                                                                        |
| **Archived**                           | A previous model that is no longer serving              | When a Candidate is assigned the `champion` alias, the previous model loses the `champion` alias and is retained in MLflow history as archived |

**Approval workflow for a new model:**

1. AI completes model training.
2. The model is logged to MLflow with parameters, metrics, signature, and artifact.
3. MLOps validates the artifact and signature.
4. AI and MLOps evaluate the results against the baseline and champion.
5. An accepted model is assigned the `champion` alias.
6. Serving loads and uses the new model.

**Authorized approvers:** Nguyet and Khiem, Nhi

---

## PART 4. MLOPS → AI / DATA PIPELINE

### 4. Model Serving Contract

*Responsible team: MLOps | Consumer teams: AI, Data Pipeline*

#### 4.1. API Configuration

| Item                     | Agreed Information                                                            |
| :----------------------- | :---------------------------------------------------------------------------- |
| **Service name**   | Prediction API *(Swagger title in `serving/main.py`: "Prediction API")*   |
| **API framework**  | FastAPI                                                                       |
| **Base URL**       | `http://localhost:8000` (host) or `http://api:8000` (inside Docker)       |
| **Port**           | 8000                                                                          |
| **API endpoints**  | `/predict` (prediction), `/health` (health check), `/docs` (Swagger UI) |
| **HTTP methods**   | POST (`/predict`), GET (`/health`, `/`)                                 |
| **Content-Type**   | `application/json`                                                          |
| **Authentication** | None (internal Docker network phase)                                          |
| **Timeout**        | 30 seconds                                                                    |

#### 4.2. API Request

| Field | Data type | Required | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | STRING | No | Name of the model (defaults to `MODEL_NAME` from environment: `pm25-forecasting`) |
| `features` | OBJECT / ARRAY | Yes | Array of observation feature samples (each sample contains 13 PM2.5 & weather features) |
| `timestamp` | STRING | No | Request submission timestamp (optional) |

#### 4.3. API Response

| Field | Data type | Description |
| :--- | :--- | :--- |
| `request_id` | STRING | Unique request identifier (UUID) generated for distributed tracing and log auditing |
| `status` | STRING | Processing status of the prediction request (`"success"`) |
| `model_name` | STRING | Name of the registered model serving predictions (`"pm25-forecasting"`) |
| `model_alias` | STRING | Active model registry alias deployed in serving (`"champion"`) |
| `predictions` | LIST[FLOAT] | List of forecasted PM2.5 concentrations in µg/m³ (non-negative floats) |

**Example response:**
```json
{
  "request_id": "c1f8d4e2-8b09-4df8-7e41-28e441b209a1",
  "status": "success",
  "model_name": "pm25-forecasting",
  "model_alias": "champion",
  "predictions": [38.5]
}
```

#### 4.4. Error Handling

| HTTP status   | Scenario                              | Handling                                                                                                                               |
| :------------ | :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------- |
| **400** | Invalid input | Return `{"detail": "Features array is empty or invalid format"}` |
| **404** | Model not found | Return `{"detail": "Model pm25-forecasting@champion not found in registry"}` |
| **422** | Invalid schema / Unprocessable Entity | Return detailed schema-validation errors from FastAPI/Pydantic (or when the feature dimensionality does not match the expected input). |
| **500** | Internal error                        | Catch the exception, log the system error, and return `{"detail": str(exc)}`.                                                        |
| **503** | Model unavailable                     | Return `{"detail": "Model is not loaded"}` when the model has not been initialized successfully.                                     |

#### 4.5. Model Loading

| Item                                              | Approach                                                                                                          |
| :------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------- |
| **Load model from**                         | MLflow Model Registry using URI `models:/{MODEL_NAME}@{MODEL_ALIAS}` or `MODEL_URI`                           |
| **Model version selection**                 | Through the model alias (default: `champion`) or the `MODEL_ALIAS` environment variable                       |
| **Model alias**                             | `champion`                                                                                                      |
| **Reload after a new version is available** | Call `POST /reload` on the service or restart the API container.                                                |
| **Model rollback**                          | MLOps changes the `champion` alias in the MLflow UI to the desired previous version and calls `POST /reload`. |

*Confirmation: AI: Trọng Thiện (AI Lead) | MLOps: Minh Nguyệt (MLOps Lead)*

---

## PART 5. ALL TEAMS

### 5. Monitoring & Observability Contract

*Infrastructure owner: MLOps | Signal providers: AI, Data Pipeline*

#### 5.1. Monitoring Metrics

| Metric name                                                      | Description                                                                          | Unit                | Responsible team | Implementation status in source code                                                                                                                                          |
| :--------------------------------------------------------------- | :----------------------------------------------------------------------------------- | :------------------ | :--------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_prediction_error` *(demo code: `prediction_error`)* | Absolute error between the prediction and actual label: `abs(prediction - actual)` | µg/m³ (Histogram) | AI / MLOps       | **IMPLEMENTED IN CODE** (`infra/observability/metrics_demo.py`)                                                                                                       |
| `model_mae`                                                    | Mean Absolute Error over the observation window                                      | µg/m³ (Gauge)     | AI               | TO BE FILLED IN *(Calculated in Grafana using PromQL: `sum(rate(prediction_error_sum[1m])) / sum(rate(prediction_error_count[1m]))`; not directly exported as a metric.)* |
| `model_rmse`                                                   | Root Mean Squared Error over the observation window                                  | µg/m³ (Gauge)     | AI               | TO BE FILLED IN *(Not implemented in code yet.)*                                                                                                                            |
| `drift_detected_total` *(docs: `drift_alerts_total`)*      | Total number of detected data or concept drift alerts                                | Events (Counter)    | AI               | TO BE FILLED IN *(Proposed in documentation; no code implementation yet.)*                                                                                                  |
| `model_retrain_total` *(docs: `retrain_events_total`)*     | Total number of model update / retraining events                                     | Events (Counter)    | MLOps            | Counter tracking successful model promotion events to @champion. |
| `model_retrain_duration_seconds`                               | Duration of a retraining run                                                         | Seconds (Histogram) | MLOps            | Duration of a retraining run measured via time.process_time() and exported via OTel Collector. |
| `model_prediction_duration_seconds`                            | Processing time for one model prediction                                             | Seconds (Histogram) | MLOps            | Processing latency for one /predict request in seconds (Histogram). |
| `model_predictions_total`                                      | Total number of predictions made                                                     | Requests (Counter)  | MLOps            | Total number of prediction requests served (Counter). |
| `process_cpu_seconds_total`                                    | Total CPU time consumed by the process                                               | Seconds (Counter)   | MLOps            | Exported via Prometheus process collector / container cgroup metrics. |
| `process_resident_memory_bytes`                                | Resident RAM currently used by the process                                           | Bytes (Gauge)       | MLOps            | Exported via Prometheus process collector / container cgroup metrics. |
| `data_missing_rate`                                            | Proportion of missing data in a batch or time window                                 | % (Gauge)           | Data Pipeline    | Computed offline via `ingest/verify_vietnam_data.py` (to be exported via OTel in streaming pipeline)                                                                         |
| `data_late_records_total`                                      | Total number of records arriving later than their expected time                      | Records (Counter)   | Data Pipeline    | Tracked in streaming ingestion phase (Sprint 2)                                                                                                                               |

#### 5.2. Metric Labels

| Label             | Data type | Description                                                                        |
| :---------------- | :-------- | :--------------------------------------------------------------------------------- |
| `model_name`    | STRING    | Model identifier (e.g. `pm25-forecasting`)                             |
| `model_version` | STRING    | Active model version string (e.g. `1`, `2`)                 |
| `environment`   | STRING    | Deployment environment (`demo`, `staging`, `production`)                     |
| `station_id`    | STRING    | Monitoring station identifier (e.g. `tiantan`, `wanliu`, `us_embassy`) |

- **Metric collection interval:** Prometheus scrape interval: `5s` (`infra/observability/prometheus.yml`); export interval: `2000ms` (`metrics_demo.py`).
- **Monitoring stack:** Python OpenTelemetry SDK → OpenTelemetry Collector (`otel-collector:4317` gRPC / `8889` Prometheus exporter) → Prometheus (`prometheus:9090`) → Grafana (`grafana:3000`).
- **Dashboard location:** Grafana provisioned dashboard `prediction-error.json` at `http://localhost:3000` (UID: `prediction-error-provisioned`).

#### 5.3. Alert Rules

| Alert Name | Trigger Condition | Responsible Team | Action |
| :--- | :--- | :--- | :--- |
| **High Prediction Error** | Rolling 15-minute MAE > 50 µg/m³ or > 2x baseline MAE | AI | Send alert, investigate model performance, and evaluate whether retraining is required. |
| **Drift Detected** | ADWIN or DDM algorithm signals statistically significant distribution shift | AI | Analyze data/concept drift and trigger model evaluation or retraining if necessary. |
| **Retraining Failed** | Retraining script exits with a non-zero code | MLOps | Send alert to Slack/Email and preserve the current champion model. |
| **High CPU Usage** | Serving container CPU > 80% for 5 minutes | MLOps | Send alert, inspect container load, and scale if needed. |
| **High Memory Usage** | Serving container RAM > 85% for 5 minutes | MLOps | Send alert and check for memory leaks during model loading. |
| **Data Quality Issue** | Missing value rate > 10% or schema validation failure in incoming batch | Data Pipeline | Send alert, investigate data quality issues, and prevent invalid data from entering the pipeline. |

---

## PART 6. AI ↔ MLOPS

### 6. Drift Detection & Retraining Contract

#### 6.1. Drift Event

- **Drift detection team:** AI
- **Event receiving team:** MLOps

*(In the current source code, the `drift/` directory contains only `.gitkeep`. Drift detection experiments are in the Jupyter notebook `notebooks/Elec2_Label_Based_Drift_Detection.ipynb`, using River ADWIN/DDM on the Elec2 dataset. The event payload schema has not yet been standardized for message-queue transmission.)*

| Field                  | Data type | Description                                                           |
| :--------------------- | :-------- | :-------------------------------------------------------------------- |
| `event_id`           | STRING    | TO BE FILLED IN                                                       |
| `event_type`         | STRING    | TO BE FILLED IN                                                       |
| `model_name`         | STRING    | TO BE FILLED IN                                                       |
| `model_version`      | STRING    | TO BE FILLED IN                                                       |
| `detector`           | STRING    | TO BE FILLED IN *(Experiments use `ADWIN` and `DDM`.)*          |
| `drift_type`         | STRING    | TO BE FILLED IN *(Experiments focus on label / performance drift.)* |
| `timestamp`          | TIMESTAMP | TO BE FILLED IN                                                       |
| `trigger_retraining` | BOOLEAN   | TO BE FILLED IN                                                       |

#### 6.2. Retraining Policy

| Item                                  | Agreed Approach                                                                                                     |
| :------------------------------------ | :------------------------------------------------------------------------------------------------------------------ |
| **Retraining trigger**          | TO BE FILLED IN                                                                                                     |
| **Drift threshold**             | TO BE FILLED IN *(The Elec2 notebook experiments use `ADWIN(delta=0.002)` and `DDM(warm_start=30)`.)*         |
| **Periodic retraining**         | TO BE FILLED IN                                                                                                     |
| **Drift-triggered retraining**  | TO BE FILLED IN                                                                                                     |
| **Label delay**                 | TO BE FILLED IN *(Marked as TBD in `docs/CONTRACTS.md`.)*                                                       |
| **Minimum training samples**    | TO BE FILLED IN *(The Elec2 notebook uses a window of 2,000 samples; no PM2.5 requirement has been defined yet.)* |
| **Retraining dataset window**   | TO BE FILLED IN                                                                                                     |
| **Retraining frequency**        | TO BE FILLED IN                                                                                                     |
| **Retraining failure handling** | Preserve current @champion model, log stacktrace, and send alert notification to Slack/Email. |

#### 6.3. Model Replacement

| Item                                        | Requirement     |
| :------------------------------------------ | :-------------- |
| **New model evaluator(s)**            | TO BE FILLED IN |
| **Comparison with current model**     | TO BE FILLED IN |
| **Acceptance criteria**               | TO BE FILLED IN |
| **Authority to promote the champion** | MLOps Lead (Nguyệt) and AI Lead (Thiện) |
| **Rollback condition**                | When model prediction error spikes abnormally or serving latency exceeds 1s; MLOps switches @champion alias to previous stable version via MLflow UI. |
| **Retraining history retention**      | TO BE FILLED IN |

#### 6.4. Retraining Workflow

1. AI detects drift.
2. AI publishes a drift event.
3. MLOps records the event.
4. The policy determines whether retraining is required.
5. AI performs model training.
6. MLOps logs and registers the model.
7. AI and MLOps evaluate the new model.
8. MLOps deploys the new model or rolls back.

*(Current status: TO BE FILLED IN – This workflow has not yet been implemented as an automated process.)*

---

## PART 7. INTEGRATION & WORKING AGREEMENT

### 7. Inter-Team Working Rules

#### 7.1. Parallel Development

| Team                    | Mock component used                                                                                                                     | Actual component to be delivered                                                                             |
| :---------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| **Data Pipeline** | Sample CSV files (`data/sample/beijing_multisite_features.csv`, `openaq_pm25_hanoi_hcmc.csv`, `openmeteo_weather_hanoi_hcmc.csv`) | Standard Feature Table in MinIO Parquet (`s3://raw/`, `s3://features/`) and the data update pipeline     |
| **AI**            | Sample features / dummy data (`data/sample/*.csv`)                                                                                    | Model training script (`models/train.py`) and drift detection code                                         |
| **MLOps**         | Dummy sklearn model (`models/dummy_model_mlflow.py`)                                                                                  | Model Registry (MLflow), Serving API (`serving/main.py`), and Monitoring Stack (OTel, Prometheus, Grafana) |

#### 7.2. Repository Conventions

*(Based on `CONTRIBUTING.md` and `docs/CONTRACTS.md`.)*

| Item                              | Rule                                                                                                                                                                                                                                 |
| :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Branch naming**           | Format: `<type>/<short-description>`. Types: `feat`, `fix`, `docs`, `test`, `ci`, `chore`, `refactor` (lowercase and concise). Example: `feat/kafka-ingestion`                                                     |
| **Commit format**           | Conventional Commits: `<type>(<scope>): <description>`. Example: `feat(ingest): add Kafka producer`, `fix(models): correct evaluation`                                                                                         |
| **Pull Request template**   | Use the configured template at `.github/PULL_REQUEST_TEMPLATE.md`.                                                                                                                                                                 |
| **Minimum reviewers**       | At least one other team member must approve each Pull Request before it can be merged.                                                                                                                                               |
| **Code formatting**         | Use Ruff (`ruff check .`, `ruff format .`) and pre-commit hooks (`.pre-commit-config.yaml`).                                                                                                                                   |
| **Testing requirements**    | Use Pytest (`pytest -v`). Test files must be located in `tests/` and follow the `test_*.py` naming convention.                                                                                                                 |
| **Contract change process** | Follow these 7 steps: (1) Describe the change → (2) Identify affected teams → (3) Discuss the change → (4) Record it in the documentation → (5) Update mocks/code → (6) Create a PR → (7) Obtain approval from affected teams. |

#### 7.3. Integration Testing

| Test case                                 | Expected result                                 | Responsible team | Current status                                                                                                                     |
| :---------------------------------------- | :---------------------------------------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **Data Pipeline provides features** | Data can be read with the correct schema        | Data Pipeline    | **TESTS AVAILABLE** (`tests/test_feature_contract.py`, `tests/test_vietnam_data.py`, `tests/test_vietnam_pipeline.py`) |
| **AI trains a model**               | Model is trained successfully                   | AI               | **EXPERIMENTS AVAILABLE** (`models/pm25-baseline.ipynb` and `models/dummy_model_mlflow.py`)                              |
| **MLOps registers a model**         | A model version is available in MLflow          | MLOps            | **TEST AVAILABLE** (`models/register_model.py`)                                                                            |
| **API prediction**                  | Prediction is returned using the correct schema | MLOps            | **CODE AVAILABLE** (`serving/main.py` runs with a dummy model)                                                             |
| **Drift event**                     | MLOps receives the event                        | AI / MLOps       | TO BE FILLED IN *(No integration code yet.)*                                                                                     |
| **Retraining**                      | A new model is logged                           | AI / MLOps       | TO BE FILLED IN *(No automated pipeline yet.)*                                                                                   |
| **Monitoring**                      | Metrics are displayed in Grafana                | MLOps            | **TESTED** (OTel/Prometheus/Grafana stack + `prediction-error.json` dashboard + `metrics_demo.py`)                       |

---

## PART 8. DECISION LOG

### 8. Agreed Decisions

| No. | Topic                        | Final decision                                                                                                                                                                                                                                                                                                           | Confirming teams         | Date            |
| :-- | :--------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------- | :-------------- |
| 1   | **Feature Table**      | Standardize 8 base columns (`station_id`, `event_time`, `pm25_ug_m3`, `temperature_c`, `relative_humidity_pct`, `wind_speed_m_s`, `wind_direction_deg`, `precipitation_mm`) and 15 engineered columns in `pm25_hourly_features` (adding lags 1h/3h/24h, rolling means 3h/24h, `hour`, `weekday`). Use UTC ISO 8601 timestamps and store Parquet files in MinIO S3 (`s3://raw/`, `s3://features/`). Do not impute missing values in the adapter layer. | Data Pipeline, AI, MLOps | 2026-11-21 |
| 2   | **Model Interface**    | Forecast PM2.5 one hour ahead. LightGBM is the primary experimental model; Persistence is the baseline. Evaluation metrics are MAE and RMSE. Store models in MLflow Model format.                                                                                                                                        | AI, MLOps                | 2026-11-21 |
| 3   | **Model Registry**     | Use MLflow with PostgreSQL as the backend store and MinIO S3 as the artifact store (`s3://mlflow/`). Use auto-incrementing versions and the `@champion` alias for production serving.                                                                                                                                | MLOps, AI                | 2026-11-21 |
| 4   | **Serving API**        | Build the API with FastAPI and package it in Docker. Load the model from MLflow Model Registry through the `@champion` alias. Support `/predict`, `/health`, and `/docs`.                                                                                                                                        | MLOps, AI                | 2026-11-21 |
| 5   | **Monitoring Metrics** | Set up Python OpenTelemetry SDK → OTel Collector → Prometheus → Grafana. The first test metric is the `prediction_error` histogram. The Grafana dashboard calculates rolling 1-minute MAE using PromQL.                                                                                                             | MLOps, AI, Data Pipeline | 2026-11-21 |
| 6   | **Drift Event**        | TO BE FILLED IN *(ADWIN and DDM have only been tested in the Elec2 notebook; the event schema has not yet been agreed.)*                                                                                                                                                                                               | AI, MLOps                | TO BE FILLED IN |
| 7   | **Retraining Policy**  | TO BE FILLED IN *(Retraining policy and trigger thresholds have not yet been agreed.)*                                                                                                                                                                                                                                 | AI, MLOps                | TO BE FILLED IN |

### Outstanding Issues

| No. | Issue                                                                                                                                                                                                 | Responsible team(s) | Deadline        |
| :-- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------ | :-------------- |
| 1   | Agree on extended Feature Table features (the `pm25_lag_*` columns, `rainfall`, and derived `hour` / `weekday` fields), as well as missing-value and outlier handling policies for the model. | Data Pipeline & AI  | Resolved (15 core columns + 8 base columns standardized on 2026-11-21) |
| 2   | Finalize the official Request/Response schema for the PM2.5 Serving API, replacing the current dummy Breast Cancer schema.                                                                            | MLOps & AI          | Resolved (Implemented in serving/main.py on 2026-11-21) |
| 3   | Finalize the Drift Event schema over the Message Queue (Kafka/Redpanda) and the conditions for triggering automated retraining.                                                                       | AI & MLOps          | TO BE FILLED IN |

---

### Final Approval

| Team                    | Representative                         | Approval | Approval date   |
| :---------------------- | :------------------------------------- | :------- | :-------------- |
| **Data Pipeline** | Nguyễn Quốc Tường (`@ngynqt`)          | [x]      | 2026-11-21      |
| **AI**            | Võ Trần Khiêm (`@KhiemVoTran`)         | [ ]      | TO BE FILLED IN |
| **MLOps**         | Minh Nguyệt (`@MinhNguyetDuckoo`)      | [x]      | 2026-11-21      |

**Final status:** ☐ Draft  ☑ Approved  
**GitHub PR:** #20, #28, #29  
**Contract completion date:** 2026-11-21  
