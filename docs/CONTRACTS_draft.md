# INTER-TEAM TECHNICAL AGREEMENT
## PROJECT: Adaptive Machine Learning Model Updating

---

### Thông tin chung

**Mục tiêu:** Thống nhất các giao diện dữ liệu, mô hình, triển khai và giám sát để ba team có thể phát triển song song mà không phụ thuộc vào việc hoàn thành của nhau.

---

## PHẦN 1. DATA PIPELINE → AI → MLOPS

### 1. Data Contract – Feature Table
*Team phụ trách: Data Pipeline | Team xác nhận: AI, MLOps*

#### 1.1. Thông tin nguồn dữ liệu

| Nội dung | Thông tin thống nhất |
| :--- | :--- |
| **Dataset sử dụng** | 1. **Historical Benchmark:** Beijing Multi-Site Air Quality Dataset & Elec2 Dataset<br>2. **Live / Vietnam Air Quality:** OpenAQ (PM2.5) + Open-Meteo ERA5 (Weather) |
| **Nguồn dữ liệu** | - Beijing Air Quality: Kaggle (`sid321axn/beijing-multisite-airquality-data-set`), UCI (`10.24432/C5RK5G`)<br>- Elec2: River ML datasets (`river.datasets.Elec2`)<br>- Vietnam PM2.5: OpenAQ API<br>- Vietnam Weather: Open-Meteo API (Historical / Forecast Weather model `era5`) |
| **Thành phố / khu vực** | - Beijing: 12 trạm quan trắc tại Bắc Kinh<br>- Vietnam: Hà Nội và TP. Hồ Chí Minh (HCMC)<br>- Elec2: New South Wales (Úc) |
| **Khoảng thời gian dữ liệu** | - Beijing: 01/03/2013 – 28/02/2017 (theo giờ)<br>- Vietnam: 14/11/2024 – 14/11/2026 (2 năm theo `vietnam_pipeline.toml`); dữ liệu mẫu 1 tuần: 15/11/2025 – 22/11/2025 |
| **Định dạng dữ liệu** | Parquet (lưu trữ xử lý chính) / CSV (dữ liệu mẫu sample / trích xuất) |
| **Nơi lưu trữ** | MinIO S3 object storage (buckets: `raw`, `staging`, `features`) tại `http://minio:9000` (nội bộ Docker) hoặc `http://localhost:9000` (host); cục bộ tại `data/raw/` và `data/sample/` |
| **Tần suất cập nhật** | Hàng giờ (Hourly) |
| **Người phụ trách** | CẦN ĐIỀN |

#### 1.2. Feature Table Schema
- **Tên bảng:** `pm25_hourly_features`
- **Mô tả:** Bảng feature hàng giờ chứa thông tin nồng độ PM2.5 lịch sử và các yếu tố khí tượng phục vụ bài toán dự báo PM2.5.

| STT | Column name | Data type | Unit | Nullable? | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `timestamp` *(trong code dùng `event_time`)* | TIMESTAMP (ISO 8601 UTC) | UTC | No | Thời điểm quan sát (làm tròn đến giờ) |
| 2 | `station_id` | STRING | - | No | Mã định danh trạm quan trắc |
| 3 | `pm25` *(trong code dùng `pm25_ug_m3`)* | FLOAT | µg/m³ | Yes | Nồng độ bụi PM2.5 đo được (giữ nguyên NaN khi thiếu) |
| 4 | `pm25_lag_1h` | CẦN ĐIỀN *(đã có thử nghiệm `pm25_lag1` trong notebook baseline nhưng chưa có trong schema contract của Data Pipeline)* | µg/m³ | CẦN ĐIỀN | Nồng độ PM2.5 trễ 1 giờ |
| 5 | `pm25_lag_3h` | CẦN ĐIỀN *(đã có thử nghiệm `pm25_lag3` trong notebook baseline nhưng chưa có trong schema contract của Data Pipeline)* | µg/m³ | CẦN ĐIỀN | Nồng độ PM2.5 trễ 3 giờ |
| 6 | `pm25_lag_24h` | CẦN ĐIỀN *(đã có thử nghiệm `pm25_lag24` trong notebook baseline nhưng chưa có trong schema contract của Data Pipeline)* | µg/m³ | CẦN ĐIỀN | Nồng độ PM2.5 trễ 24 giờ |
| 7 | `temperature` *(trong code dùng `temperature_c`)* | FLOAT | °C | Yes | Nhiệt độ không khí |
| 8 | `humidity` *(trong code dùng `relative_humidity_pct`)* | FLOAT | % | Yes | Độ ẩm tương đối |
| 9 | `wind_speed` *(trong code dùng `wind_speed_m_s`)* | FLOAT | m/s | Yes | Tốc độ gió |
| 10 | `rainfall` | CẦN ĐIỀN *(chưa có trong 7 cột contract chuẩn và chưa được thu thập trong pipeline hiện tại)* | CẦN ĐIỀN | CẦN ĐIỀN | Lượng mưa |
| 11 | `hour` | CẦN ĐIỀN *(đã có thử nghiệm trong notebook baseline trích xuất từ timestamp, chưa có trong schema bàn giao của Data Pipeline)* | Giờ (0-23) | CẦN ĐIỀN | Giờ trong ngày theo giờ địa phương |
| 12 | `weekday` | CẦN ĐIỀN *(đã có thử nghiệm `dayofweek` trong notebook baseline, chưa có trong schema bàn giao của Data Pipeline)* | 0-6 | CẦN ĐIỀN | Ngày trong tuần |

*(Ghi chú: 7 cột chuẩn hiện tại được cài đặt trong `ingest/feature_contract.py` gồm: `station_id`, `event_time`, `pm25_ug_m3`, `temperature_c`, `relative_humidity_pct`, `wind_speed_m_s`, `wind_direction_deg`).*

#### 1.3. Quy định dữ liệu

| Vấn đề | Phương án thống nhất |
| :--- | :--- |
| **Timestamp format** | ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`) |
| **Timezone** | UTC (bắt buộc timezone-aware, các nguồn địa phương phải chuyển đổi chuẩn hóa về UTC) |
| **Missing values** | Giữ nguyên `NaN` đối với các trường số học, không tự ý điền (impute) hay drop âm thầm ở tầng schema adapter |
| **Duplicate records** | Xác định dựa trên cặp khóa `(station_id, event_time)`. Không chấp nhận bản ghi trùng lặp (raise lỗi nếu phát hiện) |
| **Outliers** | CẦN ĐIỀN *(Hiện tại mã nguồn kiểm tra tính hợp lệ vật lý như nhiệt độ/độ ẩm, nếu xuất hiện giá trị vô hạn nonfinite sẽ báo lỗi)* |
| **Late-arriving data** | Giữ lại bản ghi trễ (không được loại bỏ âm thầm), liên kết với station và target_time tương ứng. Mô hình chỉ tính lỗi khi nhãn thực tế xuất hiện |
| **Dữ liệu cập nhật lại** | CẦN ĐIỀN *(MinIO upload hiện tại bỏ qua file trùng trừ khi dùng cờ `--force`)* |
| **Data validation** | Kiểm tra schema, kiểu dữ liệu, khóa chính, mốc thời gian tròn giờ thông qua `ingest/feature_contract.py` và kiểm tra độ đầy đủ/khoảng trống dữ liệu bằng `verify_vietnam_data.py` |

#### 1.4. Bàn giao giữa các team
- **Data Pipeline cung cấp dữ liệu tại:** MinIO bucket `features` (`s3://features/`) hoặc `data/sample/` (`openaq_pm25_hanoi_hcmc.csv`, `openmeteo_weather_hanoi_hcmc.csv`, `beijing_multisite_features.csv`), Parquet tại `s3://raw/beijing_pm25/`.
- **AI đọc dữ liệu bằng:** Pandas / PyArrow (`pd.read_parquet()`, `pd.read_csv()`).
- **MLOps truy cập dữ liệu bằng:** MinIO S3 API qua Boto3 hoặc mount shared volume.
- **Tần suất cập nhật:** Hàng giờ (Hourly).
- **Xác nhận:** Data Pipeline: CẦN ĐIỀN &nbsp;&nbsp;|&nbsp;&nbsp; AI: CẦN ĐIỀN &nbsp;&nbsp;|&nbsp;&nbsp; MLOps: CẦN ĐIỀN

---

## PHẦN 2. AI → MLOPS

### 2. Model Interface Contract
*Team phụ trách: AI | Team xác nhận: MLOps*

#### 2.1. Thông tin mô hình

| Nội dung | Phương án thống nhất |
| :--- | :--- |
| **Bài toán** | PM2.5 Forecasting (Dự báo nồng độ bụi mịn PM2.5) |
| **Model algorithm** | - Thử nghiệm chính: LightGBM Regressor (`LGBMRegressor`)<br>- Baseline so sánh: Persistence model (lấy giá trị PM2.5 hiện tại làm dự báo tương lai)<br>- Mock component hiện tại của MLOps: `RandomForestClassifier` (trên tập Breast Cancer) |
| **Framework** | LightGBM / Scikit-Learn |
| **Ngôn ngữ / phiên bản** | Python 3.11 |
| **Input features** | Hiện tại notebook baseline thử nghiệm: `pm25_ug_m3`, `hour`, `dayofweek`, các lag `pm25_lag1`, `pm25_lag2`, `pm25_lag3`, `pm25_lag6`, `pm25_lag12`, `pm25_lag24`, cùng các lag thời tiết (`temperature_c_lag1`, `relative_humidity_pct_lag1`, `wind_speed_m_s_lag1`, `wind_direction_deg_lag1`). Feature contract bàn giao chính thức: CẦN ĐIỀN |
| **Prediction target** | `predicted_pm25_ug_m3` (Nồng độ PM2.5 tại thời điểm dự báo) |
| **Forecast horizon** | 1 giờ (1-hour ahead forecast: `target_time = event_time + 1h`) |
| **Output type** | FLOAT (Số thực không âm: `np.maximum(prediction, 0)`) |
| **Evaluation metrics** | MAE (Mean Absolute Error) và RMSE (Root Mean Squared Error) |
| **Training frequency** | CẦN ĐIỀN |

#### 2.2. Training Interface
- **Tên hàm:** `train_model(train_data, config=None)` *(được đề xuất trong `docs/CONTRACTS.md`, chưa triển khai thành module code riêng)*
- **Input:**

| Parameter | Data type | Description |
| :--- | :--- | :--- |
| `train_data` | DataFrame | Tập dữ liệu huấn luyện chứa features và target |
| `config` | DICT / CẦN ĐIỀN | Cấu hình huấn luyện mô hình (hyperparameters, optional) |

- **Output:**

| Field | Data type | Description |
| :--- | :--- | :--- |
| `model` | OBJECT | Mô hình đã được huấn luyện |
| `metrics` | DICT | Kết quả đánh giá trên tập kiểm tra (MAE, RMSE, ...) |
| `params` | DICT / CẦN ĐIỀN | Các siêu tham số đã sử dụng |

- **Quy ước train:**
  - **Cách chia train/test:** Time-based split (chia theo thứ tự thời gian; cutoff cụ thể phân tách train/test, không chia ngẫu nhiên)
  - **Có shuffle hay không:** Không (Shuffle = False để tránh rò rỉ dữ liệu chuỗi thời gian)
  - **Random seed:** `42`
  - **Cách lưu model:** MLflow Model format (`mlflow.sklearn.log_model` hoặc flavor tương ứng)

#### 2.3. Prediction Interface
- **Tên hàm:** `predict(model, input_data)` *(được đề xuất trong `docs/CONTRACTS.md`, chưa triển khai module riêng)*

| Nội dung | Thống nhất |
| :--- | :--- |
| **Input schema** | CẦN ĐIỀN *(Serving hiện tại nhận dummy 30 features Breast Cancer, chưa hoàn thiện cho schema PM2.5)* |
| **Input format** | JSON / DataFrame |
| **Output schema** | `station_id` (STRING), `target_time` (TIMESTAMP), `predicted_pm25_ug_m3` (FLOAT), `model_version` (STRING) *(theo `docs/CONTRACTS.md`)* |
| **Output format** | JSON |
| **Batch prediction** | Có |
| **Missing feature handling** | CẦN ĐIỀN |

- **Ví dụ Input:**
  ```json
  {"pm25": 35.2, "pm25_lag_1h": 32.1}
  ```
- **Ví dụ Output:**
  ```json
  {"prediction": 36.5}
  ```

#### 2.4. Model Artifact

| Nội dung | Phương án thống nhất |
| :--- | :--- |
| **Model format** | MLflow Model format |
| **Artifact storage** | MinIO S3 (`s3://mlflow/` hoặc `s3://models/`) |
| **Model serialization** | MLflow flavors (cloudpickle / joblib qua `mlflow.sklearn` hoặc `mlflow.lightgbm`) |
| **Model signature** | Bắt buộc (Inferred qua `mlflow.models.infer_signature`) |
| **Dependencies** | File cấu hình `requirements.txt` / conda.yaml sinh tự động bởi MLflow và lockfile dự án `requirements-py311.lock` |
| **Metadata cần bàn giao** | `model_name`, `model_version`, `training_timestamp`, `dataset_version`, `model_strategy`, `feature_schema_version`, `mae`, `rmse`, `r2` |

#### 2.5. Model Evaluation & Acceptance

| Tiêu chí | Quy định |
| :--- | :--- |
| **Evaluation metrics** | MAE (µg/m³), RMSE (µg/m³) |
| **Baseline model** | Persistence model |
| **Điều kiện đạt yêu cầu** | Vượt qua Baseline model (giảm MAE và RMSE so với mô hình Persistence) |
| **So sánh với champion** | CẦN ĐIỀN *(Chưa có quy định ngưỡng cải thiện cụ thể)* |
| **Người phê duyệt model** | CẦN ĐIỀN |

**AI bàn giao cho MLOps:**
- [x] Model artifact
- [x] Model signature
- [x] Training parameters
- [x] Evaluation metrics
- [x] Dataset version / reference
- [x] Source code / Git commit
- [ ] Hướng dẫn load và predict *(CẦN ĐIỀN)*
- [ ] Test input/output *(CẦN ĐIỀN)*

*Xác nhận: AI: CẦN ĐIỀN &nbsp;&nbsp;|&nbsp;&nbsp; MLOps: CẦN ĐIỀN*

---

## PHẦN 3. MLOPS

### 3. Model Registry Contract
*Team phụ trách: MLOps | Team xác nhận: AI*

#### 3.1. MLflow Configuration

| Nội dung | Phương án thống nhất |
| :--- | :--- |
| **MLflow Tracking URI** | `http://mlflow:5000` (nội bộ Docker) hoặc `http://localhost:5000` (host) |
| **Backend Store** | PostgreSQL (`postgresql+psycopg2://mlflow:mlflow123@postgres:5432/mlflow`) |
| **Artifact Store** | MinIO (`s3://mlflow/` với endpoint `http://minio:9000`) |
| **Experiment name** | `pm25_forecasting` *(đề xuất trong `docs/CONTRACTS.md`; hiện tại mock code dùng `adaptive-model-updating`)* |
| **Registered Model name** | `pm25-forecasting` *(đề xuất trong `docs/CONTRACTS.md`; hiện tại mock code dùng `BreastCancerRandomForest`)* |
| **Model versioning** | MLflow Version (tự tăng: 1, 2, 3...) |
| **Model artifact format** | MLflow Model |
| **Model signature** | Bắt buộc |

#### 3.2. Quy tắc đặt tên

| Đối tượng | Naming convention | Ví dụ |
| :--- | :--- | :--- |
| **Experiment** | snake_case | `pm25_forecasting` |
| **Registered Model** | kebab-case | `pm25-forecasting` |
| **Model version** | Số nguyên tự tăng | `1`, `2` |
| **Run name** | `<model_type>-<purpose>` | `lightgbm-baseline`, `random-forest-baseline` |
| **Artifact** | Theo chuẩn MLflow Model | `model/model.pkl`, `MLmodel` |

#### 3.3. Model Lifecycle

| Trạng thái / Alias | Ý nghĩa | Điều kiện chuyển |
| :--- | :--- | :--- |
| **Candidate** | Model mới được log và đăng ký, chờ đánh giá | Đã đăng ký vào MLflow Registry |
| **Champion** *(Alias `@champion`)* | Model đang phục vụ sản xuất | Đạt tiêu chí đánh giá và vượt qua model hiện tại |
| **Archived** | Model cũ không còn phục vụ | CẦN ĐIỀN *(Khi có model mới thay thế làm champion)* |

**Quy trình phê duyệt model mới:**
1. AI hoàn thành training.
2. Model được log vào MLflow (kèm params, metrics, signature, artifact).
3. MLOps kiểm tra artifact và signature.
4. AI và MLOps đánh giá kết quả (so sánh với baseline và champion).
5. Model đạt tiêu chí được gán alias `champion`.
6. Serving nạp và sử dụng model mới.

**Người có quyền phê duyệt:** CẦN ĐIỀN

---

## PHẦN 4. MLOPS → AI / DATA PIPELINE

### 4. Model Serving Contract
*Team phụ trách: MLOps | Team sử dụng: AI, Data Pipeline*

#### 4.1. API Configuration

| Nội dung | Thông tin thống nhất |
| :--- | :--- |
| **Service name** | Prediction API *(tiêu đề Swagger trong `serving/main.py`: "Prediction API")* |
| **API framework** | FastAPI |
| **Base URL** | `http://localhost:8000` (host) hoặc `http://api:8000` (nội bộ Docker) |
| **Port** | 8000 |
| **API endpoint** | `/predict` (dự đoán), `/health` (kiểm tra trạng thái), `/docs` (Swagger UI) |
| **HTTP method** | POST (cho `/predict`), GET (cho `/health`, `/`) |
| **Content-Type** | `application/json` |
| **Authentication** | CẦN ĐIỀN *(Hiện tại chưa cấu hình auth trong `serving/main.py`)* |
| **Timeout** | CẦN ĐIỀN |

#### 4.2. API Request
*(Hiện tại `serving/main.py` nhận `PredictionRequest(features: list[list[float]])` phục vụ dummy model Breast Cancer. Định dạng chuẩn theo template cần được thống nhất lại như sau:)*

| Field | Data type | Required | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | STRING | CẦN ĐIỀN *(Hiện tại API đọc MODEL_NAME từ biến môi trường)* | Tên model phục vụ dự đoán |
| `features` | OBJECT / ARRAY | Có | Dữ liệu đầu vào phục vụ dự đoán |
| `timestamp` | STRING | CẦN ĐIỀN *(Hiện tại schema serving chưa có trường timestamp)* | Mốc thời gian gửi yêu cầu |

#### 4.3. API Response
*(Hiện tại `serving/main.py` trả về `model_name: str` và `predictions: list[int]`. Các trường dưới đây theo template đang ở trạng thái:)*

| Field | Data type | Description |
| :--- | :--- | :--- |
| `model_name` | STRING | Tên mô hình phục vụ dự đoán |
| `model_version` | STRING | CẦN ĐIỀN *(Hiện tại response chưa có trường model_version)* |
| `prediction` | FLOAT | CẦN ĐIỀN *(Hiện tại response trả về list int của mô hình phân loại dummy)* |
| `request_id` | STRING | CẦN ĐIỀN *(Hiện tại response chưa có trường request_id)* |
| `status` | STRING | CẦN ĐIỀN *(Hiện chỉ có trường status trong endpoint `/health`)* |

#### 4.4. Error Handling

| HTTP Status | Trường hợp | Cách xử lý |
| :--- | :--- | :--- |
| **400** | Invalid input | CẦN ĐIỀN *(Hiện tại chưa có custom handler 400 trong `serving/main.py`)* |
| **404** | Model not found | CẦN ĐIỀN |
| **422** | Invalid schema / Unprocessable Entity | Trả về chi tiết lỗi schema validation từ FastAPI/Pydantic (hoặc khi kích thước features không đúng số chiều yêu cầu) |
| **500** | Internal error | Bắt ngoại lệ, ghi log lỗi hệ thống và trả về `{"detail": str(exc)}` |
| **503** | Model unavailable | Trả về `{"detail": "Model is not loaded"}` khi model chưa được khởi tạo thành công |

#### 4.5. Model Loading

| Nội dung | Phương án |
| :--- | :--- |
| **Load model from** | MLflow Model Registry qua URI: `models:/{MODEL_NAME}@{MODEL_ALIAS}` hoặc `MODEL_URI` |
| **Model version selection** | Thông qua Model Alias (mặc định lấy alias `champion` hoặc biến môi trường `MODEL_ALIAS`) |
| **Model alias** | `champion` |
| **Reload model khi có version mới** | CẦN ĐIỀN *(Hiện tại model chỉ load 1 lần lúc startup container qua lifespan event)* |
| **Rollback model** | CẦN ĐIỀN *(Hiện tại chuyển alias trên MLflow Registry thủ công)* |

*Xác nhận: AI: CẦN ĐIỀN &nbsp;&nbsp;|&nbsp;&nbsp; MLOps: CẦN ĐIỀN*

---

## PHẦN 5. ALL TEAMS

### 5. Monitoring & Observability Contract
*Team phụ trách hạ tầng: MLOps | Team cung cấp tín hiệu: AI, Data Pipeline*

#### 5.1. Monitoring Metrics

| Metric name | Description | Unit | Responsible team | Trạng thái triển khai trong mã nguồn |
| :--- | :--- | :--- | :--- | :--- |
| `model_prediction_error`<br>*(code demo: `prediction_error`)* | Sai số tuyệt đối giữa giá trị dự báo và nhãn thực tế: `abs(prediction - actual)` | µg/m³ (Histogram) | AI / MLOps | **ĐÃ CÓ TRONG CODE** (`infra/observability/metrics_demo.py`) |
| `model_mae` | Mean Absolute Error trong cửa sổ quan sát | µg/m³ (Gauge) | AI | CẦN ĐIỀN *(Grafana tính toán qua PromQL: `sum(rate(prediction_error_sum[1m])) / sum(rate(prediction_error_count[1m]))`, chưa có metric đẩy trực tiếp)* |
| `model_rmse` | Root Mean Squared Error trong cửa sổ quan sát | µg/m³ (Gauge) | AI | CẦN ĐIỀN *(Chưa triển khai trong code)* |
| `drift_detected_total`<br>*(docs: `drift_alerts_total`)* | Số lần phát hiện dữ liệu hoặc khái niệm bị trôi (drift alert) | Events (Counter) | AI | CẦN ĐIỀN *(Mới có trong docs đề xuất, chưa có code)* |
| `model_retrain_total`<br>*(docs: `retrain_events_total`)* | Tổng số sự kiện cập nhật / huấn luyện lại mô hình | Events (Counter) | MLOps | CẦN ĐIỀN *(Mới có trong docs đề xuất, chưa có code)* |
| `model_retrain_duration_seconds` | Thời gian thực thi một lần huấn luyện lại | Seconds (Histogram) | MLOps | CẦN ĐIỀN *(Chưa triển khai trong code)* |
| `model_prediction_duration_seconds` | Thời gian xử lý một lượt dự báo của mô hình | Seconds (Histogram) | MLOps | CẦN ĐIỀN *(Chưa triển khai trong code)* |
| `model_predictions_total` | Tổng số lượt dự báo đã thực hiện | Requests (Counter) | MLOps | CẦN ĐIỀN *(Chưa triển khai trong code)* |
| `process_cpu_seconds_total` | Tổng thời gian CPU đã tiêu thụ bởi tiến trình | Seconds (Counter) | MLOps | CẦN ĐIỀN *(Mới có trong docs đề xuất, chưa export trong code)* |
| `process_resident_memory_bytes` | Dung lượng bộ nhớ RAM tiến trình đang chiếm dụng | Bytes (Gauge) | MLOps | CẦN ĐIỀN *(Mới có trong docs đề xuất, chưa export trong code)* |
| `data_missing_rate` | Tỷ lệ dữ liệu bị thiếu trong batch hoặc khoảng thời gian | % (Gauge) | Data Pipeline | CẦN ĐIỀN *(Chưa triển khai trong code)* |
| `data_late_records_total` | Tổng số bản ghi dữ liệu bị trễ so với mốc thời gian | Records (Counter) | Data Pipeline | CẦN ĐIỀN *(Chưa triển khai trong code)* |

#### 5.2. Metric Labels

| Label | Data type | Description |
| :--- | :--- | :--- |
| `model_name` | STRING | Tên định danh mô hình (đã dùng trong `metrics_demo.py`) |
| `model_version` | STRING | Phiên bản mô hình (CẦN ĐIỀN trong ứng dụng thực tế) |
| `environment` | STRING | Môi trường triển khai (`demo`, `staging`, `production`) |
| `station_id` | STRING | Mã định danh trạm quan trắc (CẦN ĐIỀN trong ứng dụng thực tế) |

- **Metric collection interval:** Scrape interval `5s` (`infra/observability/prometheus.yml`); Export interval `2000ms` (`metrics_demo.py`).
- **Monitoring stack:** Python OpenTelemetry SDK → OpenTelemetry Collector (`otel-collector:4317` gRPC / `8889` Prometheus exporter) → Prometheus (`prometheus:9090`) → Grafana (`grafana:3000`).
- **Dashboard location:** Grafana Provisioned Dashboard `prediction-error.json` tại `http://localhost:3000` (UID: `prediction-error-provisioned`).

#### 5.3. Alert Rules

| Alert name | Trigger condition | Responsible team | Action |
| :--- | :--- | :--- | :--- |
| **High prediction error** | CẦN ĐIỀN | AI | CẦN ĐIỀN |
| **Drift detected** | CẦN ĐIỀN | AI | CẦN ĐIỀN |
| **Retraining failed** | CẦN ĐIỀN | MLOps | CẦN ĐIỀN |
| **High CPU usage** | CẦN ĐIỀN | MLOps | CẦN ĐIỀN |
| **High memory usage** | CẦN ĐIỀN | MLOps | CẦN ĐIỀN |
| **Data quality issue** | CẦN ĐIỀN | Data Pipeline | CẦN ĐIỀN |

---

## PHẦN 6. AI ↔ MLOPS

### 6. Drift Detection & Retraining Contract

#### 6.1. Drift Event
- **Team phát hiện drift:** AI
- **Team tiếp nhận event:** MLOps
*(Hiện tại trong source code, thư mục `drift/` mới chỉ có `.gitkeep`. Các thử nghiệm phát hiện drift đang nằm trong Jupyter notebook `notebooks/Elec2_Label_Based_Drift_Detection.ipynb` sử dụng River ADWIN/DDM trên tập Elec2, chưa chuẩn hóa thành payload schema qua message queue)*

| Field | Data type | Description |
| :--- | :--- | :--- |
| `event_id` | STRING | CẦN ĐIỀN |
| `event_type` | STRING | CẦN ĐIỀN |
| `model_name` | STRING | CẦN ĐIỀN |
| `model_version` | STRING | CẦN ĐIỀN |
| `detector` | STRING | CẦN ĐIỀN *(Thử nghiệm dùng `ADWIN`, `DDM`)* |
| `drift_type` | STRING | CẦN ĐIỀN *(Thử nghiệm drift nhãn / hiệu năng)* |
| `timestamp` | TIMESTAMP | CẦN ĐIỀN |
| `trigger_retraining` | BOOLEAN | CẦN ĐIỀN |

#### 6.2. Retraining Policy

| Nội dung | Phương án thống nhất |
| :--- | :--- |
| **Retraining trigger** | CẦN ĐIỀN |
| **Drift threshold** | CẦN ĐIỀN *(Trong notebook Elec2 thử nghiệm `ADWIN(delta=0.002)`, `DDM(warm_start=30)`)* |
| **Periodic retraining** | CẦN ĐIỀN |
| **Drift-triggered retraining** | CẦN ĐIỀN |
| **Label delay** | CẦN ĐIỀN *(Được xác định là TBD trong `docs/CONTRACTS.md`)* |
| **Minimum training samples** | CẦN ĐIỀN *(Trong notebook Elec2 dùng window 2000 mẫu; PM2.5 chưa có quy định)* |
| **Retraining dataset window** | CẦN ĐIỀN |
| **Retraining frequency** | CẦN ĐIỀN |
| **Retraining failure handling** | CẦN ĐIỀN |

#### 6.3. Model Replacement

| Nội dung | Quy định |
| :--- | :--- |
| **Model mới được đánh giá bởi** | CẦN ĐIỀN |
| **So sánh với model hiện tại** | CẦN ĐIỀN |
| **Acceptance criteria** | CẦN ĐIỀN |
| **Quyền chuyển champion** | CẦN ĐIỀN |
| **Rollback condition** | CẦN ĐIỀN |
| **Lưu lịch sử retraining** | CẦN ĐIỀN |

#### 6.4. Retraining Workflow
1. AI phát hiện drift.
2. AI gửi drift event.
3. MLOps ghi nhận event.
4. Policy quyết định có retraining hay không.
5. AI thực hiện training.
6. MLOps log và register model.
7. AI và MLOps đánh giá model mới.
8. MLOps triển khai hoặc rollback.
*(Trạng thái hiện tại: CẦN ĐIỀN - Chưa có code triển khai tự động quy trình này).*

---

## PHẦN 7. INTEGRATION & WORKING AGREEMENT

### 7. Quy định làm việc giữa các team

#### 7.1. Quy định phát triển song song

| Team | Mock component sử dụng | Thành phần thật cần bàn giao |
| :--- | :--- | :--- |
| **Data Pipeline** | Sample CSV (`data/sample/beijing_multisite_features.csv`, `openaq_pm25_hanoi_hcmc.csv`, `openmeteo_weather_hanoi_hcmc.csv`) | Feature Table chuẩn trên MinIO Parquet (`s3://raw/`, `s3://features/`) và pipeline cập nhật |
| **AI** | Sample features / Dummy data (`data/sample/*.csv`) | Model training script (`models/train.py`) + Drift Event detection code |
| **MLOps** | Dummy sklearn model (`models/dummy_model_mlflow.py`) | Model Registry (MLflow) + Serving API (`serving/main.py`) + Monitoring Stack (OTel, Prometheus, Grafana) |

#### 7.2. Repository Convention
*(Thông tin lấy từ `CONTRIBUTING.md` và `docs/CONTRACTS.md`)*

| Nội dung | Quy định |
| :--- | :--- |
| **Branch naming** | Format: `<type>/<short-description>`. Các types: `feat`, `fix`, `docs`, `test`, `ci`, `chore`, `refactor` (chữ thường, ngắn gọn). Ví dụ: `feat/kafka-ingestion` |
| **Commit format** | Conventional Commits: `<type>(<scope>): <description>`. Ví dụ: `feat(ingest): add Kafka producer`, `fix(models): correct evaluation` |
| **Pull Request template** | Sử dụng file template cấu hình sẵn tại `.github/PULL_REQUEST_TEMPLATE.md` |
| **Minimum reviewers** | Tối thiểu 1 thành viên khác phê duyệt trước khi merge ("Each Pull Request must receive approval from at least one other team member before merging") |
| **Code formatting** | Sử dụng Ruff (`ruff check .`, `ruff format .`) và Pre-commit hooks (`.pre-commit-config.yaml`) |
| **Testing requirements** | Sử dụng Pytest (`pytest -v`), các file test nằm trong thư mục `tests/` theo tiền tố `test_*.py` |
| **Contract change process** | 7 bước theo quy định: (1) Mô tả thay đổi → (2) Xác định team ảnh hưởng → (3) Thảo luận → (4) Ghi nhận vào tài liệu → (5) Cập nhật mock/code → (6) Tạo PR → (7) Thu thập phê duyệt từ các team bị ảnh hưởng |

#### 7.3. Integration Testing

| Test case | Expected result | Responsible | Trạng thái hiện tại |
| :--- | :--- | :--- | :--- |
| **Data Pipeline cung cấp features** | Đọc được đúng schema | Data Pipeline | **ĐÃ CÓ TEST** (`tests/test_feature_contract.py`, `tests/test_vietnam_data.py`, `tests/test_vietnam_pipeline.py`) |
| **AI train model** | Model tạo thành công | AI | **ĐÃ CÓ THỬ NGHIỆM** (`models/pm25-baseline.ipynb` và `models/dummy_model_mlflow.py`) |
| **MLOps register model** | Có model version trong MLflow | MLOps | **ĐÃ CÓ TEST** (`models/register_model.py`) |
| **API predict** | Trả prediction đúng schema | MLOps | **ĐÃ CÓ CODE** (`serving/main.py` chạy trên dummy model) |
| **Drift event** | MLOps nhận được event | AI / MLOps | CẦN ĐIỀN *(Chưa có code tích hợp)* |
| **Retraining** | Model mới được log | AI / MLOps | CẦN ĐIỀN *(Chưa có pipeline tự động)* |
| **Monitoring** | Metrics hiển thị trên Grafana | MLOps | **ĐÃ CÓ TEST** (Stack OTel/Prometheus/Grafana + Dashboard `prediction-error.json` + `metrics_demo.py`) |

---

## PHẦN 8. DECISION LOG

### 8. Các quyết định đã thống nhất

| STT | Nội dung | Quyết định cuối cùng | Team xác nhận | Ngày |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Feature Table** | Thống nhất 7 cột chuẩn: `station_id`, `event_time`, `pm25_ug_m3`, `temperature_c`, `relative_humidity_pct`, `wind_speed_m_s`, `wind_direction_deg`. Múi giờ UTC ISO 8601, lưu trữ Parquet trên MinIO S3 (`s3://raw/`, `s3://features/`). Không tự ý impute missing value ở adapter. | Data Pipeline, AI, MLOps | CẦN ĐIỀN |
| 2 | **Model Interface** | Bài toán PM2.5 forecasting bước trước 1h. Model thử nghiệm chính là LightGBM, baseline so sánh là Persistence model. Tiêu chí đánh giá là MAE và RMSE. Lưu trữ dưới chuẩn MLflow Model. | AI, MLOps | CẦN ĐIỀN |
| 3 | **Model Registry** | Sử dụng MLflow với PostgreSQL backend store và MinIO S3 artifact store (`s3://mlflow/`). Quản lý phiên bản tự tăng, sử dụng alias `@champion` để phục vụ production. | MLOps, AI | CẦN ĐIỀN |
| 4 | **Serving API** | Xây dựng API bằng FastAPI, đóng gói container Docker, nạp model từ MLflow Model Registry qua alias `@champion`. Hỗ trợ các endpoint `/predict`, `/health`, `/docs`. | MLOps, AI | CẦN ĐIỀN |
| 5 | **Monitoring Metrics** | Thiết lập hạ tầng OpenTelemetry SDK → OTel Collector → Prometheus → Grafana. Metric kiểm thử đầu tiên là histogram `prediction_error`. Dashboard Grafana tính MAE rolling 1 phút bằng PromQL. | MLOps, AI, Data Pipeline | CẦN ĐIỀN |
| 6 | **Drift Event** | CẦN ĐIỀN *(Hiện mới chỉ thử nghiệm thuật toán ADWIN và DDM trong notebook Elec2, chưa thống nhất schema sự kiện)* | AI, MLOps | CẦN ĐIỀN |
| 7 | **Retraining Policy** | CẦN ĐIỀN *(Chưa thống nhất chính sách và ngưỡng kích hoạt huấn luyện lại)* | AI, MLOps | CẦN ĐIỀN |

### Những vấn đề chưa thống nhất

| STT | Vấn đề | Team phụ trách giải quyết | Deadline |
| :--- | :--- | :--- | :--- |
| 1 | Thống nhất các features mở rộng của Feature Table (các cột lag `pm25_lag_*`, `rainfall`, trích xuất `hour`, `weekday`) và chính sách xử lý missing value / outlier cho model | Data Pipeline & AI | CẦN ĐIỀN |
| 2 | Thống nhất schema Request/Response chính thức cho Serving API bài toán PM2.5 (thay thế schema dummy Breast Cancer hiện tại) | MLOps & AI | CẦN ĐIỀN |
| 3 | Thống nhất Drift Event Schema qua Message Queue (Kafka/Redpanda) và điều kiện kích hoạt Retraining tự động | AI & MLOps | CẦN ĐIỀN |

---

### Xác nhận cuối cùng

| Team | Đại diện | Approval | Ngày xác nhận |
| :--- | :--- | :--- | :--- |
| **Data Pipeline** | CẦN ĐIỀN | ☐ | CẦN ĐIỀN |
| **AI** | CẦN ĐIỀN | ☐ | CẦN ĐIỀN |
| **MLOps** | CẦN ĐIỀN | ☐ | CẦN ĐIỀN |

**Final status:** ☑ Draft &nbsp;&nbsp;&nbsp; ☐ Under Review &nbsp;&nbsp;&nbsp; ☐ Approved &nbsp;&nbsp;&nbsp; ☐ Needs Revision  
**GitHub PR:** CẦN ĐIỀN  
**Ngày hoàn thành contract:** CẦN ĐIỀN  

### Definition of Done
- [ ] Các team đã điền đầy đủ phần phụ trách.
- [ ] Feature Table schema được Data Pipeline và AI xác nhận.
- [ ] Model Interface được AI và MLOps xác nhận.
- [ ] Metrics được cả ba team thống nhất.
- [ ] Serving và Retraining Interface được thống nhất.
- [ ] Mock components có thể sử dụng độc lập.
- [ ] CONTRACTS.md được merge vào repository.
- [ ] Có ít nhất một thành viên mỗi team phê duyệt.
