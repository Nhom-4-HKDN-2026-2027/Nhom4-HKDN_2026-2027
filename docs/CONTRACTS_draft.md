# INTER-TEAM TECHNICAL AGREEMENT

## DỰ ÁN: Adaptive Machine Learning Model Updating (Cập nhật mô hình học máy thích ứng)

---

## Thông tin chung

**Mục tiêu:** Thống nhất các giao diện dữ liệu, mô hình, triển khai và giám sát để ba nhóm có thể phát triển song song mà không phụ thuộc vào việc hoàn thành của nhau.

---

## PHẦN 1. DATA PIPELINE → AI → MLOPS

### 1. Data Contract – Feature Table

*Nhóm phụ trách: Data Pipeline | Nhóm đánh giá/xác nhận: AI, MLOps*

#### 1.1. Thông tin nguồn dữ liệu

| Hạng mục | Thông tin thống nhất |
| :--- | :--- |
| **Tập dữ liệu** | 1. **Tập chuẩn lịch sử (Historical Benchmark):** Beijing Multi-Site Air Quality Dataset và Elec2 Dataset<br>2. **Dữ liệu thực tế / Không khí Việt Nam:** OpenAQ (PM2.5) + Open-Meteo ERA5 (Thời tiết) |
| **Nguồn dữ liệu** | - Beijing Air Quality: Kaggle (`sid321axn/beijing-multisite-airquality-data-set`), UCI (`10.24432/C5RK5G`)<br>- Elec2: River ML datasets (`river.datasets.Elec2`)<br>- Vietnam PM2.5: OpenAQ API<br>- Vietnam Weather: Open-Meteo API (Mô hình thời tiết lịch sử / dự báo `era5`) |
| **Thành phố / Khu vực** | - Beijing: 12 trạm quan trắc tại Bắc Kinh<br>- Việt Nam: Hà Nội và TP. Hồ Chí Minh (HCMC)<br>- Elec2: New South Wales, Úc |
| **Khoảng thời gian dữ liệu** | - Beijing: 01/03/2013 – 28/02/2017 (theo từng giờ)<br>- Việt Nam: 14/11/2024 – 14/11/2026 (2 năm, theo `vietnam_pipeline.toml`); dữ liệu mẫu 1 tuần: 15/11/2025 – 22/11/2025 |
| **Định dạng dữ liệu** | Parquet (định dạng xử lý và lưu trữ chính) / CSV (dữ liệu mẫu hoặc trích xuất) |
| **Vị trí lưu trữ** | MinIO S3 object storage (các bucket: `raw`, `staging`, `features`) tại `http://minio:9000` (nội bộ Docker) hoặc `http://localhost:9000` (host); lưu trữ cục bộ tại `data/raw/` và `data/sample/` |
| **Tần suất cập nhật** | Hàng giờ (Hourly) |
| **Người phụ trách** | CẦN ĐIỀN |

#### 1.2. Schema Feature Table

- **Tên bảng:** `pm25_hourly_features`
- **Mô tả:** Bảng đặc trưng hàng giờ chứa nồng độ PM2.5 lịch sử, các đặc trưng trễ (lag), đặc trưng trung bình trượt (rolling mean), các biến khí tượng và đặc trưng thời gian phục vụ bài toán dự báo PM2.5.

| STT | Tên cột | Kiểu dữ liệu | Đơn vị | Cho phép Null? | Mô tả |
|:---:|---|---|:---:|---|---|
| 1 | `station_id` | STRING | - | Không | Mã định danh trạm quan trắc |
| 2 | `event_time` | TIMESTAMP (ISO 8601 UTC) | UTC | Không | Thời gian quan sát, làm tròn đến đầu giờ |
| 3 | `pm25_ug_m3` | FLOAT | µg/m³ | Có | Nồng độ PM2.5 đo được tại thời điểm hiện tại |
| 4 | `pm25_lag_1h` | FLOAT | µg/m³ | Có | Nồng độ PM2.5 trễ 1 giờ |
| 5 | `pm25_lag_3h` | FLOAT | µg/m³ | Có | Nồng độ PM2.5 trễ 3 giờ |
| 6 | `pm25_lag_24h` | FLOAT | µg/m³ | Có | Nồng độ PM2.5 trễ 24 giờ |
| 7 | `pm25_rolling_mean_3h` | FLOAT | µg/m³ | Có | Nồng độ PM2.5 trung bình trong 3 giờ trước đó |
| 8 | `pm25_rolling_mean_24h` | FLOAT | µg/m³ | Có | Nồng độ PM2.5 trung bình trong 24 giờ trước đó |
| 9 | `temperature_c` | FLOAT | °C | Có | Nhiệt độ không khí |
| 10 | `relative_humidity_pct` | FLOAT | % | Có | Độ ẩm tương đối |
| 11 | `wind_speed_m_s` | FLOAT | m/s | Có | Tốc độ gió |
| 12 | `wind_direction_deg` | FLOAT | Độ (Degree) | Có | Hướng gió |
| 13 | `hour` | INTEGER | 0–23 | Không | Giờ trong ngày (0 đến 23) |
| 14 | `weekday` | INTEGER | 0–6 | Không | Ngày trong tuần (0: Thứ hai đến 6: Chủ nhật) |

**Ghi chú:**
- Các đặc trưng trễ (lag) và trung bình trượt (rolling mean) phải được tính toán riêng biệt cho từng trạm quan trắc (`station_id`).
- Các đặc trưng trễ biểu diễn quan sát lịch sử và tuyệt đối không được sử dụng dữ liệu tương lai.
- Các đặc trưng trung bình trượt chỉ được tính từ dữ liệu lịch sử để tránh rò rỉ dữ liệu (data leakage).
- Cột `hour` được trích xuất từ `event_time`.
- Cột `weekday` được trích xuất từ `event_time`, trong đó 0 đại diện cho Thứ Hai và 6 đại diện cho Chủ Nhật.
- Dữ liệu lịch sử bị thiếu có thể dẫn đến các giá trị NULL trong các cột lag và rolling mean.
- Schema này là đề xuất hợp đồng feature và cần được hai nhóm Data Pipeline và AI thống nhất trước khi triển khai chính thức.

#### 1.3. Quy định dữ liệu (Data Rules)

| Vấn đề | Phương án thống nhất |
| :--- | :--- |
| **Định dạng timestamp** | ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`) |
| **Múi giờ** | UTC (bắt buộc timestamp có nhận biết múi giờ - timezone-aware; các mốc thời gian địa phương từ nguồn phải chuẩn hóa về UTC) |
| **Dữ liệu thiếu (Missing values)** | Giữ nguyên `NaN` cho các trường số. Không tự ý điền (impute) hay âm thầm xóa bỏ bản ghi thiếu ở tầng adapter schema. |
| **Bản ghi trùng lặp (Duplicates)** | Nhận diện bản ghi dựa trên khóa kết hợp `(station_id, event_time)`. Không cho phép trùng lặp; báo lỗi nếu phát hiện. |
| **Ngoại lai (Outliers)** | CẦN ĐIỀN *(Mã nguồn hiện tại kiểm tra tính hợp lệ vật lý, chẳng hạn như dải nhiệt độ và độ ẩm; các giá trị vô hạn non-finite sẽ văng lỗi).* |
| **Dữ liệu đến muộn (Late-arriving)** | Giữ lại các bản ghi đến muộn; không âm thầm loại bỏ. Gắn kết chúng với trạm và `target_time` tương ứng. Mô hình chỉ tính toán lỗi khi nhãn thực tế xuất hiện. |
| **Cập nhật / sửa đổi dữ liệu** | CẦN ĐIỀN *(Quy trình tải lên MinIO hiện tại bỏ qua file trùng lặp trừ khi sử dụng cờ `--force`).* |
| **Kiểm định dữ liệu (Data validation)** | Kiểm định schema, kiểu dữ liệu, khóa chính và căn chỉnh timestamp theo giờ qua `ingest/feature_contract.py`; kiểm tra độ đầy đủ và khoảng trống dữ liệu bằng `verify_vietnam_data.py`. |

#### 1.4. Bàn giao giữa các nhóm

- **Data Pipeline cung cấp dữ liệu tại:** MinIO bucket `features` (`s3://features/`) hoặc `data/sample/` (`openaq_pm25_hanoi_hcmc.csv`, `openmeteo_weather_hanoi_hcmc.csv`, `beijing_multisite_features.csv`); dữ liệu Parquet Bắc Kinh tại `s3://raw/beijing_pm25/`.
- **AI đọc dữ liệu bằng:** Pandas / PyArrow (`pd.read_parquet()`, `pd.read_csv()`).
- **MLOps truy cập dữ liệu bằng:** MinIO S3 API qua Boto3 hoặc mount shared volume.
- **Tần suất cập nhật:** Hàng giờ (Hourly).
- **Xác nhận:** Data Pipeline: CẦN ĐIỀN | AI: CẦN ĐIỀN | MLOps: CẦN ĐIỀN

---

## PHẦN 2. AI → MLOPS

### 2. Model Interface Contract

*Nhóm phụ trách: AI | Nhóm đánh giá/xác nhận: MLOps*

#### 2.1. Thông tin mô hình

| Hạng mục | Phương án thống nhất |
| :--- | :--- |
| **Bài toán** | PM2.5 Forecasting (Dự báo nồng độ bụi mịn PM2.5) |
| **Thuật toán mô hình** | - Thử nghiệm chính: LightGBM Regressor (`LGBMRegressor`)<br>- Baseline so sánh: Persistence model (sử dụng giá trị PM2.5 hiện tại làm dự báo cho tương lai)<br>- Thành phần giả lập (mock) hiện tại của MLOps: `RandomForestClassifier` huấn luyện trên tập dữ liệu Breast Cancer |
| **Framework** | LightGBM / Scikit-Learn |
| **Ngôn ngữ / Phiên bản** | Python 3.11 |
| **Features đầu vào** | Các thử nghiệm trong notebook baseline hiện dùng: `pm25_ug_m3`, `hour`, `dayofweek`, các lag PM2.5 (`pm25_lag1`, `pm25_lag2`, `pm25_lag3`, `pm25_lag6`, `pm25_lag12`, `pm25_lag24`), và các lag thời tiết (`temperature_c_lag1`, `relative_humidity_pct_lag1`, `wind_speed_m_s_lag1`, `wind_direction_deg_lag1`). Hợp đồng feature chính thức để bàn giao: CẦN ĐIỀN. |
| **Mục tiêu dự báo (Target)** | `predicted_pm25_ug_m3` (Nồng độ PM2.5 tại mốc thời gian dự báo) |
| **Tầm dự báo (Horizon)** | 1 giờ (dự báo trước 1 giờ: `target_time = event_time + 1h`) |
| **Kiểu dữ liệu đầu ra** | FLOAT (Số thực không âm: `np.maximum(prediction, 0)`) |
| **Chỉ số đánh giá** | MAE (Mean Absolute Error) và RMSE (Root Mean Squared Error) |
| **Tần suất huấn luyện** | CẦN ĐIỀN |

#### 2.2. Giao diện huấn luyện (Training Interface)

- **Tên hàm:** `train_model(train_data, config=None)` *(được đề xuất trong `docs/CONTRACTS.md`; chưa triển khai thành module code riêng)*

**Đầu vào (Input):**

| Tham số | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `train_data` | DataFrame | Tập dữ liệu huấn luyện chứa features và target |
| `config` | DICT / CẦN ĐIỀN | Cấu hình huấn luyện mô hình (siêu tham số tùy chọn) |

**Đầu ra (Output):**

| Trường dữ liệu | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `model` | OBJECT | Đối tượng mô hình đã huấn luyện |
| `metrics` | DICT | Kết quả đánh giá trên tập test (MAE, RMSE, v.v.) |
| `params` | DICT / CẦN ĐIỀN | Các siêu tham số đã sử dụng |

**Quy ước huấn luyện:**
- **Chia tập train/test:** Time-based split (chia theo thứ tự thời gian; xác định điểm cutoff giữa dữ liệu train và test; không chia ngẫu nhiên).
- **Trộn dữ liệu (Shuffle):** Vô hiệu hóa (`shuffle=False`) để ngăn rò rỉ dữ liệu chuỗi thời gian.
- **Random seed:** `42`
- **Lưu trữ mô hình:** Định dạng MLflow Model (`mlflow.sklearn.log_model` hoặc flavor tương ứng).

#### 2.3. Giao diện dự đoán (Prediction Interface)

- **Tên hàm:** `predict(model, input_data)` *(được đề xuất trong `docs/CONTRACTS.md`; chưa triển khai thành module riêng)*

| Hạng mục | Phương án thống nhất |
| :--- | :--- |
| **Input schema** | CẦN ĐIỀN *(Code serving hiện tại hỗ trợ danh sách features linh hoạt, schema PM2.5 chính thức đang chờ chốt).* |
| **Định dạng Input** | JSON / DataFrame |
| **Output schema** | `station_id` (STRING), `target_time` (TIMESTAMP), `predicted_pm25_ug_m3` (FLOAT), `model_version` (STRING) *(theo đề xuất trong `docs/CONTRACTS.md`)* |
| **Định dạng Output** | JSON |
| **Dự đoán theo lô (Batch)** | Có hỗ trợ |
| **Xử lý đặc trưng thiếu** | CẦN ĐIỀN |

**Ví dụ Input:**
```json
{"pm25": 35.2, "pm25_lag_1h": 32.1}
```

**Ví dụ Output:**
```json
{"prediction": 36.5}
```

#### 2.4. Model Artifact

| Hạng mục | Phương án thống nhất |
| :--- | :--- |
| **Định dạng mô hình** | MLflow Model format |
| **Lưu trữ Artifact** | MinIO S3 (`s3://mlflow/` hoặc `s3://models/`) |
| **Tuần tự hóa mô hình** | MLflow flavors (cloudpickle / joblib qua `mlflow.sklearn` hoặc `mlflow.lightgbm`) |
| **Chữ ký mô hình (Signature)** | Bắt buộc (suy diễn bằng `mlflow.models.infer_signature`) |
| **Thư viện phụ thuộc** | File `requirements.txt` / `conda.yaml` tự động sinh bởi MLflow, và lockfile dự án `requirements-py311.lock` |
| **Metadata bắt buộc** | `model_name`, `model_version`, `training_timestamp`, `dataset_version`, `model_strategy`, `feature_schema_version`, `mae`, `rmse`, `r2` |

#### 2.5. Đánh giá và Nghiệm thu mô hình

| Tiêu chí | Yêu cầu |
| :--- | :--- |
| **Chỉ số đánh giá** | MAE (µg/m³), RMSE (µg/m³) |
| **Mô hình Baseline** | Persistence model |
| **Điều kiện nghiệm thu** | Phải vượt qua baseline model (MAE và RMSE thấp hơn Persistence model) |
| **So sánh với Champion** | CẦN ĐIỀN *(Chưa xác định ngưỡng cải thiện cụ thể).* |
| **Người phê duyệt mô hình** | CẦN ĐIỀN |

**AI bàn giao cho MLOps:**
- [x] Model artifact
- [x] Model signature
- [x] Training parameters
- [x] Evaluation metrics
- [x] Dataset version / reference
- [x] Source code / Git commit
- [ ] Hướng dẫn tải và dự đoán *(CẦN ĐIỀN)*
- [ ] Ví dụ input/output để test *(CẦN ĐIỀN)*

*Xác nhận: AI: CẦN ĐIỀN | MLOps: CẦN ĐIỀN*

---

## PHẦN 3. MLOPS

### 3. Model Registry Contract

*Nhóm phụ trách: MLOps | Nhóm đánh giá/xác nhận: AI*

#### 3.1. Cấu hình MLflow

| Hạng mục | Phương án thống nhất |
| :--- | :--- |
| **MLflow Tracking URI** | `http://mlflow:5000` (nội bộ Docker) hoặc `http://localhost:5000` (host) |
| **Backend Store** | PostgreSQL (`postgresql+psycopg2://mlflow:mlflow123@postgres:5432/mlflow`) |
| **Artifact Store** | MinIO (`s3://mlflow/` với endpoint `http://minio:9000`) |
| **Tên Experiment** | `pm25_forecasting` *(đề xuất trong `docs/CONTRACTS.md`; mock code cũ từng dùng `adaptive-model-updating`)* |
| **Tên Registered Model** | `pm25-forecasting` *(đề xuất trong `docs/CONTRACTS.md`; mock code cũ từng dùng `BreastCancerRandomForest`)* |
| **Quản lý phiên bản (Versioning)** | Số phiên bản tự tăng trong MLflow (1, 2, 3, ...) |
| **Định dạng Artifact** | MLflow Model |
| **Model Signature** | Bắt buộc |

#### 3.2. Quy tắc đặt tên (Naming Conventions)

| Đối tượng | Quy ước đặt tên | Ví dụ |
| :--- | :--- | :--- |
| **Experiment** | snake_case | `pm25_forecasting` |
| **Registered Model** | kebab-case | `pm25-forecasting` |
| **Model version** | Số nguyên tự tăng | `1`, `2` |
| **Run name** | `<model_type>-<purpose>` | `lightgbm-baseline`, `random-forest-baseline` |
| **Artifact** | Tiêu chuẩn MLflow Model | `model/model.pkl`, `MLmodel` |

#### 3.3. Vòng đời mô hình (Model Lifecycle)

| Trạng thái / Alias | Ý nghĩa | Điều kiện chuyển trạng thái |
| :--- | :--- | :--- |
| **Candidate** | Mô hình mới được log và đăng ký, đang chờ đánh giá | Đã đăng ký thành công vào MLflow Registry |
| **Champion** *(alias `@champion`)* | Mô hình đang trực tiếp phục vụ lưu lượng production | Đạt tiêu chuẩn đánh giá và vượt qua mô hình hiện tại |
| **Archived** | Mô hình cũ không còn phục vụ traffic | Khi một Candidate được gán alias `champion`, mô hình cũ mất alias `champion` và được lưu trong lịch sử MLflow dưới dạng lưu trữ |

**Quy trình phê duyệt mô hình mới:**
1. AI hoàn thành huấn luyện mô hình.
2. Mô hình được log lên MLflow kèm tham số, chỉ số, signature và artifact.
3. MLOps kiểm tra tính hợp lệ của artifact và signature.
4. AI và MLOps đánh giá kết quả so với baseline và champion hiện tại.
5. Mô hình được chấp thuận sẽ được gán alias `champion`.
6. Serving tải và đưa mô hình mới vào sử dụng.

**Người có thẩm quyền phê duyệt:** Nguyệt, Khiêm, Nhi

---

## PHẦN 4. MLOPS → AI / DATA PIPELINE

### 4. Model Serving Contract

*Nhóm phụ trách: MLOps | Nhóm sử dụng: AI, Data Pipeline*

#### 4.1. Cấu hình API

| Hạng mục | Thông tin thống nhất |
| :--- | :--- |
| **Tên dịch vụ** | Prediction API *(Tiêu đề Swagger trong `serving/main.py`: "Prediction API")* |
| **API Framework** | FastAPI |
| **Base URL** | `http://localhost:8000` (host) hoặc `http://api:8000` (nội bộ Docker) |
| **Cổng (Port)** | 8000 |
| **Các endpoint API** | `/predict` (dự đoán), `/health` (kiểm tra sức khỏe), `/docs` (Swagger UI), `/reload` (nạp nóng model) |
| **Phương thức HTTP** | POST (`/predict`, `/reload`), GET (`/health`, `/`) |
| **Content-Type** | `application/json` |
| **Xác thực (Auth)** | Không (giai đoạn mạng nội bộ Docker) |
| **Thời gian chờ (Timeout)** | 30 giây |

#### 4.2. API Request

*(Code `serving/main.py` hiện đã hỗ trợ nhận feature dạng danh sách số thực hoặc từ điển đối tượng phục vụ mô hình PM2.5. Định dạng chuẩn hóa đề xuất như sau:)*

| Trường dữ liệu | Kiểu dữ liệu | Bắt buộc | Mô tả |
| :--- | :--- | :--- | :--- |
| `model_name` | STRING | CẦN ĐIỀN *(API hiện đọc `MODEL_NAME` từ biến môi trường)* | Tên mô hình dùng để dự đoán |
| `features` | OBJECT / ARRAY | Có | Dữ liệu đầu vào phục vụ dự đoán |
| `timestamp` | STRING | CẦN ĐIỀN *(Schema serving hiện tại chưa có trường timestamp)* | Mốc thời gian gửi yêu cầu |

#### 4.3. API Response

*(Code `serving/main.py` hiện trả về `model_name: str`, `model_alias: str`, và `predictions: list[float]`. Các trường đề xuất mở rộng:)*

| Trường dữ liệu | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `model_name` | STRING | Tên mô hình phục vụ dự đoán |
| `model_version` | STRING | CẦN ĐIỀN *(Hiện chưa có trong response)* |
| `prediction` / `predictions` | FLOAT / LIST[FLOAT] | Danh sách kết quả nồng độ PM2.5 dự báo dạng số thực (µg/m³) |
| `request_id` | STRING | CẦN ĐIỀN *(Hiện chưa có trong response)* |
| `status` | STRING | CẦN ĐIỀN *(Hiện trường status chỉ có trong endpoint `/health`)* |

#### 4.4. Xử lý lỗi (Error Handling)

| Mã HTTP | Tình huống | Cách xử lý |
| :--- | :--- | :--- |
| **400** | Dữ liệu đầu vào không hợp lệ | CẦN ĐIỀN *(Hiện chưa có custom handler 400 trong `serving/main.py`)* |
| **404** | Không tìm thấy mô hình | CẦN ĐIỀN |
| **422** | Sai schema / Unprocessable Entity | Trả về chi tiết lỗi validate schema từ FastAPI/Pydantic |
| **500** | Lỗi nội bộ hệ thống | Bắt ngoại lệ, ghi log lỗi hệ thống và trả về `{"detail": str(exc)}` |
| **503** | Mô hình chưa sẵn sàng | Trả về `{"detail": "Model is not loaded"}` khi mô hình chưa khởi tạo thành công |

#### 4.5. Nạp mô hình (Model Loading)

| Hạng mục | Phương án |
| :--- | :--- |
| **Nạp mô hình từ** | MLflow Model Registry qua URI: `models:/{MODEL_NAME}@{MODEL_ALIAS}` hoặc `MODEL_URI` |
| **Lựa chọn phiên bản** | Qua Model Alias (mặc định: `champion`) hoặc biến môi trường `MODEL_ALIAS` |
| **Model Alias** | `champion` |
| **Nạp lại khi có phiên bản mới** | Gọi `POST /reload` trên service hoặc khởi động lại container API |
| **Rollback mô hình** | MLOps chuyển alias `champion` trên giao diện MLflow UI sang phiên bản cũ mong muốn và gọi `POST /reload` |

*Xác nhận: AI: CẦN ĐIỀN | MLOps: CẦN ĐIỀN*

---

## PHẦN 5. ALL TEAMS

### 5. Monitoring & Observability Contract

*Chịu trách nhiệm hạ tầng: MLOps | Cung cấp tín hiệu: AI, Data Pipeline*

#### 5.1. Các chỉ số giám sát (Monitoring Metrics)

| Tên Metric | Mô tả | Đơn vị | Nhóm phụ trách | Trạng thái triển khai trong mã nguồn |
| :--- | :--- | :--- | :--- | :--- |
| `model_prediction_error` *(code demo: `prediction_error`)* | Sai số tuyệt đối giữa dự đoán và nhãn thực tế: `abs(prediction - actual)` | µg/m³ (Histogram) | AI / MLOps | **ĐÃ TRIỂN KHAI** (`infra/observability/metrics_demo.py`) |
| `model_mae` | Sai số tuyệt đối trung bình trong cửa sổ quan sát | µg/m³ (Gauge) | AI | CẦN ĐIỀN *(Grafana tính qua PromQL: `sum(rate(prediction_error_sum[1m])) / sum(rate(prediction_error_count[1m]))`; chưa export trực tiếp).* |
| `model_rmse` | Căn bậc hai sai số bình phương trung bình trong cửa sổ | µg/m³ (Gauge) | AI | CẦN ĐIỀN *(Chưa triển khai trong code).* |
| `drift_detected_total` *(docs: `drift_alerts_total`)* | Tổng số cảnh báo trôi dạt dữ liệu hoặc khái niệm được phát hiện | Sự kiện (Counter) | AI | CẦN ĐIỀN *(Đề xuất trong tài liệu; chưa có code).* |
| `model_retrain_total` *(docs: `retrain_events_total`)* | Tổng số sự kiện cập nhật / huấn luyện lại mô hình | Sự kiện (Counter) | MLOps | Counter đếm các sự kiện thăng cấp mô hình thành công lên `@champion`. |
| `model_retrain_duration_seconds` | Thời gian thực thi một lần huấn luyện lại | Giây (Histogram) | MLOps | Thời gian một lần huấn luyện lại đo bằng `time.process_time()` và gửi qua OTel Collector. |
| `model_prediction_duration_seconds` | Thời gian xử lý một lượt dự đoán của mô hình | Giây (Histogram) | MLOps | Độ trễ xử lý cho một lượt gọi `/predict` tính bằng giây (Histogram). |
| `model_predictions_total` | Tổng số lượt dự đoán đã thực hiện | Lượt yêu cầu (Counter) | MLOps | Tổng số yêu cầu dự đoán đã phục vụ thành công (Counter). |
| `process_cpu_seconds_total` | Tổng thời gian CPU tiêu thụ bởi tiến trình | Giây (Counter) | MLOps | Thu thập qua Prometheus process collector / cgroup metrics của container. |
| `process_resident_memory_bytes` | Dung lượng bộ nhớ RAM tiến trình đang chiếm dụng | Bytes (Gauge) | MLOps | Thu thập qua Prometheus process collector / cgroup metrics của container. |
| `data_missing_rate` | Tỷ lệ dữ liệu bị thiếu trong batch hoặc khoảng thời gian | % (Gauge) | Data Pipeline | CẦN ĐIỀN *(Chưa triển khai trong code).* |
| `data_late_records_total` | Tổng số bản ghi dữ liệu đến muộn hơn thời gian dự kiến | Bản ghi (Counter) | Data Pipeline | CẦN ĐIỀN *(Chưa triển khai trong code).* |

#### 5.2. Metric Labels

| Nhãn (Label) | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `model_name` | STRING | Mã định danh mô hình (đã dùng trong `metrics_demo.py`) |
| `model_version` | STRING | Phiên bản mô hình (CẦN ĐIỀN trong ứng dụng thực tế) |
| `environment` | STRING | Môi trường triển khai (`demo`, `staging`, `production`) |
| `station_id` | STRING | Mã định danh trạm quan trắc (CẦN ĐIỀN trong ứng dụng thực tế) |

- **Chu kỳ thu thập Metric:** Prometheus scrape interval: `5s` (`infra/observability/prometheus.yml`); export interval: `2000ms` (`metrics_demo.py`).
- **Hạ tầng giám sát:** Python OpenTelemetry SDK → OpenTelemetry Collector (`otel-collector:4317` gRPC / `8889` Prometheus exporter) → Prometheus (`prometheus:9090`) → Grafana (`grafana:3000`).
- **Vị trí Dashboard:** Grafana provisioned dashboard `prediction-error.json` tại `http://localhost:3000` (UID: `prediction-error-provisioned`).

#### 5.3. Quy tắc cảnh báo (Alert Rules)

| Tên cảnh báo | Điều kiện kích hoạt | Nhóm phụ trách | Hành động |
| :--- | :--- | :--- | :--- |
| **High prediction error (Sai số dự đoán cao)** | CẦN ĐIỀN | AI | CẦN ĐIỀN |
| **Drift detected (Phát hiện trôi dạt)** | CẦN ĐIỀN | AI | CẦN ĐIỀN |
| **Retraining failed (Huấn luyện lại thất bại)** | CẦN ĐIỀN | MLOps | CẦN ĐIỀN |
| **High CPU usage (Sử dụng CPU cao)** | CẦN ĐIỀN | MLOps | CẦN ĐIỀN |
| **High memory usage (Sử dụng RAM cao)** | CẦN ĐIỀN | MLOps | CẦN ĐIỀN |
| **Data quality issue (Sự cố chất lượng dữ liệu)** | CẦN ĐIỀN | Data Pipeline | CẦN ĐIỀN |

---

## PHẦN 6. AI ↔ MLOPS

### 6. Drift Detection & Retraining Contract

#### 6.1. Sự kiện Drift (Drift Event)

- **Nhóm phát hiện Drift:** AI
- **Nhóm tiếp nhận sự kiện:** MLOps

*(Hiện tại trong mã nguồn, thư mục `drift/` mới chỉ có `.gitkeep`. Các thử nghiệm phát hiện drift nằm trong notebook `notebooks/Elec2_Label_Based_Drift_Detection.ipynb` sử dụng River ADWIN/DDM trên tập dữ liệu Elec2. Schema sự kiện chưa được chuẩn hóa để truyền qua message queue).*

| Trường dữ liệu | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `event_id` | STRING | CẦN ĐIỀN |
| `event_type` | STRING | CẦN ĐIỀN |
| `model_name` | STRING | CẦN ĐIỀN |
| `model_version` | STRING | CẦN ĐIỀN |
| `detector` | STRING | CẦN ĐIỀN *(Thử nghiệm dùng `ADWIN` và `DDM`)* |
| `drift_type` | STRING | CẦN ĐIỀN *(Thử nghiệm tập trung vào drift nhãn / hiệu năng)* |
| `timestamp` | TIMESTAMP | CẦN ĐIỀN |
| `trigger_retraining` | BOOLEAN | CẦN ĐIỀN |

#### 6.2. Chính sách huấn luyện lại (Retraining Policy)

| Hạng mục | Phương án thống nhất |
| :--- | :--- |
| **Kích hoạt Retraining** | CẦN ĐIỀN |
| **Ngưỡng Drift (Drift threshold)** | CẦN ĐIỀN *(Thử nghiệm trong notebook Elec2 dùng `ADWIN(delta=0.002)` và `DDM(warm_start=30)`)* |
| **Retraining định kỳ** | CẦN ĐIỀN |
| **Retraining theo tín hiệu Drift** | CẦN ĐIỀN |
| **Độ trễ nhãn (Label delay)** | CẦN ĐIỀN *(Ghi nhận TBD trong tài liệu)* |
| **Số lượng mẫu tối thiểu** | CẦN ĐIỀN *(Notebook Elec2 dùng cửa sổ 2.000 mẫu; PM2.5 chưa có quy định)* |
| **Cửa sổ tập dữ liệu retraining** | CẦN ĐIỀN |
| **Tần suất retraining** | CẦN ĐIỀN |
| **Xử lý sự cố retraining** | Giữ nguyên model @champion hiện tại, ghi log stacktrace và gửi thông báo cảnh báo về Slack/Email. |

#### 6.3. Thay thế mô hình (Model Replacement)

| Hạng mục | Yêu cầu |
| :--- | :--- |
| **Người đánh giá mô hình mới** | CẦN ĐIỀN |
| **So sánh với mô hình hiện tại** | CẦN ĐIỀN |
| **Tiêu chí nghiệm thu** | CẦN ĐIỀN |
| **Quyền chuyển Champion** | MLOps Lead (Nguyệt) và AI Lead (Thiện) |
| **Điều kiện Rollback** | Khi sai số dự đoán tăng vọt bất thường hoặc độ trễ API vượt 1s; MLOps chuyển alias @champion về phiên bản ổn định trước đó qua MLflow UI. |
| **Lưu trữ lịch sử retraining** | CẦN ĐIỀN |

#### 6.4. Quy trình Retraining (Workflow)

1. AI phát hiện drift.
2. AI gửi sự kiện drift (drift event).
3. MLOps ghi nhận sự kiện.
4. Policy quyết định có cần retraining hay không.
5. AI thực hiện huấn luyện mô hình.
6. MLOps log và đăng ký mô hình vào Registry.
7. AI và MLOps cùng đánh giá mô hình mới.
8. MLOps triển khai mô hình mới hoặc rollback.

*(Trạng thái hiện tại: CẦN ĐIỀN – Quy trình này chưa được triển khai tự động).*

---

## PHẦN 7. INTEGRATION & WORKING AGREEMENT

### 7. Quy định làm việc liên nhóm

#### 7.1. Phát triển song song (Parallel Development)

| Nhóm | Thành phần giả lập (Mock) sử dụng | Thành phần thật cần bàn giao |
| :--- | :--- | :--- |
| **Data Pipeline** | Các file CSV mẫu (`data/sample/beijing_multisite_features.csv`, `openaq_pm25_hanoi_hcmc.csv`, `openmeteo_weather_hanoi_hcmc.csv`) | Feature Table chuẩn dạng Parquet trên MinIO (`s3://raw/`, `s3://features/`) và pipeline cập nhật dữ liệu |
| **AI** | Sample features / dữ liệu dummy (`data/sample/*.csv`) | Script huấn luyện mô hình (`models/train.py`) và mã nguồn phát hiện drift |
| **MLOps** | Dummy sklearn model (`models/dummy_model_mlflow.py`) | Model Registry (MLflow), Serving API (`serving/main.py`), và Giám sát (OTel, Prometheus, Grafana) |

#### 7.2. Quy ước Repository (Repository Conventions)

*(Dựa trên `CONTRIBUTING.md` và `docs/CONTRACTS.md`)*

| Hạng mục | Quy định |
| :--- | :--- |
| **Đặt tên nhánh (Branch naming)** | Định dạng: `<type>/<short-description>`. Các types: `feat`, `fix`, `docs`, `test`, `ci`, `chore`, `refactor` (chữ thường, ngắn gọn). Ví dụ: `feat/kafka-ingestion` |
| **Định dạng Commit** | Conventional Commits: `<type>(<scope>): <description>`. Ví dụ: `feat(ingest): add Kafka producer`, `fix(models): correct evaluation` |
| **Template Pull Request** | Sử dụng template cấu hình sẵn tại `.github/PULL_REQUEST_TEMPLATE.md`. |
| **Số lượng Reviewer tối thiểu** | Phải có ít nhất một thành viên khác phê duyệt Pull Request trước khi merge. |
| **Định dạng mã nguồn (Formatting)** | Sử dụng Ruff (`ruff check .`, `ruff format .`) và pre-commit hooks (`.pre-commit-config.yaml`). |
| **Quy định kiểm thử (Testing)** | Sử dụng Pytest (`pytest -v`). Các file test phải nằm trong `tests/` và theo quy ước `test_*.py`. |
| **Quy trình thay đổi Contract** | Tuân thủ 7 bước: (1) Mô tả thay đổi → (2) Xác định nhóm bị ảnh hưởng → (3) Thảo luận → (4) Ghi nhận vào tài liệu → (5) Cập nhật mock/code → (6) Tạo PR → (7) Thu thập phê duyệt từ các nhóm bị ảnh hưởng. |

#### 7.3. Kiểm thử tích hợp (Integration Testing)

| Trường hợp kiểm thử | Kết quả kỳ vọng | Nhóm phụ trách | Trạng thái hiện tại |
| :--- | :--- | :--- | :--- |
| **Data Pipeline cung cấp features** | Đọc được dữ liệu với đúng schema | Data Pipeline | **ĐÃ CÓ TEST** (`tests/test_feature_contract.py`, `tests/test_vietnam_data.py`, `tests/test_vietnam_pipeline.py`) |
| **AI huấn luyện mô hình** | Mô hình được huấn luyện thành công | AI | **ĐÃ CÓ THỬ NGHIỆM** (`models/pm25-baseline.ipynb` và `models/dummy_model_mlflow.py`) |
| **MLOps đăng ký mô hình** | Phiên bản mô hình có sẵn trong MLflow | MLOps | **ĐÃ CÓ TEST** (`models/register_model.py`) |
| **API dự đoán (Prediction)** | Trả về kết quả dự đoán đúng schema | MLOps | **ĐÃ CÓ CODE & TEST** (`serving/main.py`, `tests/test_serving.py`) |
| **Sự kiện Drift** | MLOps nhận được sự kiện | AI / MLOps | CẦN ĐIỀN *(Chưa có code tích hợp)* |
| **Retraining** | Mô hình mới được log lên hệ thống | AI / MLOps | CẦN ĐIỀN *(Chưa có pipeline tự động)* |
| **Giám sát (Monitoring)** | Các chỉ số hiển thị trên Grafana | MLOps | **ĐÃ KIỂM THỬ** (Stack OTel/Prometheus/Grafana + Dashboard `prediction-error.json` + `metrics_demo.py`) |

---

## PHẦN 8. DECISION LOG

### 8. Các quyết định đã thống nhất

| STT | Nội dung | Quyết định cuối cùng | Các nhóm xác nhận | Ngày |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Feature Table** | Thống nhất 7 cột chuẩn: `station_id`, `event_time`, `pm25_ug_m3`, `temperature_c`, `relative_humidity_pct`, `wind_speed_m_s`, `wind_direction_deg`. Sử dụng timestamp UTC ISO 8601 và lưu trữ Parquet trên MinIO S3 (`s3://raw/`, `s3://features/`). Không tự ý impute missing value ở adapter layer. | Data Pipeline, AI, MLOps | CẦN ĐIỀN |
| 2 | **Model Interface** | Dự báo PM2.5 trước 1 giờ. LightGBM là mô hình thử nghiệm chính; Persistence là baseline. Chỉ số đánh giá là MAE và RMSE. Lưu trữ mô hình theo chuẩn MLflow Model. | AI, MLOps | CẦN ĐIỀN |
| 3 | **Model Registry** | Sử dụng MLflow với PostgreSQL làm backend store và MinIO S3 làm artifact store (`s3://mlflow/`). Quản lý phiên bản tự tăng và dùng alias `@champion` cho môi trường production. | MLOps, AI | CẦN ĐIỀN |
| 4 | **Serving API** | Xây dựng API bằng FastAPI và đóng gói container Docker. Nạp mô hình từ MLflow Model Registry qua alias `@champion`. Hỗ trợ các endpoint `/predict`, `/health`, `/docs`, `/reload`. | MLOps, AI | CẦN ĐIỀN |
| 5 | **Monitoring Metrics** | Thiết lập Python OpenTelemetry SDK → OTel Collector → Prometheus → Grafana. Metric kiểm thử đầu tiên là histogram `prediction_error`. Dashboard Grafana tính MAE rolling 1 phút bằng PromQL. | MLOps, AI, Data Pipeline | CẦN ĐIỀN |
| 6 | **Sự kiện Drift** | CẦN ĐIỀN *(Mới thử nghiệm ADWIN và DDM trong notebook Elec2; schema sự kiện chưa được thống nhất).* | AI, MLOps | CẦN ĐIỀN |
| 7 | **Chính sách Retraining** | CẦN ĐIỀN *(Chính sách retraining và ngưỡng kích hoạt chưa được thống nhất).* | AI, MLOps | CẦN ĐIỀN |

### Các vấn đề chưa thống nhất (Outstanding Issues)

| STT | Vấn đề | Nhóm phụ trách giải quyết | Hạn chót (Deadline) |
| :--- | :--- | :--- | :--- |
| 1 | Thống nhất các đặc trưng mở rộng của Feature Table (các cột `pm25_lag_*`, `rainfall`, và các trường trích xuất `hour` / `weekday`), cũng như chính sách xử lý missing-value và outlier cho mô hình. | Data Pipeline & AI | CẦN ĐIỀN |
| 2 | Chốt schema Request/Response chính thức cho PM2.5 Serving API, thay thế hoàn toàn schema dummy Breast Cancer cũ. | MLOps & AI | CẦN ĐIỀN |
| 3 | Chốt schema Drift Event truyền qua Message Queue (Kafka/Redpanda) và điều kiện kích hoạt retraining tự động. | AI & MLOps | CẦN ĐIỀN |

---

### Phê duyệt cuối cùng (Final Approval)

| Nhóm | Đại diện | Phê duyệt | Ngày phê duyệt |
| :--- | :--- | :---: | :--- |
| **Data Pipeline** | CẦN ĐIỀN | [ ] | CẦN ĐIỀN |
| **AI** | CẦN ĐIỀN | [ ] | CẦN ĐIỀN |
| **MLOps** | CẦN ĐIỀN | [ ] | CẦN ĐIỀN |

**Trạng thái cuối cùng:** ☑ Dự thảo (Draft)  ☐ Đang duyệt (Under Review)  ☐ Đã phê duyệt (Approved)  ☐ Cần chỉnh sửa (Needs Revision)  
**GitHub PR:** CẦN ĐIỀN  
**Ngày hoàn tất hợp đồng:** CẦN ĐIỀN  

### Tiêu chí hoàn thành (Definition of Done)

- [ ] Mỗi nhóm đã hoàn thành đầy đủ các phần được phân công.
- [ ] Schema Feature Table đã được Data Pipeline và AI phê duyệt.
- [ ] Giao diện mô hình (Model Interface) đã được AI và MLOps phê duyệt.
- [ ] Cả ba nhóm đã thống nhất các chỉ số giám sát (monitoring metrics).
- [ ] Giao diện Serving và Retraining đã được thống nhất.
- [ ] Các thành phần giả lập (mock components) có thể chạy độc lập.
- [ ] File hợp đồng được hợp nhất (merge) vào repository.
- [ ] Có ít nhất một thành viên từ mỗi nhóm phê duyệt hợp đồng.
