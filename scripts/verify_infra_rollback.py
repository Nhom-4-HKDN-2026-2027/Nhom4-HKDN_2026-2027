import math
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

print("=" * 85)
print(f"{'KIỂM CHỨNG NGƯỠNG ROLLBACK TỨC THÌ (INFRA SLA & OUTPUT SANITY)':^85}")
print("=" * 85)

# 1. Nạp dữ liệu thực tế để tạo payload giống production
current_dir = Path(__file__).resolve()
repo_root = next((p for p in current_dir.parents if (p / "data").exists()), Path.cwd())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

data_path = repo_root / "data" / "sample" / "beijing_multisite_features.csv"
assert data_path.exists(), f"Không tìm thấy file dữ liệu: {data_path}"

df = pd.read_csv(data_path)
numeric_cols = [
    c
    for c in [
        "pm25_ug_m3",
        "pm25_lag_1h",
        "pm25_lag_3h",
        "pm25_lag_24h",
        "pm25_rolling_mean_3h",
        "pm25_rolling_mean_24h",
        "temperature_c",
        "relative_humidity_pct",
        "wind_speed_m_s",
        "wind_direction_deg",
        "precipitation_mm",
        "hour",
        "weekday",
    ]
    if c in df.columns
]
sample_df = df[numeric_cols].dropna().head(200)
payloads = [{"features": [row.to_dict()]} for _, row in sample_df.iterrows()]

pm25_series = df["pm25_ug_m3"].dropna()
print(f"[*] Dữ liệu thực tế ({data_path.name}):")
print(
    f"    PM2.5 min={pm25_series.min():.2f}, P95={pm25_series.quantile(0.95):.2f}, "
    f"P99={pm25_series.quantile(0.99):.2f}, max={pm25_series.max():.2f} µg/m³\n"
)


class BaselineSimModel:
    """Mô phỏng model nhẹ tương đương Persistence/LightGBM inference."""

    def __init__(self, delay_sec: float = 0.0, force_error: bool = False, bad_output=None):
        self.delay_sec = delay_sec
        self.force_error = force_error
        self.bad_output = bad_output

    def predict(self, input_df: pd.DataFrame):
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        if self.force_error:
            raise RuntimeError("Simulated model inference crash")
        if self.bad_output is not None:
            return [self.bad_output] * len(input_df)
        return input_df["pm25_ug_m3"].astype(float).tolist()


def evaluate_sla(
    client: TestClient,
    reqs: list[dict],
    p95_limit_ms: float = 500.0,
    err_5xx_limit_pct: float = 1.0,
    invalid_limit_pct: float = 0.1,
) -> dict:
    latencies_ms = []
    err_5xx = 0
    invalid_out = 0

    for req in reqs:
        t0 = time.perf_counter()
        resp = client.post("/predict", json=req)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        if resp.status_code >= 500:
            err_5xx += 1
            continue

        preds = resp.json().get("predictions", [])
        if any((not math.isfinite(v)) or v < 0.0 or v > 1000.0 for v in preds):
            invalid_out += 1

    n = len(reqs)
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    err_pct = (err_5xx / n) * 100.0
    inv_pct = (invalid_out / n) * 100.0

    rollback = (
        (p95 > p95_limit_ms) or (err_pct > err_5xx_limit_pct) or (inv_pct > invalid_limit_pct)
    )
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "err_5xx_pct": err_pct,
        "invalid_pct": inv_pct,
        "rollback": rollback,
    }


# 2. Chạy benchmark & kiểm chứng 4 kịch bản trên serving/main.py
scenarios = [
    ("1. Normal Baseline (200 reqs)", BaselineSimModel(), payloads),
    ("2. Latency Spike (>500ms)", BaselineSimModel(delay_sec=0.55), payloads[:20]),
    ("3. HTTP 5xx Spike (Model Crash)", BaselineSimModel(force_error=True), payloads[:20]),
    ("4. Output Sanity Breach (>1000 µg/m³)", BaselineSimModel(bad_output=1500.0), payloads[:20]),
]

print(
    f"{'Kịch bản kiểm thử':<38} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | "
    f"{'5xx (%)':<8} | {'Invalid (%)':<11} | {'Quyết định'}"
)
print("-" * 98)

for name, sim_model, req_batch in scenarios:
    with patch("mlflow.pyfunc.load_model", return_value=sim_model):
        from serving.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            res = evaluate_sla(client, req_batch)
            decision = "🚨 ROLLBACK TỨC THÌ" if res["rollback"] else "✅ AN TOÀN (Giữ)"
            print(
                f"{name:<38} | {res['p50_ms']:<9.2f} | {res['p95_ms']:<9.2f} | "
                f"{res['err_5xx_pct']:<8.2f} | {res['invalid_pct']:<11.2f} | {decision}"
            )

print("=" * 98)
