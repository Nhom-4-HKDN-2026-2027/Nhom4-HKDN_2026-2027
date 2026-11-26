import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# 0. Nạp biến môi trường
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

print("=" * 80)
print(f"{'BACKTESTING NGƯỠNG ROLLBACK TRÊN DỮ LIỆU QUAN TRẮC THỰC TẾ':^80}")
print("=" * 80)


# 1. ĐỌC DỮ LIỆU THẬT CỦA DỰ ÁN
def load_data() -> tuple[pd.DataFrame, bool]:
    current_dir = Path(__file__).resolve()
    repo_root = next((p for p in current_dir.parents if (p / "data").exists()), Path.cwd())

    test_pred_path = repo_root / "data" / "sample" / "test_predictions.csv"
    sample_feat_path = repo_root / "data" / "sample" / "beijing_multisite_features.csv"

    # 1.1. Thử đọc file test_predictions.csv từ local nếu có
    if test_pred_path.exists():
        print(f"[*] Đã nạp thành công tập kiểm thử của Champion từ local: {test_pred_path.name}")
        data = pd.read_csv(test_pred_path)
        return data, True

    # 1.2. Thử tải từ MinIO nếu có cấu hình
    try:
        import boto3
        from botocore.exceptions import ClientError

        endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        access_key = os.getenv("MINIO_ROOT_USER") or os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        secret_key = os.getenv("MINIO_ROOT_PASSWORD") or os.getenv(
            "AWS_SECRET_ACCESS_KEY", "minioadmin123"
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        for bucket in ["models", "features"]:
            try:
                # Quét tìm test_predictions.csv (bao gồm cả thư mục handoff của MLflow run)
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket):
                    for item in page.get("Contents", []):
                        if item["Key"].endswith("test_predictions.csv"):
                            target_key = item["Key"]
                            print(f"[*] Đang tải {target_key} từ MinIO s3://{bucket}/...")
                            obj = s3.get_object(Bucket=bucket, Key=target_key)
                            data = pd.read_csv(obj["Body"])
                            print(
                                f"[*] Đã nạp thành công {target_key} ({len(data):,} dòng) từ MinIO!"
                            )
                            return data, True
            except (ClientError, Exception):
                continue
    except Exception as e:
        print(f"[*] Kết nối MinIO không thành công: {e}")

    # 1.3. Fallback về sample features có sẵn trong repo
    assert sample_feat_path.exists(), f"Không tìm thấy file dữ liệu mẫu tại: {sample_feat_path}"
    print("[*] Không tìm thấy test_predictions.csv trên local hoặc MinIO.")
    print(f"[*] Sử dụng file dữ liệu mẫu có sẵn trong repo: {sample_feat_path.name}")
    data = pd.read_csv(sample_feat_path)
    return data, False


df, is_test_pred = load_data()
df["event_time"] = pd.to_datetime(df["event_time"])
df = df.sort_values(["station_id", "event_time"]).reset_index(drop=True)
print(f"[*] Tổng số mẫu quan trắc thực tế: {len(df):,} dòng (12 trạm quan trắc)")

if is_test_pred and "error_serving" in df.columns:
    # Sử dụng trực tiếp sai số thực tế của Champion (LightGBM)
    df["signed_error"] = df["error_serving"]
    df["error"] = df["absolute_error_serving"]
    df["sq_error"] = df["squared_error_serving"]
else:
    # Dự báo 1 bước thời gian bằng Persistence baseline
    grouped = df.groupby("station_id")["pm25_ug_m3"]
    df["target_actual"] = grouped.shift(-1)
    df["signed_error"] = df["target_actual"] - df["pm25_ug_m3"]
    df["error"] = np.abs(df["signed_error"])
    df["sq_error"] = df["signed_error"] ** 2

clean_df = df.dropna(subset=["error"]).copy()

# 2. CHỈ SỐ CƠ SỞ THỰC TẾ
base_mae = clean_df["error"].mean()
base_rmse = np.sqrt(clean_df["sq_error"].mean())
print(f"[*] Sai số MAE Champion cơ sở : {base_mae:.2f} µg/m³")
print(f"[*] Sai số RMSE Champion cơ sở: {base_rmse:.2f} µg/m³\n")

# 3. TÍNH CÁC CHỈ SỐ ROLLING 24H THỰC TẾ CHO TỪNG TRẠM
clean_df["rolling_mae"] = (
    clean_df.groupby("station_id")["error"]
    .rolling(window=24, min_periods=12)
    .mean()
    .reset_index(level=0, drop=True)
)
clean_df["rolling_rmse"] = np.sqrt(
    clean_df.groupby("station_id")["sq_error"]
    .rolling(window=24, min_periods=12)
    .mean()
    .reset_index(level=0, drop=True)
)
clean_df["rolling_abs_bias"] = (
    clean_df.groupby("station_id")["signed_error"]
    .rolling(window=24, min_periods=12)
    .mean()
    .abs()
    .reset_index(level=0, drop=True)
)


# 4. HÀM CHẠY BACKTESTING DÙNG CHUNG CHO CÁC LOẠI NGƯỠNG (1H VS 3H LIÊN TIẾP)
def evaluate_pct(pct: float) -> str:
    if pct > 10.0:
        return "❌ Quá nhạy"
    elif 1.0 <= pct <= 5.0:
        return "🏆 TỐI ƯU"
    elif pct < 0.5:
        return "❌ Quá lỏng"
    return "⚠️ Tạm ổn"


def run_backtest_table(
    data: pd.DataFrame,
    metric_col: str,
    thresholds: list[float],
    metric_label: str,
    base_value: float | None = None,
) -> None:
    print(f"\n>>> BACKTESTING NGƯỠNG ROLLBACK THEO: {metric_label}")
    header = (
        f"{'Ngưỡng':<10} | {'Giá trị chặn':<14} | {'1h (Giờ / %)':<16} | {'Đánh giá 1h':<14} | "
        f"{'3h liên tiếp':<16} | {'Đánh giá 3h'}"
    )
    print(header)
    print("-" * 100)

    total_rows = len(data)
    for th in thresholds:
        if base_value is not None:
            limit_val = base_value * (1 + th)
            th_label = f"+{int(th * 100)}%"
        else:
            limit_val = th
            th_label = f">{th:.1f}"

        exceed = (data[metric_col] > limit_val).astype(int)
        h1_hours = int(exceed.sum())
        h1_pct = (h1_hours / total_rows) * 100

        sustained_mask = (
            exceed.groupby(data["station_id"])
            .rolling(window=3, min_periods=3)
            .sum()
            .reset_index(level=0, drop=True)
            == 3
        )
        h3_hours = int(sustained_mask.sum())
        h3_pct = (h3_hours / total_rows) * 100

        print(
            f"{th_label:<10} | {f'{limit_val:.2f} µg/m³':<14} | {f'{h1_hours}h ({h1_pct:.2f}%)':<16} | "
            f"{evaluate_pct(h1_pct):<14} | {f'{h3_hours}h ({h3_pct:.2f}%)':<16} | {evaluate_pct(h3_pct)}"
        )
    print("=" * 100)


# 5. CHẠY KIỂM CHỨNG CHO MAE, RMSE VÀ |BIAS|
run_backtest_table(
    clean_df,
    metric_col="rolling_mae",
    thresholds=[0.10, 0.20, 0.50, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00],
    metric_label="Rolling 24h-MAE (Sai số trung bình)",
    base_value=base_mae,
)

run_backtest_table(
    clean_df,
    metric_col="rolling_rmse",
    thresholds=[0.20, 0.50, 0.80, 1.00, 1.20, 1.40, 1.50, 1.80, 2.00],
    metric_label="Rolling 24h-RMSE (Phạt sai số cực đoan / Spike)",
    base_value=base_rmse,
)

run_backtest_table(
    clean_df,
    metric_col="rolling_abs_bias",
    thresholds=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0],
    metric_label="|Rolling 24h-Bias| (Độ lệch hệ thống / Concept Drift)",
    base_value=None,
)
