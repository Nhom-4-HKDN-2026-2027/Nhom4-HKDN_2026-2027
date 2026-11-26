import argparse
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# 1. Nạp cấu hình từ .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

parser = argparse.ArgumentParser(description="Kiểm chứng thăng hạng Champion cho Model mới.")
parser.add_argument(
    "--candidate-version",
    type=str,
    default=os.getenv("CANDIDATE_VERSION"),
    help="Version của model candidate cần kiểm chứng",
)
parser.add_argument(
    "--candidate-run-id",
    type=str,
    default=os.getenv("CANDIDATE_RUN_ID"),
    help="Run ID của model candidate (nếu chưa gán version)",
)
parser.add_argument(
    "--model-name",
    type=str,
    default=os.getenv("MODEL_NAME", "pm25_forecasting"),
    help="Tên model trong MLflow Registry",
)
args = parser.parse_args()

model_name = args.model_name
print(f"[*] Đang kết nối MLflow tại: {tracking_uri}...")
print(f"[*] Kiểm chứng Model: {model_name}")


def extract_metrics(run_id: str, label: str):
    run = client.get_run(run_id)
    metrics = run.data.metrics
    required = ["mae", "rmse", "persistence_mae", "persistence_rmse"]
    missing = [m for m in required if m not in metrics]
    if missing:
        print(f"[!] FAIL ({label}): Thiếu metric bắt buộc trong MLflow Run {run_id}: {missing}")
        exit(1)
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "persistence_mae": metrics["persistence_mae"],
        "persistence_rmse": metrics["persistence_rmse"],
    }


# 2. Lấy thông tin Model Champion đang chạy
champion_alias = os.getenv("MODEL_ALIAS", "champion")
champion_metrics = None
champion_ver_str = "None"

try:
    champ_ver = client.get_model_version_by_alias(name=model_name, alias=champion_alias)
    champion_ver_str = str(champ_ver.version)
    champion_metrics = extract_metrics(champ_ver.run_id, f"Champion @{champion_alias}")
    print(f"[*] Đã tải Champion hiện tại: Version {champion_ver_str} (Run ID: {champ_ver.run_id})")
except Exception as e:
    print(
        f"[!] Cảnh báo: Chưa tìm thấy model @{champion_alias} hiện tại ({e}). Sẽ chỉ kiểm chứng Lớp 1 (Baseline)."
    )

# 3. Lấy thông tin Candidate Model
candidate_run_id = args.candidate_run_id
candidate_ver_str = args.candidate_version or "Candidate"

if args.candidate_version:
    cand_ver = client.get_model_version(name=model_name, version=args.candidate_version)
    candidate_run_id = cand_ver.run_id
elif not candidate_run_id:
    # Mặc định lấy run mới nhất thuộc experiment nếu không truyền gì
    exp = client.get_experiment_by_name(model_name)
    if exp:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=1,
            order_by=["attribute.start_time DESC"],
        )
        if runs:
            candidate_run_id = runs[0].info.run_id
            print(f"[*] Tự động chọn Run mới nhất làm Candidate: {candidate_run_id}")

if not candidate_run_id:
    print(
        "[!] Không tìm thấy Run ID của Candidate để kiểm chứng. Dùng --candidate-version hoặc --candidate-run-id."
    )
    exit(1)

cand_metrics = extract_metrics(candidate_run_id, "Candidate")

# 4. Kiểm chứng 2 lớp (Two-Tier Evaluation)
print("\n" + "=" * 78)
print(f"{'BẢNG KIỂM CHỨNG TIÊU CHÍ THĂNG HẠNG CHAMPION':^78}")
print("=" * 78)
print(f"{'Tiêu chí kiểm chứng':<32} | {'Ngưỡng yêu cầu':<18} | {'Thực tế':<12} | {'Kết luận'}")
print("-" * 78)

# LỚP 1: So với Persistence Baseline
g1_mae_pass = cand_metrics["mae"] < cand_metrics["persistence_mae"]
g1_str = "ĐẠT (PASS)" if g1_mae_pass else "KHÔNG ĐẠT"
cand_mae_str = f"{cand_metrics['mae']:.2f} < {cand_metrics['persistence_mae']:.2f}"
print(
    f"{'1. Lớp 1: MAE < Persistence':<32} | {'Thấp hơn baseline':<18} | {cand_mae_str:<12} | {g1_str}"
)

# LỚP 2: So với Champion đang chạy
all_pass = g1_mae_pass
if champion_metrics:
    delta_mae = ((champion_metrics["mae"] - cand_metrics["mae"]) / champion_metrics["mae"]) * 100
    delta_rmse = (
        (cand_metrics["rmse"] - champion_metrics["rmse"]) / champion_metrics["rmse"]
    ) * 100

    # Gate 2.1: MAE thấp hơn champion ít nhất 2%
    g2_mae_pass = delta_mae >= 2.0
    g2_str = "ĐẠT (PASS)" if g2_mae_pass else "KHÔNG ĐẠT"
    delta_mae_str = f"{delta_mae:+.2f}%"
    print(
        f"{'2. Lớp 2: ΔMAE vs Champion':<32} | {'Giảm >= 2.0%':<18} | {delta_mae_str:<12} | {g2_str}"
    )

    # Gate 2.2: RMSE không cao hơn champion quá 1%
    g2_rmse_pass = cand_metrics["rmse"] <= champion_metrics["rmse"] * 1.01
    g3_str = "ĐẠT (PASS)" if g2_rmse_pass else "KHÔNG ĐẠT"
    delta_rmse_str = f"{delta_rmse:+.2f}%"
    print(
        f"{'3. Lớp 2: RMSE vs Champion':<32} | {'Tăng <= 1.0%':<18} | {delta_rmse_str:<12} | {g3_str}"
    )

    all_pass = all_pass and g2_mae_pass and g2_rmse_pass
else:
    print(
        f"{'2. Lớp 2: So với Champion':<32} | {'Bỏ qua (Chưa có)':<18} | {'N/A':<12} | ĐẠT (Khởi tạo)"
    )

print("=" * 78)
if all_pass:
    print("KẾT LUẬN: Candidate ĐẠT TOÀN BỘ TIÊU CHUẨN để thăng hạng @champion!\n")
    exit(0)
else:
    print("KẾT LUẬN: Candidate KHÔNG ĐẠT tiêu chuẩn, giữ nguyên Champion hiện tại.\n")
    exit(1)
