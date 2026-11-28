import argparse
import os
import sys
from datetime import datetime, timezone
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

parser = argparse.ArgumentParser(
    description="Kiểm chứng thăng hạng Champion cho Model mới theo tiến trình thời gian và xét tuần tự từng lớp."
)
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
parser.add_argument(
    "--promote",
    action="store_true",
    default=os.getenv("AUTO_PROMOTE", "false").lower() in ("true", "1", "yes"),
    help="Tự động cập nhật alias @champion trong MLflow Registry nếu đạt toàn bộ tiêu chuẩn",
)
args = parser.parse_args()

model_name = args.model_name
print(f"[*] Đang kết nối MLflow tại: {tracking_uri}...")
print(f"[*] Kiểm chứng Model: {model_name}")


def format_timestamp(ts_ms: int | float | None) -> str:
    """Chuyển đổi epoch milliseconds sang chuỗi thời gian UTC dễ đọc."""
    if not ts_ms:
        return "N/A"
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def extract_metrics(run_id: str, label: str) -> dict:
    if not run_id:
        raise ValueError(f"Run ID rỗng cho {label}")
    run = client.get_run(run_id)
    metrics = run.data.metrics
    required = ["mae", "rmse", "persistence_mae", "persistence_rmse"]
    missing = [m for m in required if m not in metrics]
    if missing:
        print(f"[!] FAIL ({label}): Thiếu metric bắt buộc trong MLflow Run {run_id}: {missing}")
        sys.exit(1)
    return {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "persistence_mae": float(metrics["persistence_mae"]),
        "persistence_rmse": float(metrics["persistence_rmse"]),
        "start_time_str": format_timestamp(run.info.start_time),
    }


# 2. Lấy thông tin Model Champion đang chạy (theo mốc thời gian trước đó)
champion_alias = os.getenv("MODEL_ALIAS", "champion")
champion_metrics = None
champion_ver_str = "None"
champion_created_str = "N/A"

try:
    champ_ver = client.get_model_version_by_alias(name=model_name, alias=champion_alias)
    champion_ver_str = str(champ_ver.version)
    champion_created_str = format_timestamp(champ_ver.creation_timestamp)
    if champ_ver.run_id:
        champion_metrics = extract_metrics(champ_ver.run_id, f"Champion @{champion_alias}")
        print(
            f"[*] Đã tải Champion hiện tại: Version {champion_ver_str} "
            f"(Tạo lúc: {champion_created_str}, Run ID: {champ_ver.run_id})"
        )
    else:
        print(
            f"[!] Cảnh báo: Model @{champion_alias} Version {champion_ver_str} không có run_id liên kết. "
            f"Hệ thống sẽ chỉ kiểm chứng Lớp 1 (Baseline Gate)."
        )
except Exception as e:
    print(
        f"[!] Cảnh báo: Chưa tìm thấy model @{champion_alias} hiện tại ({e}). "
        f"Hệ thống sẽ chỉ kiểm chứng Lớp 1 (Baseline Gate) để khởi tạo mốc thời gian đầu tiên."
    )

# 3. Lấy thông tin Candidate Model (bắt buộc truyền version hoặc run-id)
candidate_run_id = args.candidate_run_id
candidate_ver_str = args.candidate_version or "Candidate"
candidate_created_str = "N/A"

if not args.candidate_version and not args.candidate_run_id:
    print(
        "\n[!] LỖI BẮT BUỘC: Bạn phải chỉ định --candidate-version <version> "
        "hoặc --candidate-run-id <run_id> để kiểm chứng!\n"
        "    Hệ thống không tự ý chọn run mới nhất nhằm tránh nhầm lẫn với các thử nghiệm nháp hoặc run lỗi.\n"
        "    Ví dụ:\n"
        "      python scripts/verify_champion.py --candidate-version 2 --promote\n"
        "      python scripts/verify_champion.py --candidate-run-id 0a7f3e49309b4f7590e01a5e61939b9c\n"
    )
    sys.exit(1)

if args.candidate_version:
    try:
        cand_ver = client.get_model_version(name=model_name, version=args.candidate_version)
        candidate_run_id = cand_ver.run_id
        candidate_created_str = format_timestamp(cand_ver.creation_timestamp)
        candidate_ver_str = str(cand_ver.version)
    except Exception as ex:
        print(f"[!] Không tìm thấy Model Version {args.candidate_version} cho '{model_name}': {ex}")
        sys.exit(1)
elif candidate_run_id:
    # Nếu chỉ truyền candidate_run_id, tìm xem run này đã được đăng ký version nào chưa
    try:
        all_vers = client.search_model_versions(f"name='{model_name}'")
        for v in all_vers:
            if v.run_id == candidate_run_id:
                candidate_ver_str = str(v.version)
                candidate_created_str = format_timestamp(v.creation_timestamp)
                break
    except Exception:
        pass

if not candidate_run_id:
    print(
        "[!] Không tìm thấy Run ID của Candidate để kiểm chứng. "
        "Dùng --candidate-version hoặc --candidate-run-id."
    )
    sys.exit(1)

cand_metrics = extract_metrics(candidate_run_id, "Candidate")
if candidate_created_str == "N/A":
    candidate_created_str = cand_metrics["start_time_str"]

# 4. Kiểm chứng tuần tự từng lớp (Sequential Two-Tier Evaluation)
print("\n" + "=" * 85)
print(f"{'BẢNG KIỂM CHỨNG TIÊU CHÍ THĂNG HẠNG CHAMPION THEO THỜI GIAN':^85}")
print("=" * 85)
print(f"[*] Champion tiền nhiệm: Version {champion_ver_str:<6} | Thời gian: {champion_created_str}")
print(
    f"[*] Candidate kế nhiệm : Version {candidate_ver_str:<6} | Thời gian: {candidate_created_str}"
)
print("-" * 85)
print(f"{'Tiêu chuẩn kiểm chứng':<35} | {'Ngưỡng yêu cầu':<20} | {'Thực tế':<12} | {'Kết luận'}")
print("-" * 85)

# =========================================================================
# LỚP 1: KIỂM ĐỊNH CƠ SỞ (BASELINE GATE)
# Model mới bắt buộc phải tốt hơn mô hình Persistence ngây thơ trên cả 2 chỉ số.
# =========================================================================
print("[*] ĐANG XÉT LỚP 1: So sánh Candidate với Persistence Baseline...")

# Tiêu chí 1.1: MAE < Persistence MAE
g1_mae_pass = cand_metrics["mae"] < cand_metrics["persistence_mae"]
cand_mae_str = f"{cand_metrics['mae']:.2f} < {cand_metrics['persistence_mae']:.2f}"
g1_mae_res = "ĐẠT (PASS)" if g1_mae_pass else "KHÔNG ĐẠT"
print(
    f"{'1.1. Lớp 1: MAE < Persistence':<35} | {'Thấp hơn baseline':<20} | {cand_mae_str:<12} | {g1_mae_res}"
)

# Tiêu chí 1.2: RMSE < Persistence RMSE
g1_rmse_pass = cand_metrics["rmse"] < cand_metrics["persistence_rmse"]
cand_rmse_str = f"{cand_metrics['rmse']:.2f} < {cand_metrics['persistence_rmse']:.2f}"
g1_rmse_res = "ĐẠT (PASS)" if g1_rmse_pass else "KHÔNG ĐẠT"
print(
    f"{'1.2. Lớp 1: RMSE < Persistence':<35} | {'Thấp hơn baseline':<20} | {cand_rmse_str:<12} | {g1_rmse_res}"
)

g1_pass = g1_mae_pass and g1_rmse_pass

# KIỂM TRA ĐIỀU KIỆN TIÊN QUYẾT LỚP 1 - SHORT-CIRCUIT NẾU THẤT BẠI
if not g1_pass:
    print("-" * 85)
    print(
        "[!] KẾT QUẢ LỚP 1 (Baseline Gate): THẤT BẠI!\n"
        "    -> Candidate không vượt qua được Persistence Baseline (mô hình ngây thơ).\n"
        "    -> DỪNG KIỂM CHỨNG NGAY: Bỏ qua Lớp 2 (không so sánh với Champion) theo nguyên tắc tuần tự."
    )
    print("=" * 85)
    print(
        f"[✗] KẾT LUẬN: Candidate ({candidate_ver_str}) BỊ TỪ CHỐI THĂNG HẠNG tại Lớp 1. "
        f"Giữ nguyên Champion hiện tại (Version {champion_ver_str}).\n"
    )
    sys.exit(1)

print(
    f"{'--> Đánh giá Lớp 1 (Baseline Gate)':<35} | {'Đạt cả MAE & RMSE':<20} | {'ĐÃ ĐẠT':<12} | CHUYỂN TIẾP XÉT LỚP 2"
)
print("-" * 85)

# =========================================================================
# LỚP 2: KIỂM ĐỊNH NÂNG CẤP SO VỚI CHAMPION (CHAMPION GATE)
# Chỉ xét khi Lớp 1 đã ĐẠT: Candidate phải có cải thiện rõ rệt so với Champion.
# =========================================================================
print("[*] ĐANG XÉT LỚP 2: So sánh Candidate với Champion hiện tại...")

all_pass = False
if champion_metrics is None:
    # Trường hợp khởi tạo ban đầu (chưa có Champion nào trước đó)
    print(
        f"{'2. Lớp 2: So sánh với Champion':<35} | {'Khởi tạo mốc ban đầu':<20} | {'N/A':<12} | ĐẠT (Khởi tạo)"
    )
    all_pass = True
else:
    # 2.1. MAE phải giảm ít nhất 2.0% so với Champion
    delta_mae = ((champion_metrics["mae"] - cand_metrics["mae"]) / champion_metrics["mae"]) * 100
    g2_mae_pass = delta_mae >= 2.0
    delta_mae_str = f"{delta_mae:+.2f}%"
    g2_mae_res = "ĐẠT (PASS)" if g2_mae_pass else "KHÔNG ĐẠT"
    print(
        f"{'2.1. Lớp 2: ΔMAE vs Champion':<35} | {'Giảm >= 2.0%':<20} | {delta_mae_str:<12} | {g2_mae_res}"
    )

    # 2.2. RMSE không được tăng quá 1.0% so với Champion
    delta_rmse = (
        (cand_metrics["rmse"] - champion_metrics["rmse"]) / champion_metrics["rmse"]
    ) * 100
    g2_rmse_pass = cand_metrics["rmse"] <= champion_metrics["rmse"] * 1.01
    delta_rmse_str = f"{delta_rmse:+.2f}%"
    g2_rmse_res = "ĐẠT (PASS)" if g2_rmse_pass else "KHÔNG ĐẠT"
    print(
        f"{'2.2. Lớp 2: ΔRMSE vs Champion':<35} | {'Tăng <= 1.0%':<20} | {delta_rmse_str:<12} | {g2_rmse_res}"
    )

    g2_pass = g2_mae_pass and g2_rmse_pass
    if not g2_pass:
        print("-" * 85)
        print(
            "[!] KẾT QUẢ LỚP 2 (Champion Gate): THẤT BẠI!\n"
            "    -> Candidate không đạt biên độ cải thiện yêu cầu (cần: MAE giảm >= 2%, RMSE tăng <= 1%)."
        )
    else:
        print(
            f"{'--> Đánh giá Lớp 2 (Champion Gate)':<35} | {'Đạt biên độ cải tiến':<20} | {'ĐÃ ĐẠT':<12} | ĐẠT TẤT CẢ TIÊU CHUẨN"
        )
        all_pass = True

print("=" * 85)

# 5. KẾT LUẬN & HÀNH ĐỘNG THĂNG HẠNG (PROMOTION)
if all_pass:
    print(
        f"[✓] KẾT LUẬN: Candidate ({candidate_ver_str}) ĐẠT TOÀN BỘ TIÊU CHUẨN ĐỂ THĂNG HẠNG @{champion_alias}!\n"
    )

    if args.promote:
        if candidate_ver_str and candidate_ver_str != "Candidate":
            try:
                client.set_registered_model_alias(
                    name=model_name,
                    alias=champion_alias,
                    version=candidate_ver_str,
                )
                print(
                    f"[★] ĐÃ THĂNG HẠNG THÀNH CÔNG: Model '{model_name}' Version {candidate_ver_str} "
                    f"đã được gán alias '@{champion_alias}' trong MLflow Model Registry."
                )
                print(
                    "[*] Gợi ý: Hãy gửi 'POST /reload' đến API Serving để kích hoạt model mới tức thì."
                )
            except Exception as ex:
                print(f"[!] Lỗi khi gán alias trên MLflow: {ex}")
                sys.exit(1)
        else:
            print(
                "[!] Lưu ý: Model Candidate chưa có Version số cụ thể trong Registry nên chưa thể gán alias tự động. "
                "Hãy đăng ký model version trước khi thăng hạng."
            )
    sys.exit(0)
else:
    print(
        f"[✗] KẾT LUẬN: Candidate ({candidate_ver_str}) KHÔNG ĐẠT tiêu chuẩn thăng hạng. "
        f"Giữ nguyên Champion hiện tại (Version {champion_ver_str}).\n"
    )
    sys.exit(1)
