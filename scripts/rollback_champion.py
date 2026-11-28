import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# 1. Nạp cấu hình từ .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

parser = argparse.ArgumentParser(
    description="Tự động hóa quy trình Rollback cho Model Champion trong MLflow Model Registry."
)
parser.add_argument(
    "--to-version",
    type=str,
    required=True,
    help="Version cụ thể cần rollback về (Bắt buộc, ví dụ: '1')",
)
parser.add_argument(
    "--model-name",
    type=str,
    default=os.getenv("MODEL_NAME", "pm25_forecasting"),
    help="Tên mô hình trong MLflow Model Registry (mặc định: 'pm25_forecasting')",
)
parser.add_argument(
    "--alias",
    type=str,
    default=os.getenv("MODEL_ALIAS", "champion"),
    help="Alias cần di chuyển (mặc định: 'champion')",
)
parser.add_argument(
    "--serving-url",
    type=str,
    default=os.getenv("SERVING_URL", "http://localhost:8000"),
    help="URL của Serving API để gọi hot-reload (mặc định: 'http://localhost:8000')",
)
parser.add_argument(
    "--trigger-reload",
    action="store_true",
    help="Tự động gửi request POST /reload đến Serving API ngay sau khi rollback alias",
)
parser.add_argument(
    "--reason",
    type=str,
    default="Manual rollback triggered by operator",
    help="Lý do thực hiện rollback (lưu lại vào Audit Log)",
)
parser.add_argument(
    "--skip-confirm",
    action="store_true",
    help="Bỏ qua bước hỏi xác nhận (dùng cho CI/CD hoặc Automation)",
)

args = parser.parse_args()


def format_timestamp(ts_ms: int | float | None) -> str:
    """Chuyển đổi epoch milliseconds sang chuỗi thời gian UTC dễ đọc."""
    if not ts_ms:
        return "N/A"
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_version_metrics(client: MlflowClient, run_id: str | None) -> dict[str, Any]:
    """Trích xuất metrics chính của run nếu có."""
    if not run_id:
        return {}
    try:
        run = client.get_run(run_id)
        m = run.data.metrics
        return {
            "mae": f"{m['mae']:.2f}" if "mae" in m else "N/A",
            "rmse": f"{m['rmse']:.2f}" if "rmse" in m else "N/A",
        }
    except Exception:
        return {}


def record_audit_log(log_data: dict[str, Any]) -> None:
    """Ghi nhật ký kiểm toán (Audit Trail) phục vụ truy vết MLOps Governance."""
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    audit_file = logs_dir / "rollback_audit.jsonl"
    try:
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        print(f"[*] Đã ghi nhận Audit Log tại: {audit_file.relative_to(Path.cwd())}")
    except Exception as ex:
        print(f"[!] Cảnh báo: Không thể ghi audit log: {ex}")


def main():
    client = MlflowClient()
    model_name = args.model_name
    alias = args.alias
    to_version = str(args.to_version).strip()

    print("=" * 85)
    print(f"{'QUY TRÌNH ROLLBACK TỰ ĐỘNG CHO MODEL CHAMPION':^85}")
    print("=" * 85)
    print(f"[*] MLflow Tracking URI : {tracking_uri}")
    print(f"[*] Registered Model     : {model_name}")
    print(f"[*] Target Alias         : @{alias}")
    print(f"[*] Rollback về Version  : {to_version}")
    print(f"[*] Lý do                : {args.reason}")
    print("-" * 85)

    # BƯỚC 1: Lấy thông tin Champion hiện tại
    current_champ_ver = None
    current_champ_run = None
    try:
        champ_ver = client.get_model_version_by_alias(name=model_name, alias=alias)
        current_champ_ver = str(champ_ver.version)
        current_champ_run = champ_ver.run_id
    except Exception as ex:
        print(f"[!] Cảnh báo: Hiện tại chưa có model nào gắn alias @{alias} ({ex})")

    if current_champ_ver == to_version:
        print(
            f"[!] CẢNH BÁO: Model Version {to_version} HIỆN ĐÃ LÀ @{alias}!\n"
            f"    Không cần thực hiện rollback."
        )
        sys.exit(0)

    # BƯỚC 2: Kiểm tra Version mục tiêu có tồn tại và hợp lệ không
    try:
        target_model = client.get_model_version(name=model_name, version=to_version)
    except Exception as ex:
        print(
            f"[✗] LỖI: Không tìm thấy Version {to_version} trong Registry của '{model_name}': {ex}"
        )
        sys.exit(1)

    target_status = target_model.status
    if target_status not in ("READY", None):
        print(
            f"[✗] LỖI: Model Version {to_version} đang ở trạng thái '{target_status}', "
            f"không sẵn sàng để phục vụ (chỉ chấp nhận 'READY')."
        )
        sys.exit(1)

    # Bảng so sánh trước khi xác nhận
    cur_metrics = get_version_metrics(client, current_champ_run)
    tgt_metrics = get_version_metrics(client, target_model.run_id)

    print(
        f"{'Trường thông tin':<25} | {'Champion hiện tại':<25} | {'Version mục tiêu (Rollback về)':<25}"
    )
    print("-" * 85)
    print(f"{'Version':<25} | {str(current_champ_ver):<25} | {to_version:<25}")
    print(
        f"{'Run ID':<25} | {str(current_champ_run)[:16] + '...':<25} | {str(target_model.run_id)[:16] + '...':<25}"
    )
    print(
        f"{'MAE (µg/m³)':<25} | {cur_metrics.get('mae', 'N/A'):<25} | {tgt_metrics.get('mae', 'N/A'):<25}"
    )
    print(
        f"{'RMSE (µg/m³)':<25} | {cur_metrics.get('rmse', 'N/A'):<25} | {tgt_metrics.get('rmse', 'N/A'):<25}"
    )
    print("-" * 85)

    # BƯỚC 3: Xác nhận hành động
    if not args.skip_confirm:
        confirm = input(
            f"[?] Bạn có chắc chắn muốn chuyển alias @{alias} về Version {to_version}? (y/N): "
        )
        if confirm.lower() not in ("y", "yes"):
            print("[-] Hủy thao tác rollback.")
            sys.exit(0)

    # BƯỚC 4: Dời alias @champion trên MLflow
    try:
        client.set_registered_model_alias(name=model_name, alias=alias, version=to_version)
        print(
            f"\n[✓] THÀNH CÔNG: Đã chuyển alias @{alias} sang Version {to_version} trên MLflow Registry!"
        )
    except Exception as ex:
        print(f"[✗] LỖI khi chuyển alias trên MLflow: {ex}")
        sys.exit(1)

    # BƯỚC 5: Kích hoạt Hot-Reload trên Serving (nếu yêu cầu)
    reload_status = "SKIPPED"
    if args.trigger_reload:
        reload_url = f"{args.serving_url.rstrip('/')}/reload"
        print(f"[*] Đang gửi tín hiệu Hot-Reload đến: {reload_url}...")
        try:
            import urllib.request

            req = urllib.request.Request(
                reload_url, data=b"", headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    reloaded_ver = data.get("model_version", "unknown")
                    print(
                        f"[✓] Serving API đã Hot-Reload thành công! Active Version: {reloaded_ver}"
                    )
                    reload_status = "SUCCESS"
                else:
                    print(f"[!] Cảnh báo: Serving API trả về mã HTTP {resp.status}")
                    reload_status = f"HTTP_{resp.status}"
        except Exception as ex:
            print(
                f"[!] CẢNH BÁO: Đã đổi alias trên MLflow nhưng không thể reload Serving API ({ex}).\n"
                f"    -> Hãy đảm bảo Service đang chạy và reload thủ công: POST {reload_url}"
            )
            reload_status = "FAILED"

    # BƯỚC 6: Ghi nhận Audit Trail
    audit_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "MODEL_ROLLBACK",
        "model_name": model_name,
        "alias": alias,
        "from_version": current_champ_ver,
        "to_version": to_version,
        "from_run_id": current_champ_run,
        "to_run_id": target_model.run_id,
        "reason": args.reason,
        "reload_triggered": args.trigger_reload,
        "reload_status": reload_status,
    }
    record_audit_log(audit_record)

    print("=" * 85)
    print(f"[✓] HOÀN TẤT ROLLBACK: Model '{model_name}' @{alias} hiện là Version {to_version}.\n")


if __name__ == "__main__":
    main()
