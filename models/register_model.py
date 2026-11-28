import argparse
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)

parser = argparse.ArgumentParser(
    description="Đăng ký mô hình MLflow Model Registry đúng chuẩn (liên kết đầy đủ với Run ID)."
)
parser.add_argument(
    "--run-id",
    type=str,
    default=os.getenv("RUN_ID"),
    help="Run ID trong MLflow Experiment (ưu tiên dùng để liên kết đầy đủ metrics & metadata)",
)
parser.add_argument(
    "--artifact-path",
    type=str,
    default=os.getenv("ARTIFACT_PATH", "model"),
    help="Thư mục con chứa model artifact trong run (mặc định: 'model')",
)
parser.add_argument(
    "--model-uri",
    type=str,
    default=os.getenv("MODEL_URI"),
    help="Model URI dạng runs:/<run_id>/model hoặc s3://...",
)
parser.add_argument(
    "--model-name",
    type=str,
    default=os.getenv("MODEL_NAME", "pm25_forecasting"),
    help="Tên mô hình trong MLflow Model Registry",
)
parser.add_argument(
    "--alias",
    type=str,
    default=os.getenv("MODEL_ALIAS", "champion"),
    help="Alias để gán cho phiên bản vừa tạo (mặc định: champion, truyền rỗng nếu không muốn gán)",
)
parser.add_argument(
    "--no-alias",
    action="store_true",
    help="Không gán alias sau khi đăng ký",
)

args = parser.parse_args()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

model_name = args.model_name
run_id = args.run_id
artifact_path = args.artifact_path
model_uri = args.model_uri

# Xác định model_uri chuẩn
if run_id:
    model_uri = f"runs:/{run_id}/{artifact_path}"
elif model_uri:
    if model_uri.startswith("s3://"):
        print(
            "\n[!] CẢNH BÁO QUAN TRỌNG:\n"
            f"    Bạn đang truyền trực tiếp S3 path ({model_uri}) vào register_model!\n"
            "    Cách này sẽ khiến trường 'run_id' trong Model Registry bị rỗng (NULL), làm mất\n"
            "    liên kết với Experiment Run và khiến hệ thống kiểm định Champion bỏ qua so sánh metrics.\n"
            "    -> KHUYẾN NGHỊ: Hãy dùng tham số --run-id <RUN_ID> hoặc URI dạng 'runs:/<run_id>/model'.\n"
        )
else:
    print(
        "[!] Lỗi: Bạn cần cung cấp --run-id (hoặc biến RUN_ID trong .env) "
        "hoặc --model-uri (hoặc MODEL_URI)!"
    )
    sys.exit(1)

print(f"[*] Đang kết nối MLflow tại: {tracking_uri}")
print(f"[*] Đăng ký Model '{model_name}' từ URI: {model_uri}...")

try:
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    linked_run = registered_model.run_id or "TRỐNG (Không có do dùng URI ngoài)"
    print(
        f"[✓] Đã đăng ký thành công: {registered_model.name} Version {registered_model.version} "
        f"(Run ID liên kết: {linked_run})"
    )

    if not args.no_alias and args.alias:
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name, alias=args.alias, version=registered_model.version
        )
        print(f"[★] Đã gán alias @{args.alias} thành công cho Version {registered_model.version}!")
except Exception as ex:
    print(f"[!] Lỗi khi đăng ký model: {ex}")
    sys.exit(1)
