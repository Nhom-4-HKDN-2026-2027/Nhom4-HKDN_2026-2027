"""
MLflow + PostgreSQL + MinIO smoke test
=====================================

Mục đích:
1. Kiểm tra MLflow Tracking Server hoạt động.
2. Tạo experiment/run, ghi params và metrics.
3. Upload artifact lên artifact store (MinIO/S3).
4. Tải artifact ngược về qua MLflow và kiểm tra nội dung.
5. Tạo một run riêng để kiểm tra timestamp/duration.

Cách chạy:
- Đặt file này trong thư mục dự án.
- Chạy bên trong container MLflow để dùng đúng cấu hình MinIO/S3 của service:

    docker compose cp mlflow_minio_smoke_test.py mlflow:/tmp/mlflow_minio_smoke_test.py
    docker compose exec mlflow python /tmp/mlflow_minio_smoke_test.py

Lưu ý:
- Script tạo experiment/run mới, không xóa hoặc sửa các run cũ.
- MLflow Tracking URI mặc định là http://localhost:5000 (khi chạy bên trong
  chính container MLflow).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "smoke-test"
ARTIFACT_FILENAME = "mlflow-smoke-test.txt"
ARTIFACT_CONTENT = "artifact upload successful"


def check_tracking_server() -> MlflowClient:
    """Kiểm tra server có phản hồi và lấy client."""
    print("\n[1/5] Checking MLflow Tracking Server...")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiments = client.search_experiments()
    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Server reachable. Experiments found: {len(experiments)}")
    return client


def get_or_create_experiment(client: MlflowClient) -> str:
    """Lấy experiment smoke-test; nếu chưa có thì tạo mới."""
    print("\n[2/5] Getting/creating experiment...")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        print(f"Created experiment: {EXPERIMENT_NAME} (ID={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using experiment: {EXPERIMENT_NAME} (ID={experiment_id})")

    return experiment_id


def test_log_and_read_artifact(experiment_id: str) -> str:
    """Ghi params/metrics/artifact, sau đó tải artifact và đọc lại."""
    print("\n[3/5] Testing params, metrics, artifact upload and download...")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with tempfile.TemporaryDirectory() as temp_dir:
        source_file = Path(temp_dir) / ARTIFACT_FILENAME
        source_file.write_text(ARTIFACT_CONTENT, encoding="utf-8")

        with mlflow.start_run(run_name="storage-smoke-test") as run:
            run_id = run.info.run_id
            mlflow.log_param("test_param", "ok")
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_artifact(str(source_file))

        saved_run = mlflow.get_run(run_id)
        print(f"Run ID: {run_id}")
        print(f"Run status: {saved_run.info.status}")
        print(f"Artifact URI: {saved_run.info.artifact_uri}")
        print(f"Param test_param: {saved_run.data.params.get('test_param')}")
        print(f"Metric accuracy: {saved_run.data.metrics.get('accuracy')}")

        # Download thông qua MLflow API; không truy cập MinIO trực tiếp.
        downloaded_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=ARTIFACT_FILENAME,
        )
        downloaded_file = Path(downloaded_path)
        downloaded_content = downloaded_file.read_text(encoding="utf-8")

        print(f"Downloaded path: {downloaded_path}")
        print(f"Downloaded content: {downloaded_content}")

        if downloaded_content != ARTIFACT_CONTENT:
            raise AssertionError(
                "Artifact content mismatch: MLflow downloaded a file, "
                "but its content is not as expected."
            )

    print("PASS: MLflow uploaded and downloaded the artifact successfully.")
    return run_id


def test_timestamp(experiment_id: str) -> tuple[str, int]:
    """Tạo run mới, chờ 3 giây rồi kiểm tra duration."""
    print("\n[4/5] Testing run timestamps...")
    client = MlflowClient(tracking_uri=TRACKING_URI)

    run = client.create_run(
        experiment_id=experiment_id,
        run_name="timestamp-check",
    )
    run_id = run.info.run_id
    start_time = run.info.start_time

    try:
        client.log_metric(run_id, "test_metric", 1.0)
        time.sleep(3)
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise

    finished_run = client.get_run(run_id)
    end_time = finished_run.info.end_time

    if start_time is None or end_time is None:
        raise AssertionError("Missing start_time or end_time in MLflow run.")

    duration_ms = end_time - start_time

    print(f"Timestamp run ID: {run_id}")
    print(f"START: {start_time}")
    print(f"END: {end_time}")
    print(f"DURATION_MS: {duration_ms}")

    if duration_ms <= 0:
        raise AssertionError(
            "Run duration is not positive. Check system/container clocks "
            "and MLflow timestamp recording."
        )

    print("PASS: Run duration is positive.")
    return run_id, duration_ms


def main() -> int:
    print("=" * 65)
    print("MLflow + PostgreSQL + MinIO smoke test")
    print("=" * 65)

    try:
        client = check_tracking_server()
        experiment_id = get_or_create_experiment(client)
        artifact_run_id = test_log_and_read_artifact(experiment_id)
        timestamp_run_id, duration_ms = test_timestamp(experiment_id)

        print("\n[5/5] SUMMARY")
        print("PASS - MLflow Tracking Server is reachable.")
        print("PASS - Experiment/run metadata and params/metrics were logged.")
        print("PASS - Artifact was downloaded through MLflow and content verified.")
        print(f"PASS - Timestamp duration is positive ({duration_ms} ms).")
        print(f"Artifact test run: {artifact_run_id}")
        print(f"Timestamp test run: {timestamp_run_id}")
        print("\nAll smoke tests passed.")
        return 0

    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
