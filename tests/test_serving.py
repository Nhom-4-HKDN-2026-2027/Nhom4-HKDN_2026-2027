import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["ENABLE_OTEL"] = "false"


@pytest.fixture
def mock_mlflow_model():
    mock_model = MagicMock()
    # Mock model trả về kết quả dự báo PM2.5 dạng float
    mock_model.predict.return_value = [45.2, 38.7]
    return mock_model


def test_health_check_endpoint():
    with patch("mlflow.pyfunc.load_model"):
        from serving.main import app

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"


def test_predict_endpoint(mock_mlflow_model):
    with patch("mlflow.pyfunc.load_model", return_value=mock_mlflow_model):
        from serving.main import app

        with TestClient(app) as client:
            payload = {
                "features": [[1.0] * 7, [2.0] * 7]  # Mock 7 features chuẩn PM2.5
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
            assert "model_version" in data
            assert data["model_version"] is not None


def test_predict_endpoint_with_timestamp(mock_mlflow_model):
    with patch("mlflow.pyfunc.load_model", return_value=mock_mlflow_model):
        from serving.main import app

        with TestClient(app) as client:
            payload = {
                "timestamp": "2026-11-28T15:00:00Z",
                "features": [{"pm25_ug_m3": 45.0, "station_id": "test_station"}],
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["target_time"] == "2026-11-28T16:00:00Z"
            assert data["model_version"] is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
