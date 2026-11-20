import pytest
from unittest.mock import MagicMock, patch

# Tự động bỏ qua file test này nếu môi trường thiếu fastapi hoặc httpx
fastapi = pytest.importorskip("fastapi", reason="Cần cài fastapi để chạy test serving")
pytest.importorskip("httpx", reason="Cần cài httpx để dùng FastAPI TestClient")

from fastapi.testclient import TestClient

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

if __name__ == "__main__":
    pytest.main(["-v", __file__])