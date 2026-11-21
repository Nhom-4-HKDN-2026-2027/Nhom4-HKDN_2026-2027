import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "pm25-forecasting")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_URI = os.getenv("MODEL_URI")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Ưu tiên MODEL_URI nếu có, ngược lại fallback về tên model + alias (mặc định champion)
    resolved_model_uri = MODEL_URI or f"models:/{MODEL_NAME}@{MODEL_ALIAS or 'champion'}"
    logger.info("MLflow URI: %s", MLFLOW_TRACKING_URI)
    logger.info("Model URI: %s", resolved_model_uri)
    try:
        model = mlflow.pyfunc.load_model(resolved_model_uri)
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Failed to load model")
        raise
    yield
    model = None


# FastAPI app initialization
app = FastAPI(
    title="Prediction API",
    description="Interactive Swagger UI để test trực quan mô hình MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)


# Request / Response Schemas


class PredictionRequest(BaseModel):
    features: list[dict[str, float]] | list[list[float]] = Field(
        ..., description="Dữ liệu quan trắc đầu vào để dự báo nồng độ PM2.5."
    )


class PredictionResponse(BaseModel):
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Mã định danh lượt gọi API"
    )
    status: str = Field(default="success", description="Trạng thái xử lý")
    model_name: str
    model_alias: str | None = None
    predictions: list[float] = Field(..., description="Nồng độ PM2.5 dự báo (µg/m³)")


# Endpoints


@app.get("/", include_in_schema=False)
def index():
    """Tự động chuyển hướng từ trang chủ sang Swagger UI trực quan."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Monitoring"], summary="Kiểm tra trạng thái Service và Model")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        input_data = pd.DataFrame(request.features)
        preds = model.predict(input_data)

        # Ép sang float (PM2.5 không âm)
        return PredictionResponse(
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS or "champion",
            predictions=[max(0.0, float(v)) for v in preds],
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


# /reload GIÚP không cần restart container
@app.post("/reload", tags=["Management"], summary="Reload model champion mới nhất")
def reload_model():
    global model
    resolved_uri = MODEL_URI or f"models:/{MODEL_NAME}@{MODEL_ALIAS or 'champion'}"
    model = mlflow.pyfunc.load_model(resolved_uri)
    return {"status": "reloaded", "model_uri": resolved_uri}
