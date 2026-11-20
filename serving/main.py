import logging
import os
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
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
MODEL_URI = os.getenv("MODEL_URI")

if not MLFLOW_TRACKING_URI:
    raise RuntimeError("Missing MLFLOW_TRACKING_URI in .env")

if not MODEL_NAME:
    raise RuntimeError("Missing MODEL_NAME in .env")

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
        ...,
        description="Dữ liệu quan trắc đầu vào để dự báo nồng độ PM2.5."
    )


class PredictionResponse(BaseModel):
    model_name: str
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


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        if any(len(row) != 30 for row in request.features):
            raise HTTPException(
                status_code=422,
                detail="Each sample must contain exactly 30 features",
            )

        input_data = pd.DataFrame(request.features)

        predictions = model.predict(input_data)

        return PredictionResponse(
            model_name=MODEL_NAME,
            predictions=[int(value) for value in predictions],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
