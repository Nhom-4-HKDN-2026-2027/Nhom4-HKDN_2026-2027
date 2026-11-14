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
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "latest")
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

    if MODEL_URI:
        model_uri = MODEL_URI
    else:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    logger.info("MLflow URI: %s", MLFLOW_TRACKING_URI)
    logger.info("Model URI: %s", model_uri)

    try:
        model = mlflow.pyfunc.load_model(model_uri)
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
    features: list[list[float]] = Field(
        ...,
        description="Danh sách các mẫu dữ liệu, mỗi mẫu gồm đúng 30 features số thực (chuẩn Breast Cancer).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [
                        [
                            17.99,
                            10.38,
                            122.8,
                            1001.0,
                            0.1184,
                            0.2776,
                            0.3001,
                            0.1471,
                            0.2419,
                            0.07871,
                            1.095,
                            0.9053,
                            8.589,
                            153.4,
                            0.006399,
                            0.04904,
                            0.05373,
                            0.01587,
                            0.03003,
                            0.006193,
                            25.38,
                            17.33,
                            184.6,
                            2019.0,
                            0.1622,
                            0.6656,
                            0.7119,
                            0.2654,
                            0.4601,
                            0.1189,
                        ]
                    ]
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    model_name: str = Field(..., description="Tên model phục vụ dự đoán")
    predictions: list[int] = Field(
        ..., description="Danh sách nhãn dự đoán (0: Malignant / Ác tính, 1: Benign / Lành tính)"
    )


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
