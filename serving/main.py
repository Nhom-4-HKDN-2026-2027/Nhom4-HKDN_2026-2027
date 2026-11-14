
import os
import logging
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =========================
# Configuration
# =========================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "BreastCancerRandomForest"
)

MODEL_ALIAS = os.getenv("MODEL_ALIAS", "latest")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Model loading
# =========================

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    try:
        logger.info("Loading model from: %s", model_uri)
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully")
    except Exception:
        logger.exception("Failed to load model")
        raise

    yield

    model = None


app = FastAPI(
    title="Adaptive Model Updating API",
    description="Prediction API powered by MLflow Model Registry",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================
# Request / Response
# =========================

class PredictionRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description="Danh sách các mẫu dữ liệu, mỗi mẫu có 30 features",
    )


class PredictionResponse(BaseModel):
    model_name: str
    predictions: list[int]


# =========================
# Health check
# =========================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
    }


# =========================
# Prediction endpoint
# =========================

@app.post("/predict", response_model=PredictionResponse)
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
