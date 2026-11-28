import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from mlflow.tracking import MlflowClient

try:
    from serving.metrics import record_prediction_metrics
    from serving.schemas import (
        PredictionRequest,
        PredictionResponse,
        compute_target_time,
    )
except ImportError:
    from metrics import record_prediction_metrics
    from schemas import (
        PredictionRequest,
        PredictionResponse,
        compute_target_time,
    )

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "pm25_forecasting")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_URI = os.getenv("MODEL_URI")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
model_version: str = "unknown"


def resolve_and_load_model():
    """Tải model từ MLflow Model Registry và xác định model_version cụ thể."""
    global model, model_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    resolved_model_uri = MODEL_URI or f"models:/{MODEL_NAME}@{MODEL_ALIAS or 'champion'}"
    logger.info("MLflow URI: %s", MLFLOW_TRACKING_URI)
    logger.info("Model URI: %s", resolved_model_uri)
    try:
        model = mlflow.pyfunc.load_model(resolved_model_uri)
        v = getattr(getattr(model, "metadata", None), "model_version", None)
        if v:
            model_version = str(v)
        else:
            try:
                client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
                target_alias = MODEL_ALIAS or "champion"
                champ_ver = client.get_model_version_by_alias(MODEL_NAME, target_alias)
                model_version = str(champ_ver.version)
            except Exception:
                model_version = "1"
        logger.info("Model loaded successfully (Version: %s)", model_version)
    except Exception:
        logger.exception("Failed to load model")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    resolve_and_load_model()
    yield
    global model
    model = None


# FastAPI app initialization
app = FastAPI(
    title="Prediction API",
    description="Interactive Swagger UI để test trực quan mô hình MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)


# Endpoints


@app.get("/", include_in_schema=False)
def index():
    """Tự động chuyển hướng sang Swagger UI trực quan."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Monitoring"], summary="Kiểm tra trạng thái Service và Model")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "model_alias": MODEL_ALIAS or "champion",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    t0 = time.perf_counter()
    try:
        input_data = pd.DataFrame(request.features)
        preds = model.predict(input_data)

        target_time = None
        station_id = "unknown"
        if request.timestamp:
            target_time = compute_target_time(request.timestamp)
        elif (
            isinstance(request.features, list)
            and len(request.features) > 0
            and isinstance(request.features[0], dict)
        ):
            first_time = request.features[0].get("event_time") or request.features[0].get(
                "timestamp"
            )
            if first_time:
                target_time = compute_target_time(first_time)
            station_id = str(request.features[0].get("station_id") or "unknown")

        predictions_list = [max(0.0, float(v)) for v in preds]
        duration_sec = time.perf_counter() - t0

        # Ghi nhận OpenTelemetry Metrics (Mục 5.1 & 5.2 CONTRACTS.md)
        record_prediction_metrics(
            count=len(predictions_list),
            duration_sec=duration_sec,
            model_name=MODEL_NAME,
            model_version=model_version,
            station_id=station_id,
        )

        return PredictionResponse(
            model_name=MODEL_NAME,
            model_version=model_version,
            model_alias=MODEL_ALIAS or "champion",
            target_time=target_time,
            predictions=predictions_list,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


# /reload GIÚP không cần restart container
@app.post("/reload", tags=["Management"], summary="Reload model champion mới nhất")
def reload_model():
    resolve_and_load_model()
    resolved_uri = MODEL_URI or f"models:/{MODEL_NAME}@{MODEL_ALIAS or 'champion'}"
    return {
        "status": "reloaded",
        "model_uri": resolved_uri,
        "model_version": model_version,
    }
