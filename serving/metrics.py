import logging
import os

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

# OpenTelemetry Metrics Configuration (CONTRACTS.md Section 5.1 & 5.2)
ENABLE_OTEL = os.getenv("ENABLE_OTEL", "true").lower() in ("true", "1", "yes")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
ENVIRONMENT = os.getenv("ENVIRONMENT", "demo")

predictions_counter = None
latency_histogram = None

if ENABLE_OTEL:
    try:
        otel_exporter = OTLPMetricExporter(
            endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True,
        )
        metric_reader = PeriodicExportingMetricReader(
            otel_exporter,
            export_interval_millis=2000,
        )
        meter_provider = MeterProvider(
            resource=Resource.create({"service.name": "pm25-serving"}),
            metric_readers=[metric_reader],
        )
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter("pm25-serving")

        predictions_counter = meter.create_counter(
            "model_predictions_total",
            description="Total number of predictions made",
        )
        latency_histogram = meter.create_histogram(
            "model_prediction_duration_seconds",
            unit="s",
            description="Processing latency for one /predict request in seconds",
        )
    except Exception as ex:
        logger.warning("Không thể khởi tạo OpenTelemetry Metrics: %s", ex)


def record_prediction_metrics(
    count: int,
    duration_sec: float,
    model_name: str,
    model_version: str,
    station_id: str = "unknown",
) -> None:
    """Ghi nhận metrics thời gian thực theo chuẩn CONTRACTS.md Section 5.1 & 5.2."""
    try:
        if predictions_counter:
            predictions_counter.add(
                count,
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "environment": ENVIRONMENT,
                    "station_id": station_id,
                },
            )
        if latency_histogram:
            latency_histogram.record(
                duration_sec,
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "environment": ENVIRONMENT,
                },
            )
    except Exception as ex:
        logger.debug("Không thể ghi metric: %s", ex)
