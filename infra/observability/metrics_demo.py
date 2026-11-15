import random
import time

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource

exporter = OTLPMetricExporter(
    endpoint="http://localhost:4317",
    insecure=True,
)

reader = PeriodicExportingMetricReader(
    exporter,
    export_interval_millis=2000,
)

provider = MeterProvider(
    resource=Resource.create({"service.name": "prediction-demo"}),
    metric_readers=[reader],
)
metrics.set_meter_provider(provider)

meter = metrics.get_meter("prediction-demo")
prediction_error = meter.create_histogram(
    "prediction_error",
    description="Absolute prediction error on synthetic normalized targets",
)

random.seed(42)

try:
    while True:
        actual = random.random()
        predicted = actual + random.uniform(-0.2, 0.2)
        error = abs(predicted - actual)

        prediction_error.record(
            error,
            {"model_name": "demo_model", "environment": "demo"},
        )
        print(f"prediction_error = {error:.4f}", flush=True)
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    provider.shutdown()