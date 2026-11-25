import random
import time
from dataclasses import dataclass

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource


@dataclass(frozen=True)
class PendingPrediction:
    target_time: float
    label_due_at: float
    actual: float
    predicted: float


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

attributes = {
    "model_name": "demo_model",
    "model_version": "demo-v1",
    "environment": "demo",
    "station_id": "synthetic_station",
}

prediction_error = meter.create_histogram(
    "model_prediction_error",
    description="Absolute error on synthetic normalized targets",
)
squared_error = meter.create_histogram(
    "model_prediction_squared_error",
    description="Squared error on synthetic normalized targets",
)
predictions_created = meter.create_counter(
    "model_predictions_created",
    description="Number of synthetic forecasts created",
)
labels_received = meter.create_counter(
    "model_prediction_labels_received",
    description="Number of labels matched to pending forecasts",
)
label_delay = meter.create_histogram(
    "model_prediction_label_delay",
    unit="s",
    description="Time from target observation to label availability",
)

pending: list[PendingPrediction] = []


def observe_pending(options):
    return [metrics.Observation(len(pending), attributes)]


meter.create_observable_gauge(
    "model_prediction_labels_pending",
    callbacks=[observe_pending],
    description="Number of forecasts awaiting labels",
)

rng = random.Random(42)
received_count = 0

try:
    while True:
        now = time.monotonic()

        # The target observation occurs now; its label is delivered later.
        # Ground truth is hidden from evaluation until the simulated delivery.
        actual = rng.random()
        predicted = actual + rng.uniform(-0.2, 0.2)
        delivery_delay = rng.uniform(5.0, 20.0)

        pending.append(
            PendingPrediction(
                target_time=now,
                label_due_at=now + delivery_delay,
                actual=actual,
                predicted=predicted,
            )
        )
        predictions_created.add(1, attributes)

        ready = [item for item in pending if item.label_due_at <= now]

        for item in ready:
            error = abs(item.predicted - item.actual)

            prediction_error.record(error, attributes)
            squared_error.record(error**2, attributes)
            label_delay.record(
                item.label_due_at - item.target_time,
                attributes,
            )
            labels_received.add(1, attributes)
            received_count += 1

        # Remove evaluated predictions so each pair is counted once.
        pending[:] = [item for item in pending if item.label_due_at > now]

        print(
            f"created=1 matched={len(ready)} "
            f"pending={len(pending)} received_total={received_count}",
            flush=True,
        )
        time.sleep(1)
except KeyboardInterrupt:
    print(
        f"Stopping demo; {len(pending)} pending forecasts remain unevaluated.",
        flush=True,
    )
finally:
    provider.shutdown()
