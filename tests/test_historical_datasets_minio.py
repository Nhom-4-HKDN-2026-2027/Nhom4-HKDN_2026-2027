import hashlib
import json
from unittest.mock import Mock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from ingest.historical_datasets import prepare_beijing, prepare_elec2
from ingest.minio_storage import BUCKETS, ensure_buckets, upload_historical_datasets


def test_beijing_roundtrip(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "station_id": ["beijing_A", "beijing_B"],
            "event_time": ["2020-01-01T08:00:00+08:00", "2020-01-01T01:00:00Z"],
            **{
                c: [1.0, 2.0]
                for c in (
                    "pm25_ug_m3",
                    "temperature_c",
                    "relative_humidity_pct",
                    "wind_speed_m_s",
                    "wind_direction_deg",
                )
            },
        }
    ).to_csv(source, index=False)
    directory = prepare_beijing(source, tmp_path / "out")
    files = sorted(directory.glob("station_id=*/data.parquet"))
    assert len(files) == 2
    frame = pd.read_parquet(files[0])
    assert str(frame.event_time.dtype) == "datetime64[ns, UTC]"
    assert frame.event_time.iloc[0] == pd.Timestamp("2020-01-01T00:00:00Z")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["row_count"] == 2
    assert manifest["sha256_checksum"][files[0].relative_to(directory).as_posix()] == (
        hashlib.sha256(files[0].read_bytes()).hexdigest()
    )


def test_elec2_roundtrip(tmp_path):
    rows = [
        ({"date": 0.1, "day": 2, "period": 0.5, "nswprice": 0.2}, True),
        ({"date": 0.2, "day": 3, "period": 1.0, "nswprice": 0.3}, False),
    ]
    directory = prepare_elec2(tmp_path, samples=rows)
    frame = pd.read_parquet(directory / "elec2.parquet")
    assert frame.sample_idx.tolist() == [0, 1]
    assert frame.target.tolist() == [True, False]
    assert str(frame.day.dtype) == "int64"
    assert str(frame.period.dtype) == "float64"
    assert frame.period.tolist() == [0.5, 1.0]


def test_upload_idempotency_and_errors(tmp_path):
    directory = prepare_elec2(tmp_path, samples=[({"day": 1, "period": 0.5}, True)])
    client = Mock()
    assert upload_historical_datasets(client, [directory]) == []
    assert not client.upload_file.called
    assert len(upload_historical_datasets(client, [directory], force=True)) == 2
    client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
    keys = upload_historical_datasets(client, [directory])
    assert keys == ["elec2/elec2.parquet", "elec2/manifest.json"]
    client.head_object.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadObject")
    with pytest.raises(ClientError):
        upload_historical_datasets(client, [directory])


def test_buckets():
    client = Mock()
    client.head_bucket.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadBucket")
    ensure_buckets(client)
    assert set(BUCKETS) == {"raw", "staging", "features", "models", "mlflow"}
    assert client.create_bucket.call_count == 5


def test_compose_layout():
    from pathlib import Path

    import yaml

    config = yaml.safe_load(Path("docker-compose.yml").read_text())
    services = config["services"]
    assert services["minio"]["volumes"] == ["miniodata:/data"]
    for bucket in BUCKETS:
        assert bucket in services["minio-init"]["command"][0].replace(";", " ").split()
    assert services["minio"]["image"] == "chainguard/minio:latest"
    assert services["minio"]["healthcheck"]["test"] == [
        "CMD", "mc", "--config-dir", "/tmp/mc", "ready", "local"
    ]
    assert services["grafana"]["ports"] == ["127.0.0.1:3000:3000"]
    assert "data-seed" not in services
    assert all("build" not in service for service in services.values())


def test_beijing_rejects_path_traversal(tmp_path):
    from pathlib import Path

    source = tmp_path / "bad.csv"
    frame = pd.read_csv(Path("data/sample/beijing_multisite_features.csv")).head(1)
    frame["station_id"] = "../escape"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="station_id"):
        prepare_beijing(source, tmp_path / "out")
