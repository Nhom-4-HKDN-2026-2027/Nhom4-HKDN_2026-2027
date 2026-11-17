import hashlib
import io
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
    files = sorted(directory.glob("year=*/station_id=*/data.parquet"))
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
    manifest = json.loads((directory / "manifest.json").read_text())
    client.get_object.side_effect = lambda **kwargs: {
        "Body": io.BytesIO(json.dumps(manifest).encode())
    }
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
        "CMD",
        "mc",
        "--config-dir",
        "/tmp/mc",
        "ready",
        "local",
    ]
    assert "build" not in services["minio"]
    assert services["grafana"]["ports"] == ["127.0.0.1:3000:3000"]
    assert "data-seed" not in services
    assert "seed_cache" not in config["volumes"]
    assert "seed_venv" not in config["volumes"]


def test_beijing_rejects_path_traversal(tmp_path):
    from pathlib import Path

    source = tmp_path / "bad.csv"
    frame = pd.read_csv(Path("data/sample/beijing_multisite_features.csv")).head(1)
    frame["station_id"] = "../escape"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="station_id"):
        prepare_beijing(source, tmp_path / "out")


def test_kaggle_archive_years_and_timezone(tmp_path):
    from zipfile import ZipFile

    archive = tmp_path / "beijing.zip"
    source = pd.DataFrame(
        {
            "year": [2014, 2015],
            "month": [12, 1],
            "day": [31, 1],
            "hour": [23, 8],
            "station": ["Aotizhongxin"] * 2,
            "PM2.5": [10.0, None],
            "TEMP": [10.0, 10.0],
            "DEWP": [10.0, 20.0],
            "WSPM": [1.0, 2.0],
            "wd": ["N", "NNE"],
        }
    )
    with ZipFile(archive, "w") as z:
        z.writestr("PRSA_Data_Aotizhongxin_20130301-20170228.csv", source.to_csv(index=False))
    directory = prepare_beijing(archive, tmp_path / "out")
    files = sorted(directory.glob("year=*/station_id=*/data.parquet"))
    assert [p.parent.parent.name for p in files] == ["year=2014", "year=2015"]
    frame = pd.read_parquet(
        files,
        columns=[
            "station_id",
            "event_time",
            "pm25_ug_m3",
            "temperature_c",
            "relative_humidity_pct",
            "wind_speed_m_s",
            "wind_direction_deg",
        ],
    )
    assert frame.station_id.tolist() == ["beijing_Aotizhongxin"] * 2
    assert frame.event_time.iloc[0] == pd.Timestamp("2014-12-31T15:00:00Z")
    assert frame.relative_humidity_pct.iloc[0] == pytest.approx(100.0)
    assert pd.isna(frame.relative_humidity_pct.iloc[1])
    assert pd.isna(frame.pm25_ug_m3.iloc[1])
    assert frame.wind_direction_deg.tolist() == [0.0, 22.5]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["years"] == [2014, 2015]
    assert "kaggle.com" in manifest["source_url"]


def test_beijing_rerun_removes_stale_partitions(tmp_path):
    from pathlib import Path

    source = tmp_path / "input.csv"
    frame = pd.read_csv(Path("data/sample/beijing_multisite_features.csv"))
    frame.to_csv(source, index=False)
    directory = prepare_beijing(source, tmp_path / "out")
    frame[frame.station_id == frame.station_id.iloc[0]].to_csv(source, index=False)
    prepare_beijing(source, tmp_path / "out")
    files = list(directory.rglob("*.parquet"))
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(files) == 1
    assert sum(len(pd.read_parquet(p)) for p in files) == manifest["row_count"] == 168


def test_upload_requires_force_for_changed_dataset(tmp_path):
    directory = prepare_elec2(tmp_path, samples=[({"day": 1, "period": 0.5}, True)])
    old_manifest = {"sha256_checksum": {"old.parquet": "old"}, "row_count": 1}
    client = Mock()
    client.get_object.side_effect = lambda **kwargs: {
        "Body": io.BytesIO(json.dumps(old_manifest).encode())
    }
    with pytest.raises(ValueError, match="force"):
        upload_historical_datasets(client, [directory])
    assert not client.upload_file.called
    upload_historical_datasets(client, [directory], force=True)
    client.delete_object.assert_called_once_with(Bucket="raw", Key="elec2/old.parquet")
    assert client.upload_file.call_args.args[2] == "elec2/manifest.json"


def test_kaggle_download_validates_and_reuses_cache(tmp_path, monkeypatch):
    from zipfile import ZipFile

    import ingest.historical_datasets as historical

    content = io.BytesIO()
    with ZipFile(content, "w") as archive:
        for number in range(12):
            archive.writestr(f"station_{number}.csv", "year,station\n2013,A\n")
    calls = []

    def fake_download(*args, **kwargs):
        calls.append(args[0])
        return io.BytesIO(content.getvalue())

    monkeypatch.setattr(historical, "urlopen", fake_download)
    destination = tmp_path / "downloads" / "beijing.zip"
    assert historical.download_beijing(destination) == destination
    assert destination.read_bytes() == content.getvalue()
    assert historical.download_beijing(destination) == destination
    assert len(calls) == 1


def test_kaggle_download_does_not_cache_html(tmp_path, monkeypatch):
    from zipfile import BadZipFile

    import ingest.historical_datasets as historical

    monkeypatch.setattr(
        historical, "urlopen", lambda *args, **kwargs: io.BytesIO(b"<html>Denied</html>")
    )
    destination = tmp_path / "downloads" / "beijing.zip"
    with pytest.raises(BadZipFile):
        historical.download_beijing(destination)
    assert not destination.exists()
