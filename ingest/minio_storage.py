"""S3 storage operations for the shared MinIO layout."""

import hashlib
import json
import os

import boto3
from botocore.exceptions import ClientError

BUCKETS = ("raw", "staging", "features", "models", "mlflow")


def create_client():
    """Read credentials from the environment or the standard AWS provider chain."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("MINIO_ROOT_USER") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("MINIO_ROOT_PASSWORD")
        or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def exists(operation, **kwargs):
    """Treat only a missing resource as absent; propagate auth and service errors."""
    try:
        operation(**kwargs)
        return True
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}:
            raise
        return False


def ensure_buckets(client, buckets=BUCKETS):
    """Create missing buckets without changing existing ones."""
    for bucket in buckets:
        if not exists(client.head_bucket, Bucket=bucket):
            client.create_bucket(Bucket=bucket)


def upload_historical_datasets(client, directories, force: bool = False) -> list[str]:
    """Publish manifest-listed objects; require force to replace a different dataset."""
    plans = []
    # Check every dataset before mutating S3, so a later mismatch cannot cause partial updates.
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        for path, checksum in manifest["sha256_checksum"].items():
            local_path = (directory / path).resolve()
            if not local_path.is_relative_to(directory.resolve()) or not path.endswith(".parquet"):
                raise ValueError("Invalid manifest Parquet path")
            if hashlib.sha256(local_path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Local checksum mismatch: {path}")
        previous = {}
        key = f"{directory.name}/manifest.json"
        if exists(client.head_object, Bucket="raw", Key=key):
            response = client.get_object(Bucket="raw", Key=key)
            with response["Body"] as body:
                previous = json.loads(body.read())
            if previous.get("sha256_checksum") != manifest["sha256_checksum"] and not force:
                raise ValueError(f"Dataset {directory.name} changed; use --force to replace it")
        plans.append((directory, manifest, previous))
    uploaded = []
    for directory, manifest, previous in plans:
        for path in sorted(manifest["sha256_checksum"]):
            key = f"{directory.name}/{path}"
            if force or not exists(client.head_object, Bucket="raw", Key=key):
                client.upload_file(str(directory / path), "raw", key)
                uploaded.append(key)
        if force:
            obsolete = set(previous.get("sha256_checksum", {})) - set(manifest["sha256_checksum"])
            for path in sorted(obsolete):
                if path.startswith("/") or ".." in path.split("/") or not path.endswith(".parquet"):
                    raise ValueError("Invalid previous manifest path; refusing deletion")
                client.delete_object(Bucket="raw", Key=f"{directory.name}/{path}")
        key = f"{directory.name}/manifest.json"
        if force or not exists(client.head_object, Bucket="raw", Key=key):
            client.upload_file(str(directory / "manifest.json"), "raw", key)
            uploaded.append(key)
    return uploaded
