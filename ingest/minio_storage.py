"""S3 storage operations for the shared MinIO layout."""

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


def upload_historical_datasets(client, directories, force=False):
    """Upload Parquet objects followed by their manifest; return uploaded keys."""
    uploaded = []
    for directory in directories:
        paths = sorted(directory.rglob("*.parquet")) + [directory / "manifest.json"]
        for path in paths:
            key = f"{directory.name}/{path.relative_to(directory).as_posix()}"
            if force or not exists(client.head_object, Bucket="raw", Key=key):
                client.upload_file(str(path), "raw", key)
                uploaded.append(key)
    return uploaded
