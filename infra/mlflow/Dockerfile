FROM python:3.11-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
        "typing-extensions>=4.14.0" \
        mlflow \
        psycopg2-binary \
        boto3

# Kiểm tra thư viện ngay khi build image
RUN python -c "from typing_extensions import Sentinel; import mlflow; print('MLflow:', mlflow.__version__)"

EXPOSE 5000