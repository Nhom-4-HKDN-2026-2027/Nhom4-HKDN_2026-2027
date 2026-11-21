import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

model_uri = os.getenv("MODEL_URI")
model_name = os.getenv("MODEL_NAME")
model_alias = os.getenv("MODEL_ALIAS")

# 1. Đăng ký version mới
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Registered {registered_model.name} Version {registered_model.version}")

# 2. Gán alias @champion vào version vừa tạo
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name, alias=model_alias, version=registered_model.version
)
print(f"--> Gán alias @{model_alias} thành công cho Version {registered_model.version}!")
