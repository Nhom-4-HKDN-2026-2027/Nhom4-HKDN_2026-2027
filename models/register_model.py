import os
from pathlib import Path
import mlflow
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

model_uri = os.getenv("MODEL_URI")
model_name = os.getenv("MODEL_NAME")

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("===== MODEL REGISTRY =====")
print("Model name:", registered_model.name)
print("Version:", registered_model.version)
print("Status:", registered_model.status)
print("Source:", registered_model.source)