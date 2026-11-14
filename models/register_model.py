import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Model URI lấy từ kết quả train ở bước 2.3
model_uri = "models:/m-7aa59c5b12794a2394ecce75d1ac4ef6"

# Đăng ký model
registered_model = mlflow.register_model(model_uri=model_uri, name="BreastCancerRandomForest")

print("===== MODEL REGISTRY =====")
print("Model name:", registered_model.name)
print("Version:", registered_model.version)
print("Status:", registered_model.status)
print("Source:", registered_model.source)
