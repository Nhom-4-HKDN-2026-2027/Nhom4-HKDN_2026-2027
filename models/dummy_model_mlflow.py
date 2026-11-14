import mlflow
from pathlib import Path
from dotenv import set_key

import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 1. CONFIGURE MLFLOW

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("adaptive-model-updating")

print("MLflow Tracking URI:", mlflow.get_tracking_uri())

# 2. LOAD DATASET

data = load_breast_cancer()

X = data.data
y = data.target

print("\n===== DATASET INFORMATION =====")
print("Dataset:", "Breast Cancer Wisconsin")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. DEFINE MODEL PARAMETERS

params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

model = RandomForestClassifier(**params)

# 4. TRAIN AND LOG TO MLFLOW

with mlflow.start_run(run_name="random-forest-baseline") as run:
    print("\n===== MODEL TRAINING =====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # Log parameters
    mlflow.log_params(params)

    # Log dataset information
    mlflow.log_param("dataset", "breast_cancer")
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    # Log metrics
    mlflow.log_metrics(metrics)

    # Infer model input/output signature
    signature = infer_signature(X_test, y_pred)

    # Log trained model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="random-forest-model",
        signature=signature,
        input_example=X_test[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("\n===== MODEL EVALUATION =====")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    env_path = Path(__file__).resolve().parent.parent / ".env"

    set_key(
        str(env_path),
        "MODEL_URI",
        model_info.model_uri
    )

    print("Saved MODEL_URI:", model_info.model_uri)
    
    print("\n===== MLFLOW INFORMATION =====")
    print("Experiment ID:", run.info.experiment_id)
    print("Run ID:", run.info.run_id)
    print("Artifact URI:", run.info.artifact_uri)
    print("Model URI:", model_info.model_uri)

print("\nModel training and MLflow logging completed!")
