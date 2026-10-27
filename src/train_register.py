import os
import time
import joblib
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")
TRACKING_URI = "sqlite:///" + MLFLOW_DB.replace("\\", "/")

EXPERIMENT_NAME = "Airline Satisfaction - Lab 6"
MODEL_NAME = "Airline_Satisfaction_Production_Model"

print("[LAB 6] Loading training and validation artifacts...")

X_train = joblib.load(os.path.join(ARTIFACT_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(ARTIFACT_DIR, "y_train.joblib"))
X_valid = joblib.load(os.path.join(ARTIFACT_DIR, "X_valid.joblib"))
y_valid = joblib.load(os.path.join(ARTIFACT_DIR, "y_valid.joblib"))
preprocessor = joblib.load(
    os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient(tracking_uri=TRACKING_URI)

print("[LAB 6] Training Logistic Regression...")

with mlflow.start_run(run_name="airline_registry_training") as run:
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(
            precision_score(y_valid, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_valid, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
    }

    mlflow.log_params({
        "model": "LogisticRegression",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42,
        "target": "satisfaction",
    })

    mlflow.log_metrics(metrics)

    model_path = os.path.join(MODEL_DIR, "lab6_registry_model.joblib")
    joblib.dump(model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model_files")

    print("[LAB 6] Registering model with MLflow...")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="registered_model",
        registered_model_name=MODEL_NAME,
    )

    run_id = run.info.run_id

# Wait for the registered model version to become available.
latest_version = None

for _ in range(30):
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )
    matching_versions = [
        version for version in versions
        if version.run_id == run_id
    ]

    if matching_versions:
        latest_version = max(
            matching_versions,
            key=lambda version: int(version.version),
        )
        break

    time.sleep(1)

if latest_version is None:
    raise RuntimeError(
        "Model was logged, but its registered version was not found."
    )

report = {
    "experiment": EXPERIMENT_NAME,
    "model_name": MODEL_NAME,
    "model_version": int(latest_version.version),
    "run_id": run_id,
    "target": "satisfaction",
    "metrics": metrics,
}

import json

report_path = os.path.join(RESULTS_DIR, "lab6_registry_training.json")

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("\n[LAB 6] Training and registration completed.")
print("[LAB 6] Registered model:", MODEL_NAME)
print("[LAB 6] Model version:", latest_version.version)

for name, value in metrics.items():
    print(f"[LAB 6] {name}: {value:.4f}")

print("[LAB 6] Report saved to:", report_path)