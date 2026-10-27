import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Use SQLite instead of the deprecated local filesystem tracking backend
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")
MLFLOW_TRACKING_URI = "sqlite:///" + MLFLOW_DB.replace("\\", "/")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("[MLFLOW] Loading Lab 3 preprocessing artifacts...")

X_train = joblib.load(os.path.join(ARTIFACT_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(ARTIFACT_DIR, "y_train.joblib"))
X_valid = joblib.load(os.path.join(ARTIFACT_DIR, "X_valid.joblib"))
y_valid = joblib.load(os.path.join(ARTIFACT_DIR, "y_valid.joblib"))
preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    ),
])

# Configure MLflow to use the SQLite database
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Airline Satisfaction - Lab 4")

with mlflow.start_run(run_name="logistic_regression_baseline") as run:

    print("[MLFLOW] Training Logistic Regression...")

    model.fit(X_train, y_train)

    pred = model.predict(X_valid)
    prob = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_valid, pred)),
        "precision": float(precision_score(y_valid, pred, zero_division=0)),
        "recall": float(recall_score(y_valid, pred, zero_division=0)),
        "f1": float(f1_score(y_valid, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, prob)),
    }

    params = {
        "model": "LogisticRegression",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42,
        "test_size": 0.20,
        "split_random_state": 42,
        "target": "satisfaction",
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Save and log the trained model
    tracked_model_path = os.path.join(
        MODEL_DIR,
        "lab4_mlflow_model.joblib"
    )
    joblib.dump(model, tracked_model_path)

    mlflow.log_artifact(
        tracked_model_path,
        artifact_path="model"
    )

    metadata_path = os.path.join(
        ARTIFACT_DIR,
        "preprocessing_metadata.joblib"
    )

    if os.path.exists(metadata_path):
        mlflow.log_artifact(
            metadata_path,
            artifact_path="preprocessing"
        )

    # Generate classification report
    report = classification_report(
        y_valid,
        pred,
        target_names=["Neutral/Dissatisfied", "Satisfied"],
        digits=4,
        zero_division=0,
    )

    report_path = os.path.join(
        RESULTS_DIR,
        "lab4_mlflow_classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\nMetrics:\n")

        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    mlflow.log_artifact(
        report_path,
        artifact_path="evaluation"
    )

    # Save run information
    run_info = {
        "experiment": "Airline Satisfaction - Lab 4",
        "run_id": run.info.run_id,
        "metrics": metrics,
        "params": params,
    }

    with open(
        os.path.join(RESULTS_DIR, "lab4_mlflow_run.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(run_info, f, indent=2)

    print(f"[MLFLOW] Run ID: {run.info.run_id}")

    for name, value in metrics.items():
        print(f"[MLFLOW] {name}: {value:.4f}")

print("[MLFLOW] Experiment tracking complete.")
print("[MLFLOW] Tracking database saved to:", MLFLOW_DB)