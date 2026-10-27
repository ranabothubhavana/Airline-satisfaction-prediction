import os
import json
import mlflow

from mlflow.tracking import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")
TRACKING_URI = "sqlite:///" + MLFLOW_DB.replace("\\", "/")

MODEL_NAME = "Airline_Satisfaction_Production_Model"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

print("[REPORT] Generating model registry report...")

try:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
except Exception as error:
    raise RuntimeError(
        "Could not read the model registry. Run the Lab 6 training step first."
    ) from error

if not versions:
    raise RuntimeError(
        f"No registered versions found for {MODEL_NAME}."
    )

model_versions = []

for version in sorted(versions, key=lambda item: int(item.version)):
    run = client.get_run(version.run_id)

    model_versions.append({
        "model_name": version.name,
        "version": int(version.version),
        "stage": version.current_stage,
        "run_id": version.run_id,
        "status": version.status,
        "accuracy": run.data.metrics.get("accuracy"),
        "precision": run.data.metrics.get("precision"),
        "recall": run.data.metrics.get("recall"),
        "f1": run.data.metrics.get("f1"),
        "roc_auc": run.data.metrics.get("roc_auc"),
    })

report_path = os.path.join(
    RESULTS_DIR,
    "lab6_registry_report.json",
)

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(model_versions, file, indent=2)

print(f"\n[REPORT] Registered model: {MODEL_NAME}")

for item in model_versions:
    print(
        f"Version {item['version']} | "
        f"Stage: {item['stage']} | "
        f"Recall: {item['recall']}"
    )

print("[REPORT] Report saved to:", report_path)