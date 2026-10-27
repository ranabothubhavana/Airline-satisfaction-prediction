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

print("[LIFECYCLE] Checking registered model...")

try:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
except Exception as error:
    raise RuntimeError(
        "Could not find the registered model. Run train_registry.py first."
    ) from error

if not versions:
    raise RuntimeError(
        f"No versions found for {MODEL_NAME}. Run train_registry.py first."
    )

# Find the latest registered version.
challenger = max(versions, key=lambda version: int(version.version))
challenger_version = str(challenger.version)

# Get the challenger recall metric from its MLflow run.
challenger_run = client.get_run(challenger.run_id)
challenger_recall = challenger_run.data.metrics.get("recall")

if challenger_recall is None:
    raise RuntimeError(
        "The registered model's MLflow run does not contain a recall metric."
    )

print(f"[LIFECYCLE] Challenger version: {challenger_version}")
print(f"[LIFECYCLE] Challenger recall: {challenger_recall:.4f}")

# Move the challenger to Staging.
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=challenger_version,
    stage="Staging",
    archive_existing_versions=False,
)

production_versions = client.get_latest_versions(
    MODEL_NAME,
    stages=["Production"],
)

promoted = False
previous_production_version = None
production_recall = None

if not production_versions:
    # First registered version: promote it to Production.
    print("[LIFECYCLE] No Production model exists yet.")
    print("[LIFECYCLE] Promoting the first model to Production.")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=challenger_version,
        stage="Production",
        archive_existing_versions=True,
    )
    promoted = True

else:
    champion = max(
        production_versions,
        key=lambda version: int(version.version),
    )
    previous_production_version = str(champion.version)

    champion_run = client.get_run(champion.run_id)
    production_recall = champion_run.data.metrics.get("recall")

    if production_recall is None:
        raise RuntimeError(
            "The existing Production model has no recorded recall metric."
        )

    print(
        f"[LIFECYCLE] Production version: "
        f"{previous_production_version}"
    )
    print(f"[LIFECYCLE] Production recall: {production_recall:.4f}")

    if challenger_version == previous_production_version:
        print(
            "[LIFECYCLE] The latest version is already in Production. "
            "No transition is needed."
        )
        promoted = True

    elif challenger_recall >= production_recall:
        print("[LIFECYCLE] Challenger meets the recall quality gate.")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=challenger_version,
            stage="Production",
            archive_existing_versions=True,
        )
        promoted = True

    else:
        print("[LIFECYCLE] Challenger did not meet the quality gate.")
        print("[LIFECYCLE] Existing Production model will be retained.")

report = {
    "model_name": MODEL_NAME,
    "challenger_version": challenger_version,
    "challenger_recall": float(challenger_recall),
    "previous_production_version": previous_production_version,
    "previous_production_recall": (
        float(production_recall)
        if production_recall is not None else None
    ),
    "promoted_to_production": promoted,
}

report_path = os.path.join(
    RESULTS_DIR,
    "lab6_lifecycle_report.json",
)

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("\n[LIFECYCLE] Model lifecycle processing completed.")
print("[LIFECYCLE] Report saved to:", report_path)