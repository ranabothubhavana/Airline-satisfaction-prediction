
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "lab7_registry_model.joblib"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_PATH = BASE_DIR / "reports" / "evaluation_report.json"

LABELS = {
    0: "neutral or dissatisfied",
    1: "satisfied",
}


def main():
    print("\n[INFO] Starting Airline Satisfaction Model Evaluation...")

    required_files = [
        MODEL_PATH,
        PROCESSED_DIR / "X_valid.csv",
        PROCESSED_DIR / "y_valid.csv",
    ]

    for path in required_files:
        if not path.exists():
            print(f"[ERROR] Required file not found: {path}")
            print("[INFO] Run the earlier pipeline stages first.")
            return 1

    try:
        model = joblib.load(MODEL_PATH)

        X_valid = pd.read_csv(PROCESSED_DIR / "X_valid.csv")
        y_valid = pd.read_csv(PROCESSED_DIR / "y_valid.csv").iloc[:, 0]

        print("[INFO] Generating validation predictions...")

        predictions = model.predict(X_valid)

        accuracy = accuracy_score(y_valid, predictions)
        precision = precision_score(
            y_valid, predictions, zero_division=0
        )
        recall = recall_score(
            y_valid, predictions, zero_division=0
        )
        f1 = f1_score(
            y_valid, predictions, zero_division=0
        )

        roc_auc = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_valid)[:, 1]
            roc_auc = float(roc_auc_score(y_valid, probabilities))

        matrix = confusion_matrix(
            y_valid, predictions, labels=[0, 1]
        )

        metrics = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": (
                round(roc_auc, 4) if roc_auc is not None else None
            ),
        }

        report = {
            "project": "Airline Satisfaction",
            "stage": "06 - Model Evaluation",
            "model_path": str(MODEL_PATH),
            "evaluated_on": "validation dataset",
            "validation_rows": int(len(X_valid)),
            "metrics": metrics,
            "confusion_matrix": matrix.tolist(),
            "class_labels": {
                str(key): value for key, value in LABELS.items()
            },
            "status": "SUCCESS",
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(REPORT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\n--- Airline Satisfaction: Model Evaluation ---")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1 Score       : {f1:.4f}")

        if roc_auc is not None:
            print(f"Roc Auc        : {roc_auc:.4f}")

        print("Confusion Matrix:")
        print(matrix)
        print(f"\n[SUCCESS] Evaluation report saved to: {REPORT_PATH}")

        return 0

    except Exception as error:
        print(f"[ERROR] Model evaluation failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
