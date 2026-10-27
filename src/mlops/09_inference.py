
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "lab7_registry_model.joblib"
TEST_PATH = BASE_DIR / "test.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
PREDICTIONS_PATH = OUTPUT_DIR / "airline_predictions.csv"
REPORT_PATH = BASE_DIR / "reports" / "inference_report.json"

TARGET = "satisfaction"

LABEL_MAPPING = {
    0: "neutral or dissatisfied",
    1: "satisfied"
}


def main():
    print("\n[INFO] Starting Airline Satisfaction inference...")

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("[INFO] Run 05_train_registry.py first.")
        return 1

    if not TEST_PATH.exists():
        print(f"[ERROR] Test dataset not found: {TEST_PATH}")
        return 1

    try:
        # Load model and test data
        model = joblib.load(MODEL_PATH)
        test_df = pd.read_csv(TEST_PATH)

        # Remove the target column if it exists.
        # Inference must not use the true answer as an input feature.
        X_test = test_df.drop(columns=[TARGET], errors="ignore")

        # Generate predictions
        predictions = model.predict(X_test)

        # Convert predictions into readable labels
        predicted_labels = [
            LABEL_MAPPING.get(int(prediction), str(prediction))
            for prediction in predictions
        ]

        # Save predictions
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        result_df = pd.DataFrame({
            "predicted_label": predicted_labels,
            "predicted_class": predictions
        })

        result_df.to_csv(PREDICTIONS_PATH, index=False)

        # Count predictions
        prediction_counts = pd.Series(predicted_labels).value_counts().to_dict()

        report = {
            "project": "Airline Satisfaction",
            "stage": "09 - Inference",
            "model_path": str(MODEL_PATH),
            "input_file": str(TEST_PATH),
            "input_rows": int(len(X_test)),
            "prediction_count": int(len(predictions)),
            "prediction_distribution": {
                str(label): int(count)
                for label, count in prediction_counts.items()
            },
            "predictions_file": str(PREDICTIONS_PATH),
            "status": "SUCCESS"
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\n--- Inference Summary ---")
        print(f"Input rows        : {len(X_test)}")
        print(f"Predictions       : {len(predictions)}")

        for label, count in prediction_counts.items():
            print(f"{label:<25}: {count}")

        print(f"\nPredictions saved: {PREDICTIONS_PATH}")
        print(f"Report saved     : {REPORT_PATH}")
        print("\n[SUCCESS] Inference completed.")

        return 0

    except Exception as error:
        print(f"[ERROR] Inference failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())