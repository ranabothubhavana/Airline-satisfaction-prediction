
import json
import sys
from pathlib import Path

import pandas as pd

# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

REPORT_PATH = ARTIFACTS_DIR / "output_validation_report.json"

REQUIRED_FILES = {
    "X_train": PROCESSED_DIR / "X_train.csv",
    "X_valid": PROCESSED_DIR / "X_valid.csv",
    "y_train": PROCESSED_DIR / "y_train.csv",
    "y_valid": PROCESSED_DIR / "y_valid.csv",
}


def main():
    print("\n[INFO] Starting Airline Satisfaction output validation...")

    errors = []
    warnings = []

    # 1. Check whether required files exist
    for name, path in REQUIRED_FILES.items():
        if not path.exists():
            errors.append(f"Missing required file: {path}")

    if errors:
        print("\n[ERROR] Output validation failed.")
        for error in errors:
            print(f" - {error}")
        return 1

    try:
        # 2. Load processed files
        X_train = pd.read_csv(REQUIRED_FILES["X_train"])
        X_valid = pd.read_csv(REQUIRED_FILES["X_valid"])
        y_train = pd.read_csv(REQUIRED_FILES["y_train"])
        y_valid = pd.read_csv(REQUIRED_FILES["y_valid"])

        # 3. Check for empty datasets
        datasets = {
            "X_train": X_train,
            "X_valid": X_valid,
            "y_train": y_train,
            "y_valid": y_valid,
        }

        for name, dataset in datasets.items():
            if dataset.empty:
                errors.append(f"{name} is empty.")

        # 4. Check matching row counts
        if len(X_train) != len(y_train):
            errors.append("Training features and labels have different row counts.")

        if len(X_valid) != len(y_valid):
            errors.append("Validation features and labels have different row counts.")

        # 5. Check matching feature columns
        if list(X_train.columns) != list(X_valid.columns):
            errors.append("Training and validation feature columns do not match.")

        # 6. Check missing values
        missing_counts = {
            name: int(dataset.isnull().sum().sum())
            for name, dataset in datasets.items()
        }

        for name, count in missing_counts.items():
            if count > 0:
                warnings.append(
                    f"{name} contains {count} missing values."
                )

        # 7. Check target labels
        target_values = sorted(
            set(y_train.iloc[:, 0].dropna().unique().tolist())
            | set(y_valid.iloc[:, 0].dropna().unique().tolist())
        )

        if not set(target_values).issubset({0, 1}):
            errors.append(
                f"Unexpected target values: {target_values}. Expected 0 and 1."
            )

        if y_train.isnull().any().any() or y_valid.isnull().any().any():
            errors.append("Target labels contain missing values.")

        # 8. Build report
        status = "FAILED" if errors else "PASSED"

        report = {
            "project": "Airline Satisfaction",
            "stage": "04 - Validate Outputs",
            "status": status,
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
            "training_features": int(X_train.shape[1]),
            "validation_features": int(X_valid.shape[1]),
            "training_label_rows": int(len(y_train)),
            "validation_label_rows": int(len(y_valid)),
            "missing_values": missing_counts,
            "target_values": [int(value) for value in target_values],
            "errors": errors,
            "warnings": warnings,
        }

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(REPORT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        # 9. Display results
        print("\n--- Output Validation Summary ---")
        print(f"Training rows      : {len(X_train)}")
        print(f"Validation rows    : {len(X_valid)}")
        print(f"Training features  : {X_train.shape[1]}")
        print(f"Validation features: {X_valid.shape[1]}")
        print(f"Target values      : {target_values}")
        print(f"Status             : {status}")

        for warning in warnings:
            print(f"[WARNING] {warning}")

        for error in errors:
            print(f"[ERROR] {error}")

        print(f"\nReport saved to: {REPORT_PATH}")

        if errors:
            print("[FAILED] Output validation found errors.")
            return 1

        print("[SUCCESS] Output validation completed.")
        return 0

    except Exception as error:
        print(f"[ERROR] Validation failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())