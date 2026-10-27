
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOAD_REPORT_PATH = ARTIFACTS_DIR / "data_load_report.json"
VALIDATION_REPORT_PATH = ARTIFACTS_DIR / "data_validation_report.json"

TARGET = "satisfaction"


def main():
    print("[INFO] Starting Airline Satisfaction data validation...")

    for path in [TRAIN_PATH, TEST_PATH, LOAD_REPORT_PATH]:
        if not path.exists():
            print(f"[ERROR] Required file not found: {path}")
            print("[INFO] Run 01_load_data.py first.")
            return 1

    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)

        errors = []
        warnings = []

        # Check required target column
        if TARGET not in train_df.columns:
            errors.append(f"Training data is missing target column: {TARGET}")

        # Test data may not contain the target column
        if TARGET in test_df.columns:
            warnings.append(
                "Test data contains the target column; verify its intended use."
            )

        # Check empty datasets
        if train_df.empty:
            errors.append("Training dataset is empty.")

        if test_df.empty:
            errors.append("Test dataset is empty.")

        # Check duplicate rows
        train_duplicates = int(train_df.duplicated().sum())
        test_duplicates = int(test_df.duplicated().sum())

        if train_duplicates:
            warnings.append(
                f"Training dataset contains {train_duplicates} duplicate rows."
            )

        if test_duplicates:
            warnings.append(
                f"Test dataset contains {test_duplicates} duplicate rows."
            )

        # Check missing values
        train_missing = int(train_df.isnull().sum().sum())
        test_missing = int(test_df.isnull().sum().sum())

        if train_missing:
            warnings.append(
                f"Training dataset contains {train_missing} missing values."
            )

        if test_missing:
            warnings.append(
                f"Test dataset contains {test_missing} missing values."
            )

        # Check target labels
        target_labels = []

        if TARGET in train_df.columns:
            target_labels = sorted(
                train_df[TARGET].dropna().astype(str).unique().tolist()
            )

            if train_df[TARGET].isnull().any():
                errors.append("Training target contains missing values.")

            expected_labels = {
                "satisfied",
                "neutral or dissatisfied",
            }

            unexpected_labels = set(target_labels) - expected_labels

            if unexpected_labels:
                errors.append(
                    f"Unexpected target labels found: {sorted(unexpected_labels)}"
                )

        # Check train/test feature compatibility
        train_features = set(train_df.columns) - {TARGET}
        test_features = set(test_df.columns) - {TARGET}

        missing_test_features = sorted(train_features - test_features)
        extra_test_features = sorted(test_features - train_features)

        if missing_test_features:
            errors.append(
                f"Features missing from test data: {missing_test_features}"
            )

        if extra_test_features:
            warnings.append(
                f"Extra features in test data: {extra_test_features}"
            )

        status = "FAILED" if errors else "PASSED"

        report = {
            "project": "Airline Satisfaction",
            "target": TARGET,
            "status": status,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_columns": int(len(train_df.columns)),
            "test_columns": int(len(test_df.columns)),
            "train_missing_values": train_missing,
            "test_missing_values": test_missing,
            "train_duplicate_rows": train_duplicates,
            "test_duplicate_rows": test_duplicates,
            "target_labels": target_labels,
            "errors": errors,
            "warnings": warnings,
        }

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(
            VALIDATION_REPORT_PATH, "w", encoding="utf-8"
        ) as file:
            json.dump(report, file, indent=4)

        print("\n--- Airline Satisfaction: Data Validation ---")
        print(f"Status             : {status}")
        print(f"Training rows      : {len(train_df)}")
        print(f"Test rows          : {len(test_df)}")
        print(f"Training missing   : {train_missing}")
        print(f"Test missing       : {test_missing}")
        print(f"Training duplicates: {train_duplicates}")
        print(f"Test duplicates    : {test_duplicates}")

        for warning in warnings:
            print(f"[WARNING] {warning}")

        for error in errors:
            print(f"[ERROR] {error}")

        print(f"\nReport saved to: {VALIDATION_REPORT_PATH}")

        if errors:
            return 1

        print("[SUCCESS] Data validation completed.")
        return 0

    except Exception as error:
        print(f"[ERROR] Data validation failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
