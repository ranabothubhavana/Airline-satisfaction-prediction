
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "train.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

REPORT_PATH = ARTIFACTS_DIR / "preprocessing_report.json"

TARGET = "satisfaction"


def main():
    print("\n[INFO] Starting Airline Satisfaction preprocessing...")

    if not TRAIN_PATH.exists():
        print(f"[ERROR] Training dataset not found: {TRAIN_PATH}")
        return 1

    try:
        # 1. Load the training dataset
        df = pd.read_csv(TRAIN_PATH)

        # 2. Check target column
        if TARGET not in df.columns:
            print(f"[ERROR] Target column '{TARGET}' not found.")
            return 1

        # 3. Remove rows with missing target values
        original_rows = len(df)
        df = df.dropna(subset=[TARGET]).copy()
        removed_rows = original_rows - len(df)

        # 4. Separate features and target
        X = df.drop(columns=[TARGET])
        y = df[TARGET].copy()

        # 5. Convert categorical target labels to numbers
        label_mapping = {
            "neutral or dissatisfied": 0,
            "satisfied": 1
        }

        y = y.astype(str).str.strip().str.lower()

        unexpected_labels = set(y.unique()) - set(label_mapping.keys())

        if unexpected_labels:
            print(f"[ERROR] Unexpected target labels: {unexpected_labels}")
            return 1

        y = y.map(label_mapping).astype(int)

        # 6. Split into training and validation datasets
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # 7. Save the split datasets
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
        X_valid.to_csv(PROCESSED_DIR / "X_valid.csv", index=False)
        y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
        y_valid.to_csv(PROCESSED_DIR / "y_valid.csv", index=False)

        # 8. Save preprocessing report
        report = {
            "project": "Airline Satisfaction",
            "stage": "03 - Preprocessing",
            "original_rows": int(original_rows),
            "rows_after_target_cleanup": int(len(df)),
            "rows_removed": int(removed_rows),
            "feature_count": int(X.shape[1]),
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
            "target": TARGET,
            "label_mapping": label_mapping,
            "random_state": 42,
            "validation_size": 0.20,
            "status": "SUCCESS"
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\n--- Preprocessing Summary ---")
        print(f"Original rows      : {original_rows}")
        print(f"Rows after cleanup : {len(df)}")
        print(f"Features           : {X.shape[1]}")
        print(f"Training rows      : {len(X_train)}")
        print(f"Validation rows    : {len(X_valid)}")
        print(f"Removed rows       : {removed_rows}")
        print(f"Report saved       : {REPORT_PATH}")
        print(f"Processed data     : {PROCESSED_DIR}")
        print("\n[SUCCESS] Preprocessing completed.")

        return 0

    except Exception as error:
        print(f"[ERROR] Preprocessing failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())