
import json
import sys
from pathlib import Path

import pandas as pd

# Project root: ml project/
BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORT_PATH = ARTIFACTS_DIR / "data_load_report.json"


def main():
    print("[INFO] Starting Airline Satisfaction data loading...")

    # Check dataset files
    missing_files = [
        path for path in [TRAIN_PATH, TEST_PATH]
        if not path.exists()
    ]

    if missing_files:
        for path in missing_files:
            print(f"[ERROR] Dataset file not found: {path}")
        return 1

    try:
        # Load datasets
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)

        # Validate that datasets are not empty
        if train_df.empty or test_df.empty:
            print("[ERROR] Training or test dataset is empty.")
            return 1

        # Create artifact directory
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate data loading report
        report = {
            "project": "Airline Satisfaction",
            "target": "satisfaction",
            "train_file": str(TRAIN_PATH.relative_to(BASE_DIR)),
            "test_file": str(TEST_PATH.relative_to(BASE_DIR)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_columns": int(len(train_df.columns)),
            "test_columns": int(len(test_df.columns)),
            "train_column_names": train_df.columns.tolist(),
            "test_column_names": test_df.columns.tolist(),
            "train_missing_values": int(train_df.isnull().sum().sum()),
            "test_missing_values": int(test_df.isnull().sum().sum()),
            "status": "SUCCESS",
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\n--- Airline Satisfaction: Data Loading Report ---")
        print(f"Training rows       : {len(train_df)}")
        print(f"Test rows           : {len(test_df)}")
        print(f"Training columns    : {len(train_df.columns)}")
        print(f"Test columns        : {len(test_df.columns)}")
        print(f"Training missing    : {report['train_missing_values']}")
        print(f"Test missing        : {report['test_missing_values']}")
        print(f"\n[SUCCESS] Report saved to: {REPORT_PATH}")

        return 0

    except Exception as error:
        print(f"[ERROR] Data loading failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
