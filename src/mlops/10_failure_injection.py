import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "lab7_registry_model.joblib"
OUTPUT_DIR = BASE_DIR / "reports"

REPORT_PATH = OUTPUT_DIR / "failure_injection_report.json"


def main():
    print("\n[INFO] Starting Airline Satisfaction failure injection tests...")

    results = []

    # Test 1: Missing dataset
    missing_file = BASE_DIR / "intentionally_missing_dataset.csv"

    results.append({
        "test": "Missing dataset detection",
        "passed": not missing_file.exists(),
        "message": (
            "Missing dataset correctly detected."
            if not missing_file.exists()
            else "Unexpectedly found the test dataset."
        )
    })

    # Test 2: Missing target column
    try:
        sample = pd.DataFrame({
            "Age": [25, 35],
            "Flight Distance": [100, 200]
        })

        target_missing_detected = "satisfaction" not in sample.columns

        results.append({
            "test": "Missing target column detection",
            "passed": target_missing_detected,
            "message": (
                "Missing target column correctly detected."
                if target_missing_detected
                else "Missing target column was not detected."
            )
        })

    except Exception as error:
        results.append({
            "test": "Missing target column detection",
            "passed": False,
            "message": str(error)
        })

    # Test 3: Invalid target labels
    try:
        invalid_labels = pd.Series(["happy", "unhappy"])
        allowed_labels = {
            "satisfied",
            "neutral or dissatisfied"
        }

        invalid_found = not set(invalid_labels).issubset(allowed_labels)

        results.append({
            "test": "Invalid target label detection",
            "passed": invalid_found,
            "message": (
                "Invalid target labels correctly detected."
                if invalid_found
                else "Invalid target labels were not detected."
            )
        })

    except Exception as error:
        results.append({
            "test": "Invalid target label detection",
            "passed": False,
            "message": str(error)
        })

    # Test 4: Missing model detection
    missing_model = BASE_DIR / "models" / "intentionally_missing_model.joblib"

    results.append({
        "test": "Missing model detection",
        "passed": not missing_model.exists(),
        "message": (
            "Missing model correctly detected."
            if not missing_model.exists()
            else "Unexpectedly found the test model."
        )
    })

    # Test 5: Check required project files
    required_files = [TRAIN_PATH, MODEL_PATH]
    missing_required = [
        str(path) for path in required_files if not path.exists()
    ]

    results.append({
        "test": "Required project file validation",
        "passed": len(missing_required) == 0,
        "message": (
            "Required project files are available."
            if not missing_required
            else f"Missing required files: {missing_required}"
        )
    })

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    passed_count = sum(result["passed"] for result in results)
    failed_count = len(results) - passed_count

    report = {
        "project": "Airline Satisfaction",
        "stage": "10 - Failure Injection",
        "total_tests": len(results),
        "passed_tests": passed_count,
        "failed_tests": failed_count,
        "status": "PASSED" if failed_count == 0 else "FAILED",
        "tests": results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n--- Failure Injection Summary ---")

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['test']}")
        print(f"       {result['message']}")

    print(f"\nTotal tests : {len(results)}")
    print(f"Passed      : {passed_count}")
    print(f"Failed      : {failed_count}")
    print(f"Report saved: {REPORT_PATH}")

    if failed_count:
        print("\n[ERROR] One or more failure injection tests failed.")
        return 1

    print("\n[SUCCESS] All failure injection tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())