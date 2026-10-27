
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "config.yaml"
QUALITY_REPORT = BASE_DIR / "reports" / "quality_gate_report.json"
LIFECYCLE_REPORT = BASE_DIR / "reports" / "lifecycle_report.json"


def main():
    print("\n[INFO] Starting model lifecycle automation...")

    if not QUALITY_REPORT.exists():
        print(f"[ERROR] Quality gate report not found: {QUALITY_REPORT}")
        print("[INFO] Run 07_quality_gate.py first.")
        return 1

    try:
        with open(QUALITY_REPORT, "r", encoding="utf-8") as file:
            quality_data = json.load(file)

        # Support the report's status field and common alternatives.
        status = str(
            quality_data.get("status", quality_data.get("result", ""))
        ).upper()

        passed = status in {"PASSED", "PASS", "SUCCESS"}

        if not passed:
            report = {
                "project": "Airline Satisfaction",
                "stage": "08 - Automate Lifecycle",
                "quality_gate_status": status or "UNKNOWN",
                "lifecycle_action": "BLOCKED",
                "message": "Model lifecycle blocked because the quality gate did not pass."
            }

            LIFECYCLE_REPORT.parent.mkdir(parents=True, exist_ok=True)
            with open(LIFECYCLE_REPORT, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=4)

            print(f"[BLOCKED] Quality gate status: {status or 'UNKNOWN'}")
            print(f"Report saved: {LIFECYCLE_REPORT}")
            return 1

        report = {
            "project": "Airline Satisfaction",
            "stage": "08 - Automate Lifecycle",
            "quality_gate_status": status,
            "lifecycle_action": "APPROVED",
            "message": (
                "Quality gate passed. Model is approved to proceed "
                "to the next lifecycle stage."
            ),
            "note": (
                "This script records approval; it does not change the "
                "MLflow registry stage or promote a model version."
            )
        }

        LIFECYCLE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(LIFECYCLE_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\n--- Model Lifecycle Summary ---")
        print(f"Quality gate : {status}")
        print("Action       : APPROVED")
        print(f"Report saved : {LIFECYCLE_REPORT}")
        print("\n[SUCCESS] Lifecycle approval completed.")

        return 0

    except Exception as error:
        print(f"[ERROR] Lifecycle automation failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
