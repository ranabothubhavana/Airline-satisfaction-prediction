import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
import json

# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]
MLOPS_DIR = BASE_DIR / "src" / "mlops"
REPORTS_DIR = BASE_DIR / "reports"

PIPELINE_REPORT = REPORTS_DIR / "lab7_pipeline_report.json"

STAGES = [
    "01_load_data.py",
    "02_validate_data.py",
    "03_preprocessing.py",
    "04_validate_outputs.py",
    "05_train_registry.py",
    "06_model_evaluation.py",
    "07_quality_gate.py",
    "08_automate_lifecycle.py",
    "09_inference.py",
    "10_failure_injection.py",
]


def main():
    print("=" * 60)
    print(" AIRLINE SATISFACTION - MLOPS PIPELINE")
    print("=" * 60)

    results = []
    pipeline_start = datetime.now(timezone.utc)

    for index, stage in enumerate(STAGES, start=1):
        stage_path = MLOPS_DIR / stage

        print(f"\n[{index}/10] Running {stage}...")

        if not stage_path.exists():
            message = f"Stage file not found: {stage_path}"
            print(f"[ERROR] {message}")

            results.append({
                "stage": stage,
                "status": "FAILED",
                "message": message,
            })
            break

        try:
            completed = subprocess.run(
                [sys.executable, str(stage_path)],
                cwd=str(BASE_DIR),
                check=False,
            )

            if completed.returncode != 0:
                message = (
                    f"Stage exited with return code "
                    f"{completed.returncode}"
                )
                print(f"[ERROR] {stage}: {message}")

                results.append({
                    "stage": stage,
                    "status": "FAILED",
                    "return_code": completed.returncode,
                    "message": message,
                })
                break

            print(f"[SUCCESS] {stage} completed.")

            results.append({
                "stage": stage,
                "status": "SUCCESS",
                "return_code": 0,
            })

        except Exception as error:
            print(f"[ERROR] Could not execute {stage}: {error}")

            results.append({
                "stage": stage,
                "status": "FAILED",
                "message": str(error),
            })
            break

    failed = any(item["status"] == "FAILED" for item in results)
    all_stages_completed = len(results) == len(STAGES) and not failed

    pipeline_end = datetime.now(timezone.utc)

    report = {
        "project": "Airline Satisfaction",
        "pipeline": "Lab 7 - End-to-End MLOps Pipeline",
        "started_at_utc": pipeline_start.isoformat(),
        "finished_at_utc": pipeline_end.isoformat(),
        "total_stages": len(STAGES),
        "stages_attempted": len(results),
        "stages_succeeded": sum(
            item["status"] == "SUCCESS" for item in results
        ),
        "status": "SUCCESS" if all_stages_completed else "FAILED",
        "stages": results,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PIPELINE_REPORT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print(" PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Stages attempted : {len(results)}/10")
    print(
        "Stages succeeded : "
        f"{report['stages_succeeded']}"
    )
    print(f"Pipeline status  : {report['status']}")
    print(f"Report saved     : {PIPELINE_REPORT}")

    if not all_stages_completed:
        print("\n[FAILED] Pipeline stopped. Fix the failed stage and rerun.")
        return 1

    print("\n[SUCCESS] All 10 stages completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())