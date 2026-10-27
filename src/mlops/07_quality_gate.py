
import json
import sys
from pathlib import Path

import yaml

# Project root: ml project/
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.yaml"


def main():
    # Step 1: Check configuration file
    if not CONFIG_PATH.exists():
        print(f"[ERROR] Config file not found: {CONFIG_PATH}")
        return 1

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    paths = config["paths"]
    gate_config = config["quality_gate"]

    evaluation_path = BASE_DIR / paths["evaluation_report"]
    quality_report_path = BASE_DIR / paths["quality_gate_report"]

    # Step 2: Check evaluation report
    if not evaluation_path.exists():
        print(f"[ERROR] Evaluation report not found: {evaluation_path}")
        print("[INFO] Run 06_model_evaluation.py first.")
        return 1

    with open(evaluation_path, "r", encoding="utf-8") as file:
        evaluation = json.load(file)

    # Step 3: Read quality gate settings
    metric_name = gate_config["metric"].lower()
    minimum_value = float(gate_config["minimum_value"])

    # Step 4: Get metrics from the nested "metrics" section
    metrics = evaluation.get("metrics", {})

    # Match metric names without case sensitivity
    metric_lookup = {key.lower(): key for key in metrics}

    if metric_name not in metric_lookup:
        print(f"[ERROR] Metric '{metric_name}' not found in evaluation report.")
        print(f"[INFO] Available metrics: {list(metrics.keys())}")
        return 1

    actual_value = float(metrics[metric_lookup[metric_name]])

    # Step 5: Check whether the model passes the quality gate
    passed = actual_value >= minimum_value

    report = {
        "project": config["project"]["name"],
        "metric": metric_name,
        "actual_value": actual_value,
        "minimum_value": minimum_value,
        "status": "PASSED" if passed else "FAILED",
    }

    # Step 6: Save quality gate report
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(quality_report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Step 7: Display results
    print("\n--- Airline Satisfaction: Quality Gate ---")
    print(f"Metric        : {metric_name}")
    print(f"Actual value  : {actual_value:.4f}")
    print(f"Minimum value : {minimum_value:.4f}")
    print(f"Status        : {report['status']}")
    print(f"Report saved  : {quality_report_path}")

    if not passed:
        print("[FAILURE] Model did not meet the quality threshold.")
        return 1

    print("[SUCCESS] Model passed the quality gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


