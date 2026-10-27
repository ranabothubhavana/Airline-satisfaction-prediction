import os
import sys
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

prediction_path = os.path.join(OUTPUT_DIR, "lab5_production_predictions.csv")
features_path = os.path.join(OUTPUT_DIR, "lab5_transformed_features.npy")
metadata_path = os.path.join(OUTPUT_DIR, "lab5_pipeline_metadata.json")

print("[VALIDATE_OUTPUTS] Checking Lab 5 production outputs...")

required = [prediction_path, features_path, metadata_path]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    print("[VALIDATE_OUTPUTS] Missing output files:", missing)
    sys.exit(1)

pred = pd.read_csv(prediction_path)
source = pd.read_csv(TEST_PATH).drop(columns=["Unnamed: 0"], errors="ignore")
features = np.load(features_path)

errors = []

if len(pred) != len(source):
    errors.append(f"Prediction row count {len(pred)} != input row count {len(source)}")

if pred["prediction"].isna().any():
    errors.append("Prediction output contains missing predictions")

if not set(pred["prediction"].unique()).issubset({0, 1}):
    errors.append("Predictions must be 0 or 1")

if features.shape[0] != len(source):
    errors.append(f"Transformed feature rows {features.shape[0]} != input rows {len(source)}")

with open(metadata_path, encoding="utf-8") as f:
    metadata = json.load(f)

if int(metadata.get("rows_processed", -1)) != len(source):
    errors.append("Metadata row count does not match input row count")

summary = {
    "input_rows": len(source),
    "prediction_rows": len(pred),
    "transformed_rows": int(features.shape[0]),
    "transformed_columns": int(features.shape[1]),
    "output_validation": "PASS" if not errors else "FAIL",
}
pd.DataFrame([summary]).to_csv(
    os.path.join(RESULTS_DIR, "lab5_output_validation_summary.csv"), index=False
)

if errors:
    with open(os.path.join(RESULTS_DIR, "lab5_output_validation_errors.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(errors))
    for error in errors:
        print("[VALIDATE_OUTPUTS]", error)
    sys.exit(1)

print("[VALIDATE_OUTPUTS] PASS: all Lab 5 production outputs are valid.")
print("[VALIDATE_OUTPUTS] Summary saved to:", RESULTS_DIR)
