import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_PATH = os.path.join(BASE_DIR, "test.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[PREPROCESS_PIPELINE] Loading production/test data:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

target = "satisfaction"
ids = df["id"].copy() if "id" in df.columns else pd.Series(np.arange(len(df)), name="id")
X = df.drop(columns=[target, "id"], errors="ignore")

preprocessor_path = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
model_path = os.path.join(MODEL_DIR, "best_model.joblib")

if not os.path.exists(preprocessor_path):
    raise FileNotFoundError("Lab 3 preprocessor artifact is missing. Run Lab 3 first.")
if not os.path.exists(model_path):
    raise FileNotFoundError("Lab 3 model is missing. Run Lab 3 first.")

preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)

# Transform new data using the already-fitted training preprocessor.
X_transformed = preprocessor.transform(X)

# Save a dense CSV only when it is reasonably sized; otherwise save a joblib artifact.
if hasattr(X_transformed, "toarray"):
    transformed_for_file = X_transformed.toarray()
else:
    transformed_for_file = np.asarray(X_transformed)

np.save(os.path.join(OUTPUT_DIR, "lab5_transformed_features.npy"), transformed_for_file)
joblib.dump(X_transformed, os.path.join(OUTPUT_DIR, "lab5_transformed_features.joblib"))

predictions = model.predict(X)
result = pd.DataFrame({
    "id": ids,
    "prediction": predictions.astype(int)
})

if hasattr(model, "predict_proba"):
    result["satisfaction_probability"] = model.predict_proba(X)[:, 1]

result.to_csv(os.path.join(OUTPUT_DIR, "lab5_production_predictions.csv"), index=False)

metadata = {
    "input": "test.csv",
    "rows_processed": int(len(df)),
    "transformed_feature_count": int(X_transformed.shape[1]),
    "model": "models/best_model.joblib",
    "preprocessor": "artifacts/preprocessor.joblib",
}
with open(os.path.join(OUTPUT_DIR, "lab5_pipeline_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("[PREPROCESS_PIPELINE] Production preprocessing complete.")
print("[PREPROCESS_PIPELINE] Rows processed:", len(df))
print("[PREPROCESS_PIPELINE] Transformed shape:", X_transformed.shape)
print("[PREPROCESS_PIPELINE] Outputs saved to:", OUTPUT_DIR)
