import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("[TRAIN] Loading preprocessing artifacts...")
X_train = joblib.load(os.path.join(ARTIFACT_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(ARTIFACT_DIR, "y_train.joblib"))
preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))

# Lab 3 baseline model: Logistic Regression.
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
])

print("[TRAIN] Training baseline Logistic Regression...")
model.fit(X_train, y_train)

model_path = os.path.join(MODEL_DIR, "lab3_baseline_model.joblib")
joblib.dump(model, model_path)
# Keep the existing application compatible with the Lab 3-trained model.
joblib.dump(model, os.path.join(MODEL_DIR, "best_model.joblib"))

metadata = {
    "stage": "Lab 3 baseline",
    "model": "Logistic Regression",
    "random_state": 42,
    "target": "satisfaction",
    "label_mapping": {"neutral or dissatisfied": 0, "satisfied": 1},
}
joblib.dump(metadata, os.path.join(MODEL_DIR, "lab3_baseline_metadata.joblib"))

print("[TRAIN] Model saved to:", model_path)
