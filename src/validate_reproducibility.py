import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("[REPRO] Loading the fixed Lab 3 train/validation split...")
X_train = joblib.load(os.path.join(ARTIFACT_DIR, "X_train.joblib"))
y_train = joblib.load(os.path.join(ARTIFACT_DIR, "y_train.joblib"))
X_valid = joblib.load(os.path.join(ARTIFACT_DIR, "X_valid.joblib"))
y_valid = joblib.load(os.path.join(ARTIFACT_DIR, "y_valid.joblib"))
preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))


def train_and_predict():
    # Re-create the same pipeline and fixed random state used in Lab 4.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    prob = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_valid, pred),
        "precision": precision_score(y_valid, pred, zero_division=0),
        "recall": recall_score(y_valid, pred, zero_division=0),
        "f1": f1_score(y_valid, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_valid, prob),
    }
    return model, pred, prob, metrics

print("[REPRO] Training the same pipeline twice...")
model_1, pred_1, prob_1, metrics_1 = train_and_predict()
model_2, pred_2, prob_2, metrics_2 = train_and_predict()

metric_names = list(metrics_1.keys())
metrics_match = all(np.isclose(metrics_1[k], metrics_2[k], rtol=0, atol=1e-12) for k in metric_names)
predictions_match = np.array_equal(pred_1, pred_2)
probabilities_match = np.allclose(prob_1, prob_2, rtol=0, atol=1e-12)

comparison = pd.DataFrame([
    {"metric": k, "run_1": metrics_1[k], "run_2": metrics_2[k], "match": np.isclose(metrics_1[k], metrics_2[k], rtol=0, atol=1e-12)}
    for k in metric_names
])
comparison.to_csv(os.path.join(RESULTS_DIR, "lab4_reproducibility_metrics.csv"), index=False)

status = metrics_match and predictions_match and probabilities_match
with open(os.path.join(RESULTS_DIR, "lab4_reproducibility_report.txt"), "w", encoding="utf-8") as f:
    f.write("Lab 4 Reproducibility Validation\n")
    f.write("================================\n")
    f.write(f"Metrics identical: {metrics_match}\n")
    f.write(f"Predictions identical: {predictions_match}\n")
    f.write(f"Probabilities identical: {probabilities_match}\n")
    f.write(f"Overall reproducible: {status}\n")

if not status:
    raise RuntimeError("Reproducibility validation failed: the two runs did not produce identical results.")

print("[REPRO] Metrics identical: True")
print("[REPRO] Predictions identical: True")
print("[REPRO] Probabilities identical: True")
print("[REPRO] Overall reproducible: True")
print("[REPRO] Reproducibility results saved to:", RESULTS_DIR)
