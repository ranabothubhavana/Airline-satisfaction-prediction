import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("[EVALUATE] Loading validation data and model...")
X_valid = joblib.load(os.path.join(ARTIFACT_DIR, "X_valid.joblib"))
y_valid = joblib.load(os.path.join(ARTIFACT_DIR, "y_valid.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "lab3_baseline_model.joblib"))

pred = model.predict(X_valid)
prob = model.predict_proba(X_valid)[:, 1]

metrics = {
    "Model": "Logistic Regression (Lab 3 Baseline)",
    "Accuracy": accuracy_score(y_valid, pred),
    "Precision": precision_score(y_valid, pred, zero_division=0),
    "Recall": recall_score(y_valid, pred, zero_division=0),
    "F1": f1_score(y_valid, pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_valid, prob),
}

pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, "lab3_baseline_results.csv"), index=False)

report = classification_report(
    y_valid, pred,
    target_names=["Neutral/Dissatisfied", "Satisfied"],
    digits=4,
    zero_division=0,
)
with open(os.path.join(RESULTS_DIR, "lab3_baseline_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
    f.write("\n\nMetrics:\n")
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            f.write(f"{key}: {value:.4f}\n")
        else:
            f.write(f"{key}: {value}\n")

cm = confusion_matrix(y_valid, pred)
pd.DataFrame(
    cm,
    index=["Actual Neutral/Dissatisfied", "Actual Satisfied"],
    columns=["Predicted Neutral/Dissatisfied", "Predicted Satisfied"],
).to_csv(os.path.join(RESULTS_DIR, "lab3_confusion_matrix.csv"))

# Save row-level validation predictions for inspection.
predictions = X_valid.copy()
predictions["Actual_Satisfaction"] = ["Satisfied" if x == 1 else "Neutral or Dissatisfied" for x in y_valid]
predictions["Predicted_Satisfaction"] = ["Satisfied" if x == 1 else "Neutral or Dissatisfied" for x in pred]
predictions["Satisfaction_Probability"] = prob
predictions.to_csv(os.path.join(RESULTS_DIR, "lab3_validation_predictions.csv"), index=False)

print("[EVALUATE] Baseline evaluation complete.")
for key, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"[EVALUATE] {key}: {value:.4f}")
    else:
        print(f"[EVALUATE] {key}: {value}")
print("[EVALUATE] Results saved to:", RESULTS_DIR)
