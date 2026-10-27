import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "test.csv")

model = joblib.load(MODEL_PATH)
test = pd.read_csv(TEST_PATH)

for col in ["Unnamed: 0"]:
    if col in test.columns:
        test = test.drop(columns=[col])

# Use the first unseen row as a sample prediction.
# The test file's satisfaction column is NOT passed to the model.
target = "satisfaction"
X_new = test.drop(columns=[target, "id"], errors="ignore")

prediction = model.predict(X_new)[0]
probability = model.predict_proba(X_new)[0, 1]

label = "Satisfied" if prediction == 1 else "Neutral or Dissatisfied"

print("Sample unseen passenger prediction")
print("-----------------------------------")
print("Predicted class:", label)
print(f"Probability of satisfaction: {probability:.4f}")

# Predict all test rows and save
pred = model.predict(X_new)
probs = model.predict_proba(X_new)[:, 1]

output = test.copy()
output["Predicted_Satisfaction"] = ["Satisfied" if p == 1 else "Neutral or Dissatisfied" for p in pred]
output["Satisfaction_Probability"] = probs

out_path = os.path.join(BASE_DIR, "results", "test_predictions.csv")
output.to_csv(out_path, index=False)

print("\nAll test predictions saved to:", out_path)
