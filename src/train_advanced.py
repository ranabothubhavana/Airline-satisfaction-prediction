import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
df = pd.read_csv(TRAIN_PATH)

# Remove accidental index column if present
for col in ["Unnamed: 0"]:
    if col in df.columns:
        df = df.drop(columns=[col])

TARGET = "satisfaction"
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")

# Basic cleaning
df = df.drop_duplicates().copy()

# Clean target text
df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

# Save a compact dataset report
report = {
    "rows_after_duplicate_removal": len(df),
    "columns": df.shape[1],
    "duplicate_rows_removed": int(pd.read_csv(TRAIN_PATH).duplicated().sum()),
    "missing_values_total": int(df.isna().sum().sum()),
    "target_distribution": df[TARGET].value_counts(dropna=False).to_dict(),
}
pd.DataFrame([report]).to_csv(os.path.join(RESULTS_DIR, "dataset_summary.csv"), index=False)

# -------------------- EDA --------------------
plt.figure(figsize=(7, 5))
df[TARGET].value_counts().plot(kind="bar")
plt.title("Passenger Satisfaction Distribution")
plt.xlabel("Satisfaction")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "target_distribution.png"), dpi=150)
plt.close()

missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if len(missing):
    plt.figure(figsize=(9, 5))
    missing.plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "missing_values.png"), dpi=150)
    plt.close()

numeric_all = df.select_dtypes(include=np.number)
if not numeric_all.empty:
    corr = numeric_all.corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.title("Numerical Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_matrix.png"), dpi=150)
    plt.close()

# -------------------- Feature engineering --------------------
# Keep Passenger ID out of the predictive features because it is an identifier.
drop_cols = [TARGET]
if "id" in df.columns:
    drop_cols.append("id")
if "Unnamed: 0" in df.columns:
    drop_cols.append("Unnamed: 0")

X = df.drop(columns=drop_cols)
y = df[TARGET].map({"satisfied": 1, "neutral or dissatisfied": 0})

if y.isna().any():
    raise ValueError("Unexpected target values found. Check the satisfaction column.")

# Convert low-cardinality categorical columns and preserve numeric columns.
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = []
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_valid)
    prob = pipe.predict_proba(X_valid)[:, 1] if hasattr(pipe, "predict_proba") else None

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_valid, pred),
        "Precision": precision_score(y_valid, pred, zero_division=0),
        "Recall": recall_score(y_valid, pred, zero_division=0),
        "F1": f1_score(y_valid, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_valid, prob) if prob is not None else np.nan
    })
    trained_pipelines[name] = pipe

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

# Plot comparison
plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
plot_df.plot(kind="bar", figsize=(12, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150)
plt.close()

# -------------------- Cross-validation on top candidates --------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

for name in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
    pipe = trained_pipelines[name]
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    cv_rows.append({
        "Model": name,
        "CV_F1_Mean": scores.mean(),
        "CV_F1_STD": scores.std()
    })

cv_df = pd.DataFrame(cv_rows).sort_values("CV_F1_Mean", ascending=False)
cv_df.to_csv(os.path.join(RESULTS_DIR, "cross_validation_results.csv"), index=False)

# -------------------- Hyperparameter tuning --------------------
# Tune Random Forest and Gradient Boosting; these are strong candidates for this tabular dataset.
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))
])

rf_grid = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [None, 15],
    "model__min_samples_split": [2, 5]
}

rf_search = GridSearchCV(
    rf_pipe, rf_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
)
rf_search.fit(X_train, y_train)

gb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(random_state=42))
])

gb_grid = {
    "model__n_estimators": [100, 150],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3]
}

gb_search = GridSearchCV(
    gb_pipe, gb_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
)
gb_search.fit(X_train, y_train)

tuned_models = {
    "Tuned Random Forest": rf_search.best_estimator_,
    "Tuned Gradient Boosting": gb_search.best_estimator_
}

tuning_rows = []
for name, pipe in tuned_models.items():
    pred = pipe.predict(X_valid)
    prob = pipe.predict_proba(X_valid)[:, 1]
    tuning_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_valid, pred),
        "Precision": precision_score(y_valid, pred, zero_division=0),
        "Recall": recall_score(y_valid, pred, zero_division=0),
        "F1": f1_score(y_valid, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_valid, prob)
    })

tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv(os.path.join(RESULTS_DIR, "tuned_model_results.csv"), index=False)

all_results = pd.concat([results_df, tuning_df], ignore_index=True)
all_results = all_results.sort_values("F1", ascending=False)
all_results.to_csv(os.path.join(RESULTS_DIR, "all_model_results.csv"), index=False)

best_name = all_results.iloc[0]["Model"]
if best_name == "Tuned Random Forest":
    best_model = rf_search.best_estimator_
elif best_name == "Tuned Gradient Boosting":
    best_model = gb_search.best_estimator_
else:
    best_model = trained_pipelines[best_name]

# Validation report and confusion matrix
best_pred = best_model.predict(X_valid)
best_prob = best_model.predict_proba(X_valid)[:, 1]
cm = confusion_matrix(y_valid, best_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for (i, j), value in np.ndenumerate(cm):
    plt.text(j, i, value, ha="center", va="center")
plt.xticks([0, 1], ["Neutral/Dissatisfied", "Satisfied"], rotation=15)
plt.yticks([0, 1], ["Neutral/Dissatisfied", "Satisfied"])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "best_model_confusion_matrix.png"), dpi=150)
plt.close()

with open(os.path.join(RESULTS_DIR, "best_model_classification_report.txt"), "w") as f:
    f.write(classification_report(
        y_valid, best_pred,
        target_names=["Neutral/Dissatisfied", "Satisfied"],
        digits=4
    ))
    f.write("\nBest model: " + best_name)
    f.write("\nBest parameters, if tuned: " + str(
        rf_search.best_params_ if best_name == "Tuned Random Forest"
        else gb_search.best_params_ if best_name == "Tuned Gradient Boosting"
        else "N/A"
    ))

# Save model and metadata
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))

metadata = {
    "target": TARGET,
    "feature_columns": X.columns.tolist(),
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "best_model": best_name,
    "label_mapping": {"neutral or dissatisfied": 0, "satisfied": 1}
}
joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.joblib"))

print("\nTraining complete.")
print("Best model:", best_name)
print(all_results.to_string(index=False))
print("\nSaved:", os.path.join(MODEL_DIR, "best_model.joblib"))
