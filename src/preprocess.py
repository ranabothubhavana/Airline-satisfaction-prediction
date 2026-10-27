import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TARGET = "satisfaction"

print("[PREPROCESS] Loading dataset:", TRAIN_PATH)

df = pd.read_csv(TRAIN_PATH)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates().copy()
df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

label_map = {
    "neutral or dissatisfied": 0,
    "satisfied": 1
}

y = df[TARGET].map(label_map)

if y.isna().any():
    bad = df.loc[y.isna(), TARGET].unique().tolist()
    raise ValueError(f"Unexpected target values: {bad}")

X = df.drop(columns=[TARGET, "id"], errors="ignore")

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_cols = X.select_dtypes(
    include="number"
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save raw split data
for name, obj in {
    "X_train": X_train,
    "X_valid": X_valid,
    "y_train": y_train,
    "y_valid": y_valid,
}.items():
    joblib.dump(
        obj,
        os.path.join(ARTIFACT_DIR, f"{name}.joblib")
    )

# IMPORTANT: Fit the preprocessor before saving it.
# Only the training data is used to fit the transformations.
print("[PREPROCESS] Fitting preprocessor on training data...")
preprocessor.fit(X_train)

# Save the fitted preprocessor
joblib.dump(
    preprocessor,
    os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
)

# Save preprocessing metadata
joblib.dump({
    "target": TARGET,
    "label_mapping": label_map,
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "feature_columns": X.columns.tolist(),
}, os.path.join(ARTIFACT_DIR, "preprocessing_metadata.joblib"))

# Save preprocessing summary
summary = pd.DataFrame([{
    "rows_after_cleaning": len(df),
    "features_used": X.shape[1],
    "training_rows": len(X_train),
    "validation_rows": len(X_valid),
    "missing_values_before_imputation": int(X.isna().sum().sum()),
    "duplicates_removed": int(
        pd.read_csv(TRAIN_PATH).duplicated().sum()
    ),
    "satisfied": int((y == 1).sum()),
    "neutral_or_dissatisfied": int((y == 0).sum()),
}])

summary.to_csv(
    os.path.join(RESULTS_DIR, "lab3_preprocessing_summary.csv"),
    index=False
)

print("[PREPROCESS] Cleaning complete.")
print(f"[PREPROCESS] Train: {X_train.shape}, Validation: {X_valid.shape}")
print("[PREPROCESS] Fitted preprocessor saved successfully.")
print("[PREPROCESS] Artifacts saved to:", ARTIFACT_DIR)