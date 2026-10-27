
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "train.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = MODELS_DIR / "lab7_registry_model.joblib"
REPORT_PATH = REPORTS_DIR / "lab7_training_report.json"

TARGET = "satisfaction"


def main():
    print("\n[INFO] Starting Airline Satisfaction model training...")

    if not TRAIN_PATH.exists():
        print(f"[ERROR] Dataset not found: {TRAIN_PATH}")
        return 1

    try:
        # 1. Load data
        df = pd.read_csv(TRAIN_PATH)

        if TARGET not in df.columns:
            print(f"[ERROR] Target column '{TARGET}' not found.")
            return 1

        df = df.dropna(subset=[TARGET]).copy()

        X = df.drop(columns=[TARGET])
        y = df[TARGET].astype(str).str.strip().str.lower()

        label_mapping = {
            "neutral or dissatisfied": 0,
            "satisfied": 1
        }

        unexpected = set(y.unique()) - set(label_mapping)

        if unexpected:
            print(f"[ERROR] Unexpected target labels: {unexpected}")
            return 1

        y = y.map(label_mapping).astype(int)

        # 2. Split data
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # 3. Identify categorical and numeric columns
        categorical_columns = X_train.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        numerical_columns = X_train.select_dtypes(
            include=["number"]
        ).columns.tolist()

        # 4. Build preprocessing pipeline
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers = []

        if numerical_columns:
            transformers.append(
                ("numeric", numeric_pipeline, numerical_columns)
            )

        if categorical_columns:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_columns)
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        # 5. Build model pipeline
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])

        # 6. Train model
        print("[INFO] Training Random Forest model...")
        model.fit(X_train, y_train)

        # 7. Evaluate model
        predictions = model.predict(X_valid)

        accuracy = accuracy_score(y_valid, predictions)
        precision = precision_score(
            y_valid, predictions, zero_division=0
        )
        recall = recall_score(
            y_valid, predictions, zero_division=0
        )
        f1 = f1_score(
            y_valid, predictions, zero_division=0
        )

        # 8. Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_PATH)

        # 9. Save report
        report = {
            "project": "Airline Satisfaction",
            "stage": "05 - Train and Register Model",
            "model_name": "AirlineSatisfactionRandomForest",
            "algorithm": "Random Forest",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
            "feature_count": int(X.shape[1]),
            "metrics": {
                "accuracy": round(float(accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4)
            },
            "model_path": str(MODEL_PATH),
            "status": "SUCCESS"
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\n--- Model Training Summary ---")
        print("Algorithm        : Random Forest")
        print(f"Training rows    : {len(X_train)}")
        print(f"Validation rows  : {len(X_valid)}")
        print(f"Accuracy         : {accuracy:.4f}")
        print(f"Precision        : {precision:.4f}")
        print(f"Recall           : {recall:.4f}")
        print(f"F1 Score         : {f1:.4f}")
        print(f"Model saved      : {MODEL_PATH}")
        print(f"Report saved     : {REPORT_PATH}")

        print(
            "\n[INFO] Model artifact saved locally. "
            "MLflow Model Registry registration is a separate step."
        )
        print("[SUCCESS] Model training completed.")

        return 0

    except Exception as error:
        print(f"[ERROR] Model training failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())