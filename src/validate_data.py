import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
TARGET = "satisfaction"

REQUIRED_COLUMNS = [
    "id", "Gender", "Customer Type", "Age", "Type of Travel", "Class",
    "Flight Distance", "Inflight wifi service",
    "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service",
    "Cleanliness", "Departure Delay in Minutes",
    "Arrival Delay in Minutes", "satisfaction"
]

RATING_COLUMNS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness"
]


def validate_frame(df, name, require_target=True):
    errors = []

    # Check required columns
    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns and (require_target or c != TARGET)
    ]

    if missing:
        errors.append(f"{name}: missing required columns: {missing}")

    # Validate target column
    if require_target and TARGET in df.columns:
        target_values = set(
            df[TARGET].dropna().astype(str).str.strip().str.lower().unique()
        )

        allowed = {"neutral or dissatisfied", "satisfied"}
        unexpected = sorted(target_values - allowed)

        if unexpected:
            errors.append(
                f"{name}: unexpected target values: {unexpected}"
            )

        if df[TARGET].isna().any():
            errors.append(f"{name}: target contains missing values")

    # Validate rating columns (0-5)
    for col in RATING_COLUMNS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")

            if vals.isna().any():
                errors.append(
                    f"{name}: non-numeric or missing values found in {col}"
                )
            elif ((vals < 0) | (vals > 5)).any():
                errors.append(
                    f"{name}: values outside 0-5 range in {col}"
                )

    # Validate Age
    if "Age" in df.columns:
        age = pd.to_numeric(df["Age"], errors="coerce")

        if age.isna().any() or ((age < 0) | (age > 120)).any():
            errors.append(f"{name}: invalid Age values")

    # Validate distance and delays
    for col in [
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes"
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")

            # Arrival Delay can legitimately contain missing values.
            if col == "Arrival Delay in Minutes":
                # Allow missing values, but reject non-numeric non-missing
                # values and negative delays.
                original_non_missing = df[col].notna()
                invalid_numeric = vals.isna() & original_non_missing

                if invalid_numeric.any() or (vals.dropna() < 0).any():
                    errors.append(
                        f"{name}: invalid values in {col}"
                    )

            else:
                # Flight Distance and Departure Delay must be valid
                # non-negative numeric values.
                if vals.isna().any() or (vals < 0).any():
                    errors.append(
                        f"{name}: invalid values in {col}"
                    )

    return errors


print("[VALIDATE] Starting Lab 5 production data validation...")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Training data not found: {TRAIN_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"Test/production input not found: {TEST_PATH}"
    )

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Remove an unnamed CSV index if present
train = train.drop(columns=["Unnamed: 0"], errors="ignore")
test = test.drop(columns=["Unnamed: 0"], errors="ignore")

# Validate both datasets
errors = validate_frame(
    train,
    "train",
    require_target=True
)

test_errors = validate_frame(
    test,
    "test",
    require_target=True
)

errors.extend(test_errors)

# Save validation summary
summary = {
    "train_rows": len(train),
    "test_rows": len(test),
    "train_columns": len(train.columns),
    "test_columns": len(test.columns),
    "train_duplicates": int(train.duplicated().sum()),
    "test_duplicates": int(test.duplicated().sum()),
    "train_missing_cells": int(train.isna().sum().sum()),
    "test_missing_cells": int(test.isna().sum().sum()),
    "validation_status": "PASS" if not errors else "FAIL"
}

pd.DataFrame([summary]).to_csv(
    os.path.join(
        RESULTS_DIR,
        "lab5_data_validation_summary.csv"
    ),
    index=False
)

# Report validation errors if any
if errors:
    with open(
        os.path.join(
            RESULTS_DIR,
            "lab5_data_validation_errors.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(errors))

    print("[VALIDATE] FAILED")

    for error in errors:
        print("[VALIDATE]", error)

    sys.exit(1)

print(
    "[VALIDATE] PASS: train and test data passed schema and value checks."
)
print("[VALIDATE] Summary saved to:", RESULTS_DIR)