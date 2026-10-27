import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "train.csv")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(15))
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe(include="all").T)

# Numerical distributions
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    plt.figure(figsize=(7, 4))
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in col)
    plt.savefig(os.path.join(FIG_DIR, f"distribution_{safe}.png"), dpi=120)
    plt.close()

print("\nEDA complete. Charts are in results/figures/")
