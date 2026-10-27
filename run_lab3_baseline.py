import os
import subprocess
import sys


def execute_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 3: Baseline ML Pipeline")
    print("[INFO] =========================================")

    scripts = [
        "src/preprocess.py",
        "src/train.py",
        "src/evaluate.py",
    ]

    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, os.path.join(base_dir, script)], cwd=base_dir)
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed with exit code {result.returncode}.")
            sys.exit(1)

    print("\n[SUCCESS] Lab 3 Pipeline fully executed!")


if __name__ == "__main__":
    execute_pipeline()
