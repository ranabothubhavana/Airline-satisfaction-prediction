import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    "src/train_registry.py",
    "src/automate_lifecycle.py",
    "src/generate_registry_report.py",
]

def main():
    python_executable = sys.executable

    print("=" * 60)
    print("AIRLINE SATISFACTION - LAB 6 MODEL REGISTRY")
    print("=" * 60)

    for script in SCRIPTS:
        script_path = os.path.join(BASE_DIR, script)

        if not os.path.isfile(script_path):
            print(f"[ERROR] Script not found: {script_path}")
            return 1

        print(f"\n[RUNNER] Executing {script}...")

        result = subprocess.run(
            [python_executable, script_path],
            cwd=BASE_DIR,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"[ERROR] {script} failed with exit code "
                f"{result.returncode}."
            )
            return result.returncode

    print("\n[SUCCESS] Lab 6 completed successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())