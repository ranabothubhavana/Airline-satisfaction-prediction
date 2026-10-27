# Lab 3 and Lab 4

## Lab 3 - Baseline ML Pipeline
Run from the project root:
`python pipelines/run_lab3_baseline.py`

## Lab 4 - MLflow Experiment Tracking
Install dependencies once:
`python -m pip install -r requirements.txt`
Then run:
`python pipelines/run_lab4_tracking.py`

Lab 4 uses the same dataset and preprocessing artifacts as Lab 3 and adds MLflow tracking plus reproducibility validation.

## Lab 5 — Production Data Pipeline

Run from the project root:

```powershell
python pipelines/run_lab5_pipeline.py
```

The pipeline performs:
1. Production/train-test data validation.
2. Preprocessing of the supplied test/production data using the fitted Lab 3 preprocessor.
3. Prediction using the Lab 3 trained model.
4. Output validation.

Lab 5 writes its generated files to `outputs/` and validation summaries to `results/`.
