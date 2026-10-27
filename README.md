# Airline Passenger Satisfaction Prediction

## 1. Project Overview
This project predicts whether an airline passenger is **Satisfied** or **Neutral/Dissatisfied** using demographic, travel, service-rating, flight-distance, and delay-related information.

## 2. Problem Type
**Supervised Binary Classification**

Target:
- `satisfaction`

Classes:
- `satisfied`
- `neutral or dissatisfied`

## 3. Dataset
Files:
- `dataset/train.csv`
- `dataset/test.csv`

The model is trained using `train.csv`. The test file is treated as unseen data for final sample/all-row prediction.

## 4. Project Workflow
1. Understand the dataset and define the problem
2. Exploratory Data Analysis
3. Missing-value handling
4. Duplicate handling
5. Outlier inspection through EDA
6. Feature selection
7. Categorical encoding
8. Numerical scaling
9. Train/validation split
10. Multiple ML models
11. Cross-validation
12. Hyperparameter tuning
13. Model comparison
14. Best-model selection
15. Unseen-data prediction
16. Final conclusion

## 5. Models
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Gradient Boosting
- Tuned Random Forest
- Tuned Gradient Boosting

## 6. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

F1-score is used as the main selection metric because it balances precision and recall.

## 7. Preprocessing
- Duplicate rows are removed.
- Missing numerical values are imputed with the median.
- Missing categorical values are imputed with the most frequent category.
- Categorical features are one-hot encoded.
- Numerical features are standardized.
- Passenger `id` is excluded because it is an identifier rather than a meaningful predictive feature.

## 8. How to Run in VS Code

Open this folder in VS Code.

### Create virtual environment
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install packages
```bash
pip install -r requirements.txt
```

### Run EDA
```bash
python src/eda.py
```

### Train models
```bash
python src/train.py
```

### Predict unseen data
```bash
python src/predict.py
```

## 9. Output Files
After training, check:
- `results/dataset_summary.csv`
- `results/model_comparison.csv`
- `results/cross_validation_results.csv`
- `results/tuned_model_results.csv`
- `results/all_model_results.csv`
- `results/best_model_classification_report.txt`
- `results/figures/`
- `models/best_model.joblib`

After prediction:
- `results/test_predictions.csv`

## 10. Final Conclusion Template
The project developed a binary classification system for airline passenger satisfaction. Multiple machine learning algorithms were trained and evaluated using accuracy, precision, recall, F1-score and ROC-AUC. Cross-validation and hyperparameter tuning were applied to stronger candidate models. The model with the strongest validation F1-score was selected as the final model and used to predict unseen passenger records.

**Important:** Replace the final conclusion with the actual metric values printed by `python src/train.py`.

## 11. Lab 3 Baseline Pipeline
The project now also supports the Lab 3 three-stage baseline pipeline:

1. `src/preprocess.py` - cleans the airline dataset, creates the stratified train/validation split, and prepares the preprocessing pipeline.
2. `src/train.py` - trains a Logistic Regression baseline and saves the model.
3. `src/evaluate.py` - evaluates the baseline using Accuracy, Precision, Recall, F1-score and ROC-AUC.

Run all three stages from the project root with:
```bash
python run_lab3_baseline.py
```

Lab 3 outputs are stored in `results/`, including `lab3_baseline_results.csv`, `lab3_baseline_classification_report.txt`, `lab3_confusion_matrix.csv`, and `lab3_validation_predictions.csv`.
