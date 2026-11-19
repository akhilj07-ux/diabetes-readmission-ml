# Predicting 30-Day Hospital Readmission in Diabetes Patients

**MSc Business Analytics dissertation (2023) | Reconstructed ML portfolio implementation (2026)**

This repository revisits a 2023 MSc dissertation completed at Queen’s University Belfast. It benchmarks machine-learning classifiers for a binary hospital readmission label using a **different, supplied 25,000-row dataset**. The original dissertation described a 101,766-encounter, 50-variable dataset. Therefore **the reconstruction is not an exact replication of the dissertation results**.

> **Research/educational demonstration only.** These models are not validated for clinical use, patient triage, or decision-making.

## What is included

- Original 2023 dissertation, technical report and surviving exploratory notebook under `docs/original/`
- Clean, executable preprocessing and modelling pipeline in `src/train.py`
- Seven classifiers: Logistic Regression, Decision Tree, Random Forest, SVM, Histogram Gradient Boosting, CatBoost and Multilayer Perceptron
- Stratified 80/20 train/test split; train-only fitted imputation, scaling and one-hot encoding
- Accuracy, precision, recall, F1, ROC AUC and average precision
- Optional 3-fold cross-validation on training data (`--cv`)
- SHAP and LIME explanation code in `src/explain.py` (requires installing optional `lime`; not executed in the build environment)
- Generated model-comparison plots and confusion matrix in `results/figures/`

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
# Optional: more thorough but slower
python -m src.train --cv
# Optional: SHAP + LIME; may be slow
python -m src.explain
pytest -q
```

Run from the repository root. The supplied data is `data/hospital_readmissions.csv`.

## Dataset and limitations

The supplied CSV has 25,000 rows, 16 predictor columns and a binary `readmitted` target (`yes`/`no`). It does not include patient identifiers, raw ICD-9 codes, or the same variables as the 2023 dissertation. Diagnostic values are already grouped into broad categories. Consequently, original ICD-9 grouping, patient-level deduplication and original feature selection cannot be faithfully reproduced. The data's original provenance, redistribution licence, and whether the label specifically represents **30-day** readmission still require confirmation. Do not infer clinical validity from a high score.

This implementation deliberately does **not** use SMOTE by default: the supplied labels are approximately 53% `no` and 47% `yes`. Oversampling without a train-only pipeline risks leakage. The models use basic regularisation and fixed hyperparameters, **not the dissertation's original grid search**. The original report's performance claims have not been reproduced.

The `results/model_comparison.csv` table compares all models on one holdout set. Ranking models by that same test set is exploratory and makes the winning score optimistic. For publishable model selection, select hyperparameters/models by cross-validation **within the training set** and evaluate the chosen model once on untouched holdout data. Patient-level or temporal validation is not possible with the supplied fields.

## Outputs

- `results/model_comparison.csv` — measured performance for this reconstruction
- `results/figures/model_comparison.png` — model ROC AUC comparison
- `results/figures/confusion_matrix.png` and `roc_curve.png`
- `results/figures/shap_waterfall.png` and `lime_explanation.html` after installing dependencies and running `src.explain`

## Project structure

```text
src/             training, evaluation and explanation modules
data/            supplied alternative dataset
docs/original/   original 2023 academic work
results/         regenerated outputs
notebooks/       explanatory notebook
tests/          basic checks
```

## Academic context

**Author:** Akhil Javvadi  
**Degree:** MSc Business Analytics, Queen's University Belfast (2023)  
**Research topic:** Predicting the likelihood of readmission of diabetes patients within 30 days of discharge.

**Original study vs reconstruction:** The original study discusses feature selection (ANOVA/RFE), oversampling (SMOTE), ensemble classifiers and SHAP/LIME. The code surviving from 2023 is exploratory; the code in `src/` was newly reconstructed in 2026, not recovered verbatim. 

