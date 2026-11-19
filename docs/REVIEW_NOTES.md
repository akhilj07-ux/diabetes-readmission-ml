# Reconstruction review and limitations

1. The original 2023 proposal describes 101,766 encounters and 50 variables. The supplied 2026 CSV has 25,000 rows and 17 columns (including target).
2. The uploaded notebook is a partial exploratory artefact, not the final modelling pipeline. Its original code is preserved unmodified.
3. The 2023 technical report states Random Forest was preferred and cites high precision; these numbers cannot be attributed to the reconstructed dataset.
4. The CSV `readmitted` label has `yes` and `no`, but its exact time horizon is not documented in the CSV. This is a material limitation for a 30-day claim.
5. The CSV has no patient ID; patient-level leakage and independence cannot be assessed. Train/test split is row-based.
6. The dataset has near-balanced classes, so SMOTE is not applied. Raw ICD-9 recoding is not possible because diagnoses are pre-grouped.
7. No deployment, clinical validation, subgroup fairness audit, calibration or external validation is claimed.
8. Models and explanations were reconstructed in 2026, not copied from missing original 2023 modelling notebooks.
9. The scripts rank models using holdout ROC AUC; this is descriptive, not unbiased final model selection.
10. Verify rights to publish the original academic PDFs and to redistribute the CSV before making the repository public.
