# Predicting 30-Day Hospital Readmission in Diabetes Patients

**Machine Learning | Healthcare Analytics | Predictive Modelling | Explainable AI**

**MSc Business Analytics — Queen's University Belfast (2023)**

## Project Overview

This project explores the application of machine learning techniques to predict hospital readmissions among patients diagnosed with diabetes.

I originally undertook this research as part of my MSc Business Analytics dissertation at Queen's University Belfast in 2023. The study focused on identifying factors associated with hospital readmissions and evaluating the effectiveness of different machine learning algorithms in predicting patient outcomes.

The research involved data preprocessing, exploratory data analysis, feature engineering, statistical feature selection, predictive modelling, and model interpretability.

I have since revisited the project to improve its code organisation, reproducibility, documentation, and evaluation framework, while incorporating a structured Python implementation.

The updated implementation uses a dataset containing 25,000 records and evaluates seven machine learning classification algorithms. As this dataset differs from the original dissertation dataset, the current results represent a separate implementation rather than an exact reproduction of the original research findings.

The project demonstrates how machine learning can be applied to healthcare data to investigate patterns, assess predictive performance, and explore factors associated with hospital readmissions.

## Research Objectives

The primary objective of this research is to investigate the effectiveness of machine learning in predicting hospital readmissions among patients with diabetes.

The project focuses on the following objectives:

1. Explore demographic, clinical, and hospitalisation-related characteristics associated with patient readmissions.
2. Prepare and transform healthcare data for predictive modelling.
3. Apply feature engineering and preprocessing techniques to improve data quality and model compatibility.
4. Develop and compare multiple machine learning classification algorithms.
5. Evaluate predictive performance using appropriate classification metrics.
6. Explore model interpretability using SHAP and LIME.
7. Identify limitations and opportunities for improving predictive modelling approaches in healthcare.

## Technologies and Tools

| Category | Technologies |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn, CatBoost |
| Data visualisation | Matplotlib, Seaborn |
| Model interpretability | SHAP, LIME |
| Development environment | Jupyter Notebook, Python |
| Testing | Pytest |
| Version control | Git, GitHub |

## Dataset

The current implementation uses a structured healthcare dataset containing 25,000 records, 16 predictor variables, and one binary target variable.

The target variable, `readmitted`, indicates whether a record is labelled as readmitted (`yes`) or not readmitted (`no`).

The dataset includes demographic, clinical, and hospitalisation-related information used to investigate patterns associated with readmission outcomes.

### Dataset characteristics

| Attribute | Description |
|---|---|
| Total records | 25,000 |
| Predictor variables | 16 |
| Target variable | `readmitted` |
| Target classes | Yes / No |
| Problem type | Binary classification |
| Data format | CSV |

The dataset contains approximately 53% negative and 47% positive readmission labels, indicating a relatively balanced target distribution.

**Dataset note:** My original MSc dissertation used a larger dataset comprising 101,766 hospital encounters and approximately 50 variables. The current dataset has a different structure and does not contain all the original clinical attributes.

The source, redistribution permissions, and precise readmission timeframe of the current dataset require confirmation. Consequently, the current implementation should not be interpreted as independently validating 30-day readmission prediction.

## Methodology

The project follows a structured machine learning workflow covering data preparation, model development, evaluation, and interpretation.

### 1. Exploratory Data Analysis

Exploratory data analysis was used to examine the dataset's structure, identify potential data-quality issues, and understand the distribution of predictor and target variables.

The analysis considered:

- Dataset dimensions and variable types
- Missing values and categorical variables
- Distribution of readmission outcomes
- Numerical and categorical feature characteristics
- Potential relationships between predictors and readmission outcomes

These investigations informed subsequent preprocessing and modelling decisions.

### 2. Data Preprocessing and Feature Engineering

Data preprocessing was performed to prepare the dataset for machine learning.

The current implementation includes:

- Missing-value imputation
- Categorical variable encoding
- Numerical feature scaling where appropriate
- Train/test separation
- Consistent preprocessing through Scikit-learn pipelines

The dataset was divided into training and testing subsets using a stratified 80/20 split to maintain approximately consistent class proportions.

Preprocessing transformations were fitted using the training data to reduce the risk of information leakage into the evaluation dataset.

The original dissertation additionally explored ICD-9 diagnostic grouping, statistical feature selection, and oversampling techniques. These procedures cannot be reproduced exactly because the current dataset does not contain the same underlying variables.

### 3. Machine Learning Models

Seven supervised machine learning algorithms were implemented and compared.

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification model used as a baseline |
| Decision Tree | Tree-based classifier using recursive feature splits |
| Random Forest | Ensemble method combining multiple decision trees |
| Support Vector Machine | Classification using a separating decision boundary |
| Histogram Gradient Boosting | Iterative ensemble learning using boosted decision trees |
| CatBoost | Gradient boosting algorithm with support for categorical data |
| Multilayer Perceptron | Feedforward neural network for classification |

The models were selected to compare different approaches to predictive classification, including linear models, tree-based algorithms, ensemble learning, kernel-based classification, and neural networks.

The current implementation uses predefined model configurations. It does not reproduce the full hyperparameter optimisation process described in the original dissertation.

### 4. Model Evaluation

The models were evaluated using a stratified holdout test set.

The following metrics were used:

- **Accuracy:** Proportion of correctly classified observations.
- **Precision:** Proportion of predicted positive cases that were correctly identified.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC AUC:** Ability of the model to distinguish between positive and negative classes across classification thresholds.
- **Average Precision:** Summary of precision-recall performance across thresholds.

An optional three-fold cross-validation procedure is also available for additional assessment of model performance on the training data.

## Model Performance

The following results were obtained from the current implementation using the 25,000-record dataset.

| Model | Accuracy | F1-score | ROC AUC |
|---|---:|---:|---:|
| CatBoost | 62.14% | 0.537 | **0.660** |
| Histogram Gradient Boosting | 62.14% | 0.563 | 0.655 |
| Multilayer Perce
