# AURIC – Alternate Credit Scoring System  
An end-to-end, explainable machine learning pipeline for predicting credit default risk using traditional and alternative financial data.

## Overview
**AURIC (Alternate Unified Risk Intelligence & Credit Scoring)** is designed to improve credit risk assessment by:
- Integrating alternative behavioral data (bureau + installment history)
- Engineering financial risk indicators
- Handling class imbalance effectively
- Using ensemble learning for robust prediction
- Providing transparent, explainable decisions via SHAP
The goal is to improve detection of high-risk applicants while maintaining interpretability for financial decision-making.

## Problem Statement
Traditional credit scoring systems:
- Overlook behavioral repayment signals  
- Struggle with class imbalance  
- Lack decision transparency  
- Often exclude thin-file or partially banked applicants  
AURIC addresses these issues using a structured ML pipeline that enhances both predictive performance and explainability.

## Dataset
Built on the **Home Credit Default Risk dataset**, including:
- `application_train.csv` – Applicant financial and demographic data  
- `bureau.csv` – External credit bureau records  
- `installments_payments.csv` – Historical repayment behavior  

## Pipeline Architecture
### 1. Data Integration
Merged applicant data with alternative features derived from:
**Bureau history**
- `BUREAU_LOAN_COUNT`
- `BUREAU_AVG_OVERDUE`
**Installment repayment**
- `AVG_DAYS_LATE`
- `MAX_DAYS_LATE`

Missing values in derived features are logically filled with `0` (no history assumption).

### 2. Exploratory Data Analysis
Key analyses performed:
- Target imbalance visualization (~8% defaulters)
- Income tier vs default probability
- Education level vs risk
- Age vs risk distribution
- Repayment delay impact on default

### 3. Feature Engineering
Engineered financial indicators:
- **Payment-to-Income Ratio (PTI)**
- **Credit-to-Income Ratio**
- **Credit Term Length**
- Age (converted from `DAYS_BIRTH`)
- Repayment lateness categories

### 4. Preprocessing
- Median imputation for numerical features
- Most frequent imputation for categorical features
- One-hot encoding
- Standard scaling (for Logistic Regression)
- SMOTE for imbalance handling (baseline model)
- Class weighting for boosting models
Final feature matrix: **~229 features after encoding**.

## Models Implemented
- Logistic Regression (Baseline)
- XGBoost
- LightGBM
- CatBoost
- Soft Voting Ensemble (Final Model)

## Final Model Performance (Soft Voting Ensemble)

| Metric     | Score  |
|------------|--------|
| ROC-AUC    | 0.7726 |
| Recall     | 0.6709 |
| Precision  | 0.1831 |
| Accuracy   | 0.7318 |

The ensemble improves performance over the Logistic Regression baseline (ROC-AUC ~0.74).
The model prioritizes **recall** to capture a higher proportion of defaulters.

## Risk Tier Classification
Predicted probabilities are mapped into:
- **Low Risk** (< 0.30)
- **Medium Risk** (0.30 – 0.60)
- **High Risk** (> 0.60)
This enables:
- Automated approval for low risk
- Manual review for medium risk
- Rejection for high risk

## Explainability – SHAP Integration
To ensure transparency:
### Global Interpretability
- Identifies top risk drivers
- `EXT_SOURCE` scores are strongest predictors
- Repayment delay and bureau history significantly influence risk
### Local Interpretability
For each applicant:
- Waterfall plots
- Decision summaries
- Feature-level contribution analysis
This supports compliance and human-in-the-loop review.

## Key Highlights
- Integrated alternative behavioral data
- Built ensemble boosting architecture
- Addressed severe class imbalance
- Achieved strong defaulter recall
- Implemented model explainability
- Enabled risk-tier automation
