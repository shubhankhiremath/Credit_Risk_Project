# Credit Risk Prediction Using Machine Learning

Multiclass classification of loan-prospect **priority levels** (`Approved_Flag`: P1, P2, P3, P4) from bureau trade-line and CIBIL-style features. One Jupyter notebook trains a scikit-learn pipeline; a Streamlit app serves the saved artifact.

This is an educational portfolio project, not a lending decision system.

## Overview

Two Excel case-study tables describe the **same 51,336 prospects** and are joined on `PROSPECTID`. The model maps application and bureau fields to a four-class priority flag. `Credit_Score` is excluded because it tracks the label too closely to be a fair predictor.

## Problem Statement

Lenders need a consistent way to bucket applicants when the majority class (P2) is about **62.7%** of rows. Accuracy alone is not a useful objective. This project optimizes **macro-F1** so minority priority levels are not ignored.

## Dataset

Files in `dataset/` (not replaced):

| File | Sheet | Role |
| --- | --- | --- |
| `Features_Target_Description.xlsx` | `Sheet1` | Data dictionary |
| `case_study1.xlsx` | `case_study1` | 51,336 × 26 internal trade-line counts |
| `case_study2.xlsx` | `case_study2` | 51,336 × 62 bureau, demographic, score, and target fields |

Join is 1:1 on `PROSPECTID`. Pandas NaNs are essentially absent; missingness is coded as **`-99999`**.

The dictionary describes `Approved_Flag` only as **priority levels**. Class meanings beyond that label are not invented here.

## Features

- **Internal:** account counts (total/active/closed, product types, missed payments, age of oldest/newest trade line).
- **Bureau / CIBIL-style:** delinquency counts and levels, standard/substandard/doubtful/loss flags, enquiries, utilization (often missing), current-employer tenure, income, education, marital status, gender, product enquired.
- **Excluded:** `PROSPECTID` (identifier), `Credit_Score` (leakage candidate).
- **Dropped on training data only:** columns with >50% missing (`time_since_first_deliquency`, `time_since_recent_deliquency`, `max_delinquency_level`, `CC_utilization`, `PL_utilization`) and near-duplicate numeric columns with |r| > 0.95.

`GENDER` and `NETMONTHLYINCOME` have no description in the source dictionary.

## Machine Learning Workflow

```text
Excel files
  → join on PROSPECTID
  → sentinel -99999 → NaN
  → row-wise feature engineering
  → stratified 80/20 split
  → impute/encode fit on train
  → drop high-missing / high-correlation using train only
  → class-weighted models
  → compare → tune on train CV
  → test evaluation + SHAP
  → save pipeline → Streamlit
```

## Exploratory Data Analysis

- Target is imbalanced: P2 ~62.7%, P3 ~14.5%, P4 ~11.5%, P1 ~11.3%.
- Credit score ranges overlap the flags but separate P1 (higher scores on average) from P4 (lower on average). That association is why the score is not a model input.
- Plots in the notebook cover class mix, numeric distributions, and risk mix by categoricals that actually exist in the data.

## Feature Engineering

Row-wise ratios (no train-set statistics):

- `missed_pmnt_per_tl` — missed payments relative to trade-line count
- `delinquent_per_tl` — delinquency events relative to trade-line count
- `recent_enq_share` — last-3-month enquiries as a share of last-12-month enquiries

Debt-to-income and loan-to-income were **not** created; those source columns are not in this dataset.

## Models

Compared with the same feature set and a stratified held-out test set (20%, `random_state=42`):

1. Logistic Regression (scaled numeric features, `class_weight="balanced"`)
2. Decision Tree
3. Random Forest
4. XGBoost (`multi:softprob`, balanced sample weights)

SMOTE was **not** used. Imbalance is handled with class/sample weights so the test distribution is unchanged.

## Model Comparison

Test-set metrics from notebook execution (not typed by hand):

| Model | Accuracy | Precision_w | Recall_w | F1_w | Precision_macro | Recall_macro | F1_macro | ROC-AUC OVR | PR-AUC OVR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6872 | 0.7525 | 0.6872 | 0.7047 | 0.6206 | 0.7037 | 0.6463 | 0.8786 | 0.8041 |
| Decision Tree | 0.6964 | 0.7637 | 0.6964 | 0.7157 | 0.6295 | 0.7036 | 0.6523 | 0.8481 | 0.7400 |
| Random Forest | 0.7532 | 0.7823 | 0.7532 | 0.7625 | 0.6679 | 0.7342 | 0.6948 | 0.9157 | 0.8401 |
| XGBoost | 0.7300 | 0.7956 | 0.7300 | 0.7462 | 0.6650 | 0.7603 | 0.6952 | 0.9197 | 0.8504 |

Final family chosen by **macro-F1**: XGBoost (narrowly ahead of Random Forest).

## Hyperparameter Tuning

`RandomizedSearchCV`, 8 draws, 3-fold **stratified** CV on **training data only**, scoring `f1_macro`.

Selected search result: `n_estimators=180`, `max_depth=7`, `learning_rate=0.12`, `subsample=0.85`, `colsample_bytree=0.7`, `min_child_weight=1`, `gamma=0.1`. Best CV macro-F1 **0.7074**.

## Final Model Results

Untouched test set (n = 10,268):

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7651 |
| Precision (weighted) | 0.7989 |
| Recall (weighted) | 0.7651 |
| F1 (weighted) | 0.7759 |
| Precision (macro) | 0.6899 |
| Recall (macro) | 0.7544 |
| F1 (macro) | 0.7152 |
| ROC-AUC (weighted OVR) | 0.9234 |
| PR-AUC (weighted OVR) | 0.8542 |

Stratified CV on train: macro-F1 mean **0.7075** (std **0.0021**).

Per-class test F1: P1 0.7936, P2 0.8483, P3 0.4730, P4 0.7457. **P3 is the hardest class** and is the main limitation of the model.

High accuracy would still be the wrong headline: the majority class is large, and P3 errors matter for a priority system.

## SHAP Explainability

Tree SHAP on the fitted XGBoost pipeline. Summary plots show global associations; a single test row is explained for the predicted class. SHAP is **not** a causal statement about creditworthiness.

## Web Application

Streamlit loads `models/credit_risk_pipeline.pkl` (preprocessing + model). The form uses the pipeline’s real input columns (categoricals as dropdowns, numerics as numbers). Engineered ratios are computed in the app the same way as in the notebook. The result page shows predicted class, class probabilities, and top associated features from the fitted model.

## Project Structure

```text
dataset/                      Excel sources
Credit_Risk_Prediction.ipynb  all ML work
models/credit_risk_pipeline.pkl
streamlit_app.py             Streamlit entry point
app/app.py                    Flask compatibility app
app/templates/                index.html, result.html
app/static/css|js
requirements.txt
Procfile
README.md
PROJECT_RESUME.md
INTERVIEW_PREPARATION.md
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Running Locally

1. Train and save the pipeline (once):

```bash
jupyter notebook Credit_Risk_Prediction.ipynb
```

Run all cells. The last modeling cell writes `models/credit_risk_pipeline.pkl`.

3. Start the Streamlit app from the project root:

```bash
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

## Deployment

`Procfile` starts the Streamlit app on platforms that provide a `PORT` environment variable:

```text
web: streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT
```

Paths are relative to the repository root. Bind port comes from `PORT`. **This README does not claim a live Render URL**; deploy and smoke-test separately.

## Limitations

- One static dataset; no time-based or out-of-time validation.
- `Approved_Flag` is a priority label, not an observed default event.
- P3 recall/precision remain weak.
- Form collects many bureau fields; a production UI would use a much smaller application schema.
- No fairness, policy, or regulatory review.

## Future Improvements

- Cost-sensitive thresholds and a clearer P3 strategy
- Temporal split if application dates become available
- Calibration plots and error analysis by segment
- Smaller production feature set with monitoring

## Disclaimer

This application is an educational/portfolio machine-learning project and should not be used as the sole basis for real-world lending decisions.
