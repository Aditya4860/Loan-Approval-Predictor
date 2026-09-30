# Loan Approval Predictor

A production-ready machine-learning system that predicts whether a loan application will be approved, built end-to-end in Python with scikit-learn and Streamlit.

---

## Table of Contents

- [Project Purpose](#project-purpose)
- [Prediction Modes](#prediction-modes)
- [Risk Metrics](#risk-metrics)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Regenerating the Model](#regenerating-the-model)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Running with Docker](#running-with-docker)
- [Running the Test Suite](#running-the-test-suite)

---

## Project Purpose

This project demonstrates a complete ML pipeline for binary classification of loan applications as **Approved (Y)** or **Rejected (N)**. It covers:

- Synthetic data generation with realistic statistical distributions
- Feature engineering (three domain-specific credit-risk ratios)
- Preprocessing pipeline (imputation → feature engineering → encoding → scaling), serialised with `joblib` for identical train/inference behaviour
- Model selection via 5-fold stratified cross-validation across three candidates
- A Streamlit frontend with single-entry, batch-upload, and what-if sensitivity tabs
- A pytest suite (73 tests) covering the full inference stack

**Winner model:** `GradientBoostingClassifier` — ROC-AUC **0.91**, F1 **0.94** on held-out test set.

---

## Prediction Modes

### 1. Single Entry (`predict_single`)
Pass one applicant's raw fields as a Python `dict`. Returns:

```python
{
    "approved": bool,
    "approval_probability": float,   # 0–1
    "emi": float,                    # ₹/month
    "dti": float,
    "lti": float,
}
```

### 2. Batch Prediction (`predict_batch`)
Pass a `pd.DataFrame` with the same schema (minus the target column). Returns the original DataFrame with five columns appended: `approved`, `approval_probability`, `emi`, `dti`, `lti`. A sample file for testing is at `data/sample_batch.csv`.

### 3. What-If Analysis (`predict_what_if`)
Pass a base applicant dict and a partial `overrides` dict. The two are merged and run through the same pipeline, so you can vary a single field (e.g. `{"LoanAmount": 800}`) and observe how the approval probability shifts.

---

## Risk Metrics

Three derived ratios are computed from raw fields before the model sees any data:

| Metric | Formula | Threshold |
|--------|---------|-----------|
| **EMI** – Estimated Monthly Instalment | `(LoanAmount × 1000) / Loan_Amount_Term` | — |
| **DTI** – Debt-to-Income ratio | `EMI / (monthly_income)` where `monthly_income = (ApplicantIncome + CoapplicantIncome) / 12` | > 0.43 flagged ⚠️ |
| **LTI** – Loan-to-Income ratio | `LoanAmount / annual_income` where `annual_income = ApplicantIncome + CoapplicantIncome` | > 5.0 flagged ⚠️ |

`LoanAmount` is stored in **thousands of INR** in the raw data; the EMI formula scales it back to absolute rupees.

---

## Project Structure

```
loan-approval-predictor/
├── app/
│   └── streamlit_app.py        # Three-tab Streamlit frontend
├── data/
│   ├── raw/
│   │   └── loan_data.csv       # 5,000-row synthetic dataset
│   ├── processed/
│   │   ├── train.csv           # Preprocessed training split (4,000 rows)
│   │   └── test.csv            # Preprocessed test split (1,000 rows)
│   └── sample_batch.csv        # 10 example applicants for Tab 2 demo
├── models/
│   ├── preprocessor.joblib     # Fitted ColumnTransformer
│   ├── loan_model.joblib       # Fitted GradientBoostingClassifier
│   └── feature_importance.png  # Gini importance + confusion matrix
├── notebooks/
│   ├── eda.py                  # EDA script (shape, distributions, plots)
│   └── plots/                  # Four PNG charts from EDA
├── src/
│   ├── __init__.py
│   ├── generate_synthetic_data.py
│   ├── risk_metrics.py         # EMI / DTI / LTI feature engineering
│   ├── data_processing.py      # Full preprocessing pipeline + prepare_data()
│   ├── model_training.py       # CV, model selection, evaluation, artifacts
│   └── predictor.py            # Inference layer (predict_single/batch/what_if)
├── tests/
│   ├── test_risk_metrics.py    # 18 tests
│   ├── test_predictor.py       # 31 tests
│   └── test_data_processing.py # 24 tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup

> **Requires Python 3.12.** The pinned dependencies do not yet ship prebuilt wheels for Python 3.14+.

```bash
# 1. Clone the repository
git clone https://github.com/Aditya4860/Loan-Approval-Predictor.git
cd Loan-Approval-Predictor

# 2. Create and activate a virtual environment
py -3.12 -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install all pinned dependencies
pip install --only-binary=:all: -r requirements.txt
```

> `--only-binary=:all:` prevents pip from trying to compile packages from source, which avoids long build times and C-compiler requirements.

### Required raw-data fields

When calling the predictor programmatically, every applicant record must contain:

| Field | Type | Example |
|-------|------|---------|
| `ApplicantIncome` | float | `8500.0` |
| `CoapplicantIncome` | float | `2000.0` |
| `LoanAmount` | float (₹ thousands) | `180.0` |
| `Loan_Amount_Term` | int (months) | `360` |
| `Credit_History` | float (`1.0` / `0.0`) | `1.0` |
| `Employment_Status` | str | `"Salaried"` or `"Self_Employed"` |
| `Marital_Status` | str | `"Yes"` or `"No"` |
| `Dependents` | str | `"0"`, `"1"`, `"2"`, or `"3+"` |
| `Education` | str | `"Graduate"` or `"Not Graduate"` |
| `Property_Area` | str | `"Urban"`, `"Semiurban"`, or `"Rural"` |

---

## Regenerating the Model

The fitted model and preprocessor are already committed to the repository. To retrain from scratch (e.g. after modifying features or swapping the dataset):

```bash
# Regenerate the synthetic dataset
python -m src.generate_synthetic_data

# Rerun the full training pipeline
# This will re-fit the preprocessor and all three candidate models,
# select the best by CV ROC-AUC, evaluate on the test set, and save:
#   models/preprocessor.joblib
#   models/loan_model.joblib
#   models/feature_importance.png
python -m src.model_training
```

---

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app opens at **http://localhost:8501** and provides three tabs:

| Tab | Description |
|-----|-------------|
| 📋 **Single Entry** | Fill in one applicant's details and get an instant pass/fail decision with risk metrics |
| 📂 **Batch CSV** | Upload `data/sample_batch.csv` (or your own CSV) to predict multiple applicants at once and download results |
| 🔬 **What-If Analysis** | Set a base applicant, choose a field to vary (e.g. Loan Amount), and plot how approval probability changes across a range |

---

## Running with Docker

```bash
# Build the image
docker build -t loan-approval-predictor .

# Run the container
docker run -p 8501:8501 loan-approval-predictor
```

The app will be available at **http://localhost:8501**.

---

## Running the Test Suite

```bash
pytest tests/ -v
```

Expected output: **73 passed** in under 3 seconds.

The suite covers:
- `test_risk_metrics.py` — EMI / DTI / LTI formulas, edge cases, and `ValueError` guards
- `test_predictor.py` — return shapes, types, value ranges, and error handling for all three prediction modes
- `test_data_processing.py` — split shapes, NaN-free outputs, target encoding, and stratified balance
