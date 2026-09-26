"""
Smoke-test for src/predictor.py — exercises all three public functions.
"""

import pandas as pd
from src.predictor import predict_single, predict_batch, predict_what_if

GOOD_APPLICANT = {
    "ApplicantIncome": 8000.0,
    "CoapplicantIncome": 2000.0,
    "LoanAmount": 150.0,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Employment_Status": "Salaried",
    "Marital_Status": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Property_Area": "Semiurban",
}

RISKY_APPLICANT = {
    "ApplicantIncome": 3000.0,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 400.0,
    "Loan_Amount_Term": 120,
    "Credit_History": 0.0,
    "Employment_Status": "Self_Employed",
    "Marital_Status": "No",
    "Dependents": "3+",
    "Education": "Not Graduate",
    "Property_Area": "Rural",
}


def run_smoke_tests():
    print("=" * 60)
    print("  predict_single — good applicant")
    print("=" * 60)
    result = predict_single(GOOD_APPLICANT)
    for key, value in result.items():
        print(f"  {key:<24}: {value}")

    print()
    print("=" * 60)
    print("  predict_single — risky applicant")
    print("=" * 60)
    result = predict_single(RISKY_APPLICANT)
    for key, value in result.items():
        print(f"  {key:<24}: {value}")

    print()
    print("=" * 60)
    print("  predict_batch — two applicants")
    print("=" * 60)
    batch_df = pd.DataFrame([GOOD_APPLICANT, RISKY_APPLICANT])
    batch_result = predict_batch(batch_df)
    print(batch_result[["approved", "approval_probability", "emi", "dti", "lti"]].to_string())

    print()
    print("=" * 60)
    print("  predict_what_if — increase loan amount on good applicant")
    print("=" * 60)
    baseline = predict_single(GOOD_APPLICANT)
    what_if = predict_what_if(GOOD_APPLICANT, {"LoanAmount": 800.0})
    print(f"  {'Field':<24}  {'Baseline':>12}  {'LoanAmount=800':>14}")
    print(f"  {'-'*24}  {'-'*12}  {'-'*14}")
    for key in ["approved", "approval_probability", "emi", "dti", "lti"]:
        print(f"  {key:<24}  {str(baseline[key]):>12}  {str(what_if[key]):>14}")

    print()
    print("=" * 60)
    print("  Error handling — missing field")
    print("=" * 60)
    try:
        predict_single({"ApplicantIncome": 5000.0})
    except ValueError as error:
        print(f"  ValueError caught: {error}")


if __name__ == "__main__":
    run_smoke_tests()
