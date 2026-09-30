"""
Risk feature engineering for loan applicants.

Three derived ratios are standard inputs in credit-risk modelling:

  EMI  (Estimated Monthly Instalment)
       = (LoanAmount * 1000) / Loan_Amount_Term
       LoanAmount is stored in thousands, so multiply back to get rupees.
       Dividing by term (months) gives a flat EMI approximation (no interest
       rate is available in the raw data, so we treat it as principal-only).

  DTI  (Debt-to-Income ratio)
       = EMI / monthly_income
       Monthly income = (ApplicantIncome + CoapplicantIncome) / 12.
       DTI > 0.43 is the conventional "qualified mortgage" ceiling; keeping it
       as a continuous feature lets tree-based and linear models find the
       threshold themselves.

  LTI  (Loan-to-Income ratio)
       = LoanAmount / annual_income
       A macro-level affordability gauge that is less sensitive to loan term
       than DTI and is widely used by Indian banking regulators (RBI guidelines
       reference LTI caps of 4–6x for home loans).

All three are computed on the already-imputed DataFrame so that no NaN
propagates into the feature matrix.
"""

import pandas as pd


MONTHLY_INCOME_DIVISOR = 12
LOAN_AMOUNT_SCALE = 1000


def build_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMI, DTI, and LTI columns to *df* and return the augmented DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ApplicantIncome, CoapplicantIncome, LoanAmount,
        and Loan_Amount_Term.  Missing values must have been imputed
        before calling this function.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with three additional columns appended in-place:
        ``emi``, ``dti``, ``lti``.

    Raises
    ------
    ValueError
        If any row has ApplicantIncome <= 0 or Loan_Amount_Term <= 0.
        Applicant income must be strictly positive to produce a meaningful
        DTI ratio, and loan tenure must be strictly positive to avoid
        division by zero in the EMI calculation.
    """
    df = df.copy()

    non_positive_income_mask = df["ApplicantIncome"] <= 0
    if non_positive_income_mask.any():
        bad_values = df.loc[non_positive_income_mask, "ApplicantIncome"].tolist()
        raise ValueError(
            f"ApplicantIncome must be strictly positive. "
            f"Got non-positive values: {bad_values}"
        )

    non_positive_term_mask = df["Loan_Amount_Term"] <= 0
    if non_positive_term_mask.any():
        bad_values = df.loc[non_positive_term_mask, "Loan_Amount_Term"].tolist()
        raise ValueError(
            f"Loan_Amount_Term must be strictly positive. "
            f"Got non-positive values: {bad_values}"
        )

    monthly_income = (df["ApplicantIncome"] + df["CoapplicantIncome"]) / MONTHLY_INCOME_DIVISOR
    annual_income = df["ApplicantIncome"] + df["CoapplicantIncome"]

    df["emi"] = (df["LoanAmount"] * LOAN_AMOUNT_SCALE) / df["Loan_Amount_Term"]
    df["dti"] = df["emi"] / monthly_income.replace(0, float("nan"))
    df["lti"] = df["LoanAmount"] / annual_income.replace(0, float("nan"))

    return df
