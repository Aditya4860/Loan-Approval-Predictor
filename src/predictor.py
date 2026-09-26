"""
Inference layer for the loan-approval predictor.

Artifacts are loaded once at module import time so that repeated calls to
predict_single, predict_batch, or predict_what_if share a single in-memory
copy of both the preprocessing pipeline and the trained model.  This avoids
the I/O overhead of reloading large joblib files on every request and makes
the module safe to use behind a long-running web server (Streamlit, FastAPI,
etc.).

Required raw-data fields (same schema as the training CSV, minus Loan_Status):
    ApplicantIncome    (float) – monthly income of the primary applicant, INR
    CoapplicantIncome  (float) – monthly income of the co-applicant, 0 if none
    LoanAmount         (float) – requested loan amount in thousands of INR
    Loan_Amount_Term   (int)   – repayment tenure in months
    Credit_History     (float) – 1.0 = good history, 0.0 = bad, NaN = unknown
    Employment_Status  (str)   – "Salaried" or "Self_Employed"
    Marital_Status     (str)   – "Yes" or "No"
    Dependents         (str)   – "0", "1", "2", or "3+"
    Education          (str)   – "Graduate" or "Not Graduate"
    Property_Area      (str)   – "Urban", "Semiurban", or "Rural"

Derived risk fields (emi, dti, lti) are computed internally by
build_risk_features and must NOT be supplied by the caller.
"""

import joblib
import pandas as pd

from src.data_processing import get_feature_names
from src.risk_metrics import build_risk_features

PREPROCESSOR_PATH = "models/preprocessor.joblib"
MODEL_PATH = "models/loan_model.joblib"

REQUIRED_RAW_FIELDS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Employment_Status",
    "Marital_Status",
    "Dependents",
    "Education",
    "Property_Area",
]

RISK_FEATURE_FIELDS = ["emi", "dti", "lti"]

_preprocessor = joblib.load(PREPROCESSOR_PATH)
_model = joblib.load(MODEL_PATH)


def _validate_fields(applicant: dict) -> None:
    """
    Raise ValueError if any required raw field is absent from *applicant*.

    Parameters
    ----------
    applicant : dict
        Raw applicant field dictionary to validate.

    Raises
    ------
    ValueError
        Lists every missing field in a single human-readable message so the
        caller can fix all problems in one round-trip.
    """
    missing = [field for field in REQUIRED_RAW_FIELDS if field not in applicant]
    if missing:
        raise ValueError(
            f"The following required fields are missing from the applicant record: "
            f"{missing}. "
            f"All of {REQUIRED_RAW_FIELDS} must be present."
        )


def _applicant_to_dataframe(applicant: dict) -> pd.DataFrame:
    """
    Convert a single applicant dict to a one-row DataFrame with correct dtypes.

    Parameters
    ----------
    applicant : dict
        Raw applicant fields (must have passed _validate_fields).

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with column order matching the training schema.
    """
    return pd.DataFrame([{field: applicant[field] for field in REQUIRED_RAW_FIELDS}])


def _run_pipeline(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Apply feature engineering and the fitted preprocessing pipeline to *raw_df*.

    Parameters
    ----------
    raw_df : pd.DataFrame
        DataFrame containing only the required raw fields (no target column).

    Returns
    -------
    featured_df : pd.DataFrame
        raw_df augmented with emi, dti, lti columns (before scaling).
    approved : pd.Series of bool
        Binary approval decision for each row.
    probabilities : pd.Series of float
        Approval probability (class=1) for each row.
    """
    featured_df = build_risk_features(raw_df)
    transformed_array = _preprocessor.transform(featured_df)
    transformed_df = pd.DataFrame(
        transformed_array,
        columns=get_feature_names(_preprocessor),
        index=raw_df.index,
    )
    probabilities = pd.Series(
        _model.predict_proba(transformed_df)[:, 1],
        index=raw_df.index,
    )
    approved = probabilities >= 0.5
    return featured_df, approved, probabilities


def predict_single(applicant: dict) -> dict:
    """
    Predict loan approval for a single applicant.

    Parameters
    ----------
    applicant : dict
        Raw applicant fields.  All keys in REQUIRED_RAW_FIELDS must be
        present.  Extra keys are silently ignored.

    Returns
    -------
    dict with keys:
        approved            (bool)  – True if the model predicts approval.
        approval_probability (float) – Probability of approval (0–1).
        emi                 (float) – Estimated monthly instalment, INR.
        dti                 (float) – Debt-to-income ratio.
        lti                 (float) – Loan-to-income ratio.

    Raises
    ------
    ValueError
        If any required field is missing from *applicant*.
    """
    _validate_fields(applicant)
    raw_df = _applicant_to_dataframe(applicant)
    featured_df, approved, probabilities = _run_pipeline(raw_df)

    return {
        "approved": bool(approved.iloc[0]),
        "approval_probability": round(float(probabilities.iloc[0]), 4),
        "emi": round(float(featured_df["emi"].iloc[0]), 2),
        "dti": round(float(featured_df["dti"].iloc[0]), 4),
        "lti": round(float(featured_df["lti"].iloc[0]), 4),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict loan approval for a batch of applicants.

    Parameters
    ----------
    df : pd.DataFrame
        Each row is one applicant.  All columns in REQUIRED_RAW_FIELDS must
        be present.  Extra columns (including a Loan_Status target, if
        accidentally included) are silently ignored.

    Returns
    -------
    pd.DataFrame
        The original DataFrame extended with five new columns:
        ``approved``, ``approval_probability``, ``emi``, ``dti``, ``lti``.
        Row order and index are preserved.

    Raises
    ------
    ValueError
        If any required column is absent from *df*.
    """
    missing_columns = [col for col in REQUIRED_RAW_FIELDS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The following required columns are missing from the DataFrame: "
            f"{missing_columns}. "
            f"All of {REQUIRED_RAW_FIELDS} must be present."
        )

    raw_df = df[REQUIRED_RAW_FIELDS].copy()
    featured_df, approved, probabilities = _run_pipeline(raw_df)

    result = df.copy()
    result["approved"] = approved.values
    result["approval_probability"] = probabilities.round(4).values
    result["emi"] = featured_df["emi"].round(2).values
    result["dti"] = featured_df["dti"].round(4).values
    result["lti"] = featured_df["lti"].round(4).values

    return result


def predict_what_if(base_applicant: dict, overrides: dict) -> dict:
    """
    Predict approval after selectively overriding fields on a base applicant.

    Useful for scenario analysis: a caller can hold all fields constant and
    vary a single field (e.g. increase LoanAmount) to observe how the
    prediction changes.  The merge is shallow — override values replace
    base values key-by-key; nested structures are not supported.

    Parameters
    ----------
    base_applicant : dict
        Complete applicant record.  Must pass validation after merging with
        *overrides*.
    overrides : dict
        Partial field dictionary whose values replace the corresponding
        fields in *base_applicant*.  Keys not in REQUIRED_RAW_FIELDS are
        accepted but have no effect on the model (they are ignored during
        validation and inference).

    Returns
    -------
    dict
        Same structure as predict_single:
        {approved, approval_probability, emi, dti, lti}.

    Raises
    ------
    ValueError
        If the merged applicant record is missing any required field.
    """
    merged_applicant = {**base_applicant, **overrides}
    return predict_single(merged_applicant)
