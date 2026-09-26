"""
Preprocessing pipeline for the loan-approval dataset.

Imputation strategy
-------------------
Numeric columns (LoanAmount, Credit_History):
    Median imputation is chosen over mean because both columns are
    right-skewed and contain outliers (very high loan amounts, lump of
    credit-history values at 0.0 vs 1.0).  The median is robust to
    these extremes and preserves the central tendency without inflating
    it.  For the binary Credit_History column the median coincides with
    the majority class (1.0), which is also what a domain expert would
    assign as a conservative default.

Categorical columns (Employment_Status, Marital_Status, Dependents,
Education, Property_Area):
    Mode imputation is used because the missing-at-random assumption
    holds (missingness was introduced uniformly in generation), and the
    mode is the only statistically sensible point-estimate for nominal
    data.  Mean or median have no meaning for unordered categories.

Encoding
--------
All five nominal categoricals are one-hot encoded with
``drop='first'`` to avoid perfect multicollinearity (dummy-variable
trap) in linear models.  Tree-based models are unaffected by this
choice.  ``handle_unknown='ignore'`` ensures that unseen category
values at inference time (e.g. new property areas) silently produce
an all-zero row rather than raising an error.

Scaling
-------
StandardScaler (zero mean, unit variance) is applied to every numeric
column, including the engineered risk features.  This is necessary for
regularised linear models (Logistic Regression, SVM) and k-NN; gradient
boosting is scale-invariant but is not harmed by it.  The scaler is
fitted exclusively on the training split to prevent data leakage.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.risk_metrics import build_risk_features

RAW_DATA_PATH = "data/raw/loan_data.csv"
PROCESSED_DIR = "data/processed"
PIPELINE_PATH = "models/preprocessor.joblib"

TARGET_COLUMN = "Loan_Status"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "emi",
    "dti",
    "lti",
]

NOMINAL_COLUMNS = [
    "Employment_Status",
    "Marital_Status",
    "Dependents",
    "Education",
    "Property_Area",
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    return pd.read_csv(path)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values before feature engineering.

    Numeric columns receive the column median; categorical columns
    receive the column mode.  See module docstring for the full
    justification of these choices.
    """
    df = df.copy()

    numeric_cols_with_nulls = ["LoanAmount", "Credit_History"]
    for col in numeric_cols_with_nulls:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = NOMINAL_COLUMNS
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_target(series: pd.Series) -> pd.Series:
    """Map 'Y' -> 1 and 'N' -> 0."""
    return series.map({"Y": 1, "N": 0})


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Construct and return an unfitted sklearn preprocessing pipeline.

    The pipeline is a ColumnTransformer with two branches:
      - numeric_pipeline  : StandardScaler (imputation already done)
      - nominal_pipeline  : OneHotEncoder (drop='first', handle_unknown='ignore')

    Returns
    -------
    ColumnTransformer
        An unfitted transformer ready to be fit on training data.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("nominal", nominal_pipeline, NOMINAL_COLUMNS),
        ],
        remainder="drop",
    )

    return preprocessing_pipeline


def get_feature_names(fitted_pipeline: ColumnTransformer) -> list[str]:
    """Return ordered list of feature names after transformation."""
    ohe = fitted_pipeline.named_transformers_["nominal"].named_steps["encoder"]
    ohe_feature_names = ohe.get_feature_names_out(NOMINAL_COLUMNS).tolist()
    return NUMERIC_COLUMNS + ohe_feature_names


def prepare_data(
    raw_path: str = RAW_DATA_PATH,
    processed_dir: str = PROCESSED_DIR,
    pipeline_path: str = PIPELINE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Full preprocessing flow: load, impute, engineer features, encode,
    scale, split, save.

    Steps
    -----
    1. Load raw CSV.
    2. Impute missing values with median/mode (column-specific).
    3. Call build_risk_features() to add emi, dti, lti columns.
    4. Encode the binary target (Y -> 1, N -> 0).
    5. Split into 80/20 train/test, stratified on the target.
    6. Fit the ColumnTransformer on the training split only.
    7. Transform both splits.
    8. Save the fitted pipeline to ``pipeline_path``.
    9. Save processed splits as CSVs to ``processed_dir``.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.
    processed_dir : str
        Directory where processed CSVs are saved.
    pipeline_path : str
        Path where the fitted pipeline is persisted via joblib.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Feature matrices with named columns.
    y_train, y_test : pd.Series
        Binary target arrays (0 = rejected, 1 = approved).
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)

    raw_df = load_raw_data(raw_path)
    imputed_df = impute_missing_values(raw_df)
    featured_df = build_risk_features(imputed_df)

    y = encode_target(featured_df[TARGET_COLUMN])
    X = featured_df.drop(columns=[TARGET_COLUMN])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessing_pipeline = build_preprocessing_pipeline()
    X_train_arr = preprocessing_pipeline.fit_transform(X_train_raw)
    X_test_arr = preprocessing_pipeline.transform(X_test_raw)

    feature_names = get_feature_names(preprocessing_pipeline)
    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_raw.index)

    joblib.dump(preprocessing_pipeline, pipeline_path)

    X_train.assign(**{TARGET_COLUMN: y_train}).to_csv(
        os.path.join(processed_dir, "train.csv"), index=True
    )
    X_test.assign(**{TARGET_COLUMN: y_test}).to_csv(
        os.path.join(processed_dir, "test.csv"), index=True
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Training set  : {X_train.shape}")
    print(f"Test set      : {X_test.shape}")
    print(f"Features      : {X_train.columns.tolist()}")
    print(f"Target balance (train):\n{y_train.value_counts(normalize=True)}")
    print(f"Pipeline saved to {PIPELINE_PATH}")
