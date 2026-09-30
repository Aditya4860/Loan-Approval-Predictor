"""
Unit tests for src/data_processing.py — prepare_data().
"""

import pytest
import pandas as pd
import numpy as np

from src.data_processing import (
    prepare_data,
    NUMERIC_COLUMNS,
    NOMINAL_COLUMNS,
    impute_missing_values,
    encode_target,
    build_preprocessing_pipeline,
    get_feature_names,
)


TOTAL_ROWS = 5000
TRAIN_FRACTION = 0.80
EXPECTED_TRAIN_ROWS = int(TOTAL_ROWS * TRAIN_FRACTION)
EXPECTED_TEST_ROWS = TOTAL_ROWS - EXPECTED_TRAIN_ROWS
TOLERANCE = 5


@pytest.fixture(scope="module")
def pipeline_outputs():
    """Run prepare_data once for the whole module; expensive I/O happens only once."""
    X_train, X_test, y_train, y_test = prepare_data()
    return X_train, X_test, y_train, y_test


class TestSplitShapes:
    def test_train_row_count_is_approximately_80_percent(self, pipeline_outputs):
        X_train, _, _, _ = pipeline_outputs
        assert abs(len(X_train) - EXPECTED_TRAIN_ROWS) <= TOLERANCE

    def test_test_row_count_is_approximately_20_percent(self, pipeline_outputs):
        _, X_test, _, _ = pipeline_outputs
        assert abs(len(X_test) - EXPECTED_TEST_ROWS) <= TOLERANCE

    def test_train_and_test_rows_sum_to_total(self, pipeline_outputs):
        X_train, X_test, _, _ = pipeline_outputs
        assert len(X_train) + len(X_test) == TOTAL_ROWS

    def test_train_and_test_have_equal_feature_counts(self, pipeline_outputs):
        X_train, X_test, _, _ = pipeline_outputs
        assert X_train.shape[1] == X_test.shape[1]

    def test_feature_count_is_nonzero(self, pipeline_outputs):
        X_train, _, _, _ = pipeline_outputs
        assert X_train.shape[1] > 0

    def test_target_train_length_matches_feature_train_length(self, pipeline_outputs):
        X_train, _, y_train, _ = pipeline_outputs
        assert len(X_train) == len(y_train)

    def test_target_test_length_matches_feature_test_length(self, pipeline_outputs):
        _, X_test, _, y_test = pipeline_outputs
        assert len(X_test) == len(y_test)


class TestNoNaNsAfterPreprocessing:
    def test_X_train_has_no_nan_values(self, pipeline_outputs):
        X_train, _, _, _ = pipeline_outputs
        assert not X_train.isnull().any().any()

    def test_X_test_has_no_nan_values(self, pipeline_outputs):
        _, X_test, _, _ = pipeline_outputs
        assert not X_test.isnull().any().any()

    def test_y_train_has_no_nan_values(self, pipeline_outputs):
        _, _, y_train, _ = pipeline_outputs
        assert not y_train.isnull().any()

    def test_y_test_has_no_nan_values(self, pipeline_outputs):
        _, _, _, y_test = pipeline_outputs
        assert not y_test.isnull().any()


class TestTargetEncoding:
    def test_y_train_contains_only_zeros_and_ones(self, pipeline_outputs):
        _, _, y_train, _ = pipeline_outputs
        assert set(y_train.unique()).issubset({0, 1})

    def test_y_test_contains_only_zeros_and_ones(self, pipeline_outputs):
        _, _, _, y_test = pipeline_outputs
        assert set(y_test.unique()).issubset({0, 1})

    def test_both_classes_present_in_train(self, pipeline_outputs):
        _, _, y_train, _ = pipeline_outputs
        assert 0 in y_train.values
        assert 1 in y_train.values

    def test_both_classes_present_in_test(self, pipeline_outputs):
        _, _, _, y_test = pipeline_outputs
        assert 0 in y_test.values
        assert 1 in y_test.values


class TestStratifiedBalance:
    def test_train_approval_rate_close_to_dataset_rate(self, pipeline_outputs):
        _, _, y_train, y_test = pipeline_outputs
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.02


class TestFeatureNames:
    def test_X_train_is_dataframe_with_named_columns(self, pipeline_outputs):
        X_train, _, _, _ = pipeline_outputs
        assert isinstance(X_train, pd.DataFrame)
        assert len(X_train.columns) > 0

    def test_numeric_columns_present_in_features(self, pipeline_outputs):
        X_train, _, _, _ = pipeline_outputs
        for col in NUMERIC_COLUMNS:
            assert col in X_train.columns

    def test_nominal_columns_not_present_as_raw_in_features(self, pipeline_outputs):
        X_train, _, _, _ = pipeline_outputs
        for col in NOMINAL_COLUMNS:
            assert col not in X_train.columns


class TestImputeMissingValues:
    def test_no_nulls_remain_after_imputation(self):
        df = pd.DataFrame({
            "LoanAmount": [100.0, None, 200.0],
            "Credit_History": [1.0, None, 0.0],
            "Employment_Status": ["Salaried", None, "Self_Employed"],
            "Marital_Status": ["Yes", "No", None],
            "Dependents": ["0", "1", None],
            "Education": ["Graduate", None, "Graduate"],
            "Property_Area": ["Urban", "Rural", None],
        })
        result = impute_missing_values(df)
        assert result.isnull().sum().sum() == 0

    def test_loan_amount_imputed_with_median(self):
        df = pd.DataFrame({
            "LoanAmount": [100.0, None, 300.0],
            "Credit_History": [1.0, 1.0, 1.0],
            "Employment_Status": ["Salaried"] * 3,
            "Marital_Status": ["Yes"] * 3,
            "Dependents": ["0"] * 3,
            "Education": ["Graduate"] * 3,
            "Property_Area": ["Urban"] * 3,
        })
        result = impute_missing_values(df)
        assert result["LoanAmount"].iloc[1] == pytest.approx(200.0)


class TestEncodeTarget:
    def test_y_maps_to_one(self):
        series = pd.Series(["Y", "N", "Y"])
        result = encode_target(series)
        assert result.iloc[0] == 1

    def test_n_maps_to_zero(self):
        series = pd.Series(["Y", "N", "Y"])
        result = encode_target(series)
        assert result.iloc[1] == 0


class TestGetFeatureNames:
    def test_returns_list(self):
        pipeline = build_preprocessing_pipeline()
        sample_df = pd.DataFrame([{
            "ApplicantIncome": 8000.0, "CoapplicantIncome": 2000.0,
            "LoanAmount": 150.0, "Loan_Amount_Term": 360,
            "Credit_History": 1.0, "emi": 416.67, "dti": 0.5, "lti": 0.015,
            "Employment_Status": "Salaried", "Marital_Status": "Yes",
            "Dependents": "1", "Education": "Graduate", "Property_Area": "Semiurban",
        }])
        pipeline.fit(sample_df)
        names = get_feature_names(pipeline)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_numeric_columns_appear_first(self):
        pipeline = build_preprocessing_pipeline()
        sample_df = pd.DataFrame([{
            "ApplicantIncome": 8000.0, "CoapplicantIncome": 2000.0,
            "LoanAmount": 150.0, "Loan_Amount_Term": 360,
            "Credit_History": 1.0, "emi": 416.67, "dti": 0.5, "lti": 0.015,
            "Employment_Status": "Salaried", "Marital_Status": "Yes",
            "Dependents": "1", "Education": "Graduate", "Property_Area": "Semiurban",
        }])
        pipeline.fit(sample_df)
        names = get_feature_names(pipeline)
        for i, col in enumerate(NUMERIC_COLUMNS):
            assert names[i] == col
