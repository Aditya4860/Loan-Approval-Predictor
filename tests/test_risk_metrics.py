"""
Unit tests for src/risk_metrics.py — build_risk_features().
"""

import pytest
import pandas as pd

from src.risk_metrics import (
    build_risk_features,
    LOAN_AMOUNT_SCALE,
    MONTHLY_INCOME_DIVISOR,
)


def make_single_row(
    applicant_income=10000.0,
    coapplicant_income=2000.0,
    loan_amount=200.0,
    loan_amount_term=360,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
    }])


class TestEmiCalculation:
    def test_emi_formula_matches_manual_computation(self):
        df = make_single_row(loan_amount=300.0, loan_amount_term=360)
        result = build_risk_features(df)
        expected_emi = (300.0 * LOAN_AMOUNT_SCALE) / 360
        assert result["emi"].iloc[0] == pytest.approx(expected_emi)

    def test_emi_short_term_is_higher_than_long_term(self):
        short_term = build_risk_features(make_single_row(loan_amount_term=120))
        long_term = build_risk_features(make_single_row(loan_amount_term=360))
        assert short_term["emi"].iloc[0] > long_term["emi"].iloc[0]

    def test_emi_scales_linearly_with_loan_amount(self):
        small = build_risk_features(make_single_row(loan_amount=100.0))
        large = build_risk_features(make_single_row(loan_amount=200.0))
        assert large["emi"].iloc[0] == pytest.approx(2 * small["emi"].iloc[0])


class TestDtiCalculation:
    def test_dti_formula_matches_manual_computation(self):
        df = make_single_row(
            applicant_income=10000.0,
            coapplicant_income=2000.0,
            loan_amount=200.0,
            loan_amount_term=360,
        )
        result = build_risk_features(df)
        expected_emi = (200.0 * LOAN_AMOUNT_SCALE) / 360
        expected_dti = expected_emi / (12000.0 / MONTHLY_INCOME_DIVISOR)
        assert result["dti"].iloc[0] == pytest.approx(expected_dti)

    def test_dti_decreases_when_income_increases(self):
        low_income = build_risk_features(make_single_row(applicant_income=5000.0))
        high_income = build_risk_features(make_single_row(applicant_income=20000.0))
        assert high_income["dti"].iloc[0] < low_income["dti"].iloc[0]

    def test_dti_is_nan_when_combined_income_is_zero(self):
        df = make_single_row(applicant_income=5000.0, coapplicant_income=-5000.0)
        result = build_risk_features(df)
        assert pd.isna(result["dti"].iloc[0])

    def test_zero_coapplicant_income_still_computes_valid_dti(self):
        df = make_single_row(applicant_income=8000.0, coapplicant_income=0.0)
        result = build_risk_features(df)
        assert not pd.isna(result["dti"].iloc[0])
        assert result["dti"].iloc[0] > 0


class TestLtiCalculation:
    def test_lti_formula_matches_manual_computation(self):
        df = make_single_row(
            applicant_income=10000.0,
            coapplicant_income=2000.0,
            loan_amount=200.0,
        )
        result = build_risk_features(df)
        expected_lti = 200.0 / (10000.0 + 2000.0)
        assert result["lti"].iloc[0] == pytest.approx(expected_lti)

    def test_lti_increases_with_loan_amount(self):
        small = build_risk_features(make_single_row(loan_amount=100.0))
        large = build_risk_features(make_single_row(loan_amount=500.0))
        assert large["lti"].iloc[0] > small["lti"].iloc[0]


class TestOutputSchema:
    def test_output_contains_all_risk_columns(self):
        result = build_risk_features(make_single_row())
        assert {"emi", "dti", "lti"}.issubset(result.columns)

    def test_original_columns_are_preserved(self):
        df = make_single_row()
        result = build_risk_features(df)
        for col in df.columns:
            assert col in result.columns

    def test_input_dataframe_is_not_mutated(self):
        df = make_single_row()
        original_columns = list(df.columns)
        build_risk_features(df)
        assert list(df.columns) == original_columns

    def test_batch_input_preserves_row_count(self):
        rows = [
            make_single_row(applicant_income=5000.0 + i * 1000).iloc[0].to_dict()
            for i in range(10)
        ]
        batch_df = pd.DataFrame(rows)
        result = build_risk_features(batch_df)
        assert len(result) == 10


class TestInvalidInputValidation:
    def test_zero_applicant_income_raises_value_error(self):
        df = make_single_row(applicant_income=0.0)
        with pytest.raises(ValueError, match="ApplicantIncome"):
            build_risk_features(df)

    def test_negative_applicant_income_raises_value_error(self):
        df = make_single_row(applicant_income=-500.0)
        with pytest.raises(ValueError, match="ApplicantIncome"):
            build_risk_features(df)

    def test_zero_loan_term_raises_value_error(self):
        df = make_single_row(loan_amount_term=0)
        with pytest.raises(ValueError, match="Loan_Amount_Term"):
            build_risk_features(df)

    def test_negative_loan_term_raises_value_error(self):
        df = make_single_row(loan_amount_term=-12)
        with pytest.raises(ValueError, match="Loan_Amount_Term"):
            build_risk_features(df)

    def test_error_message_contains_offending_value(self):
        df = make_single_row(applicant_income=-999.0)
        with pytest.raises(ValueError, match="-999"):
            build_risk_features(df)
