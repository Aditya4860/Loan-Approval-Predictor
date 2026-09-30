"""
Unit tests for src/predictor.py — predict_single, predict_batch, predict_what_if.
"""

import pytest
import pandas as pd

from src.predictor import predict_single, predict_batch, predict_what_if, REQUIRED_RAW_FIELDS

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

EXPECTED_RESULT_KEYS = {"approved", "approval_probability", "emi", "dti", "lti"}


class TestPredictSingleReturnShape:
    def test_result_contains_all_expected_keys(self):
        result = predict_single(GOOD_APPLICANT)
        assert EXPECTED_RESULT_KEYS == set(result.keys())

    def test_approved_is_bool(self):
        result = predict_single(GOOD_APPLICANT)
        assert isinstance(result["approved"], bool)

    def test_approval_probability_is_float(self):
        result = predict_single(GOOD_APPLICANT)
        assert isinstance(result["approval_probability"], float)

    def test_emi_is_float(self):
        result = predict_single(GOOD_APPLICANT)
        assert isinstance(result["emi"], float)

    def test_dti_is_float(self):
        result = predict_single(GOOD_APPLICANT)
        assert isinstance(result["dti"], float)

    def test_lti_is_float(self):
        result = predict_single(GOOD_APPLICANT)
        assert isinstance(result["lti"], float)


class TestPredictSingleValueRanges:
    def test_approval_probability_is_between_zero_and_one(self):
        result = predict_single(GOOD_APPLICANT)
        assert 0.0 <= result["approval_probability"] <= 1.0

    def test_emi_is_positive(self):
        result = predict_single(GOOD_APPLICANT)
        assert result["emi"] > 0

    def test_dti_is_positive(self):
        result = predict_single(GOOD_APPLICANT)
        assert result["dti"] > 0

    def test_lti_is_positive(self):
        result = predict_single(GOOD_APPLICANT)
        assert result["lti"] > 0

    def test_good_applicant_has_higher_probability_than_risky(self):
        good_prob = predict_single(GOOD_APPLICANT)["approval_probability"]
        risky_prob = predict_single(RISKY_APPLICANT)["approval_probability"]
        assert good_prob > risky_prob

    def test_good_applicant_is_approved(self):
        assert predict_single(GOOD_APPLICANT)["approved"] is True

    def test_risky_applicant_is_rejected(self):
        assert predict_single(RISKY_APPLICANT)["approved"] is False


class TestPredictSingleValidation:
    def test_missing_single_field_raises_value_error(self):
        incomplete = {k: v for k, v in GOOD_APPLICANT.items() if k != "LoanAmount"}
        with pytest.raises(ValueError, match="LoanAmount"):
            predict_single(incomplete)

    def test_empty_dict_raises_value_error_listing_all_fields(self):
        with pytest.raises(ValueError):
            predict_single({})

    def test_extra_keys_are_silently_ignored(self):
        applicant_with_extra = {**GOOD_APPLICANT, "UnknownField": "some_value"}
        result = predict_single(applicant_with_extra)
        assert EXPECTED_RESULT_KEYS == set(result.keys())


class TestPredictBatch:
    def test_output_row_count_matches_input(self):
        batch_df = pd.DataFrame([GOOD_APPLICANT, RISKY_APPLICANT])
        result = predict_batch(batch_df)
        assert len(result) == 2

    def test_output_adds_all_expected_columns(self):
        batch_df = pd.DataFrame([GOOD_APPLICANT, RISKY_APPLICANT])
        result = predict_batch(batch_df)
        for col in ["approved", "approval_probability", "emi", "dti", "lti"]:
            assert col in result.columns

    def test_original_columns_are_preserved(self):
        batch_df = pd.DataFrame([GOOD_APPLICANT, RISKY_APPLICANT])
        result = predict_batch(batch_df)
        for col in batch_df.columns:
            assert col in result.columns

    def test_row_order_is_preserved(self):
        batch_df = pd.DataFrame([GOOD_APPLICANT, RISKY_APPLICANT])
        result = predict_batch(batch_df)
        single_good = predict_single(GOOD_APPLICANT)["approval_probability"]
        single_risky = predict_single(RISKY_APPLICANT)["approval_probability"]
        assert result["approval_probability"].iloc[0] == pytest.approx(single_good)
        assert result["approval_probability"].iloc[1] == pytest.approx(single_risky)

    def test_approved_column_is_boolean_dtype(self):
        batch_df = pd.DataFrame([GOOD_APPLICANT, RISKY_APPLICANT])
        result = predict_batch(batch_df)
        assert result["approved"].dtype == bool

    def test_large_batch_preserves_row_count(self):
        rows = [GOOD_APPLICANT if i % 2 == 0 else RISKY_APPLICANT for i in range(50)]
        batch_df = pd.DataFrame(rows)
        result = predict_batch(batch_df)
        assert len(result) == 50

    def test_missing_column_raises_value_error(self):
        bad_df = pd.DataFrame([GOOD_APPLICANT]).drop(columns=["LoanAmount"])
        with pytest.raises(ValueError, match="LoanAmount"):
            predict_batch(bad_df)

    def test_extra_columns_in_input_are_preserved_in_output(self):
        df_with_extra = pd.DataFrame([{**GOOD_APPLICANT, "applicant_id": "A001"}])
        result = predict_batch(df_with_extra)
        assert "applicant_id" in result.columns


class TestPredictWhatIf:
    def test_empty_overrides_matches_predict_single(self):
        whatif_result = predict_what_if(GOOD_APPLICANT, {})
        single_result = predict_single(GOOD_APPLICANT)
        assert whatif_result["approved"] == single_result["approved"]
        assert whatif_result["approval_probability"] == pytest.approx(
            single_result["approval_probability"]
        )

    def test_all_expected_keys_present(self):
        result = predict_what_if(GOOD_APPLICANT, {"LoanAmount": 300.0})
        assert EXPECTED_RESULT_KEYS == set(result.keys())

    def test_increasing_loan_amount_changes_probability(self):
        base = predict_what_if(GOOD_APPLICANT, {})
        high_loan = predict_what_if(GOOD_APPLICANT, {"LoanAmount": 1500.0})
        assert base["approval_probability"] != high_loan["approval_probability"]

    def test_overrides_take_precedence_over_base_values(self):
        result_with_good_credit = predict_what_if(GOOD_APPLICANT, {"Credit_History": 1.0})
        result_with_bad_credit = predict_what_if(GOOD_APPLICANT, {"Credit_History": 0.0})
        assert (
            result_with_good_credit["approval_probability"]
            > result_with_bad_credit["approval_probability"]
        )

    def test_base_applicant_is_not_mutated_by_overrides(self):
        original_loan = GOOD_APPLICANT["LoanAmount"]
        predict_what_if(GOOD_APPLICANT, {"LoanAmount": 999.0})
        assert GOOD_APPLICANT["LoanAmount"] == original_loan

    def test_missing_required_field_in_merged_record_raises_value_error(self):
        incomplete_base = {k: v for k, v in GOOD_APPLICANT.items() if k != "LoanAmount"}
        with pytest.raises(ValueError, match="LoanAmount"):
            predict_what_if(incomplete_base, {})
