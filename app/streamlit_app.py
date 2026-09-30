"""
Streamlit frontend for the Loan Approval Predictor.

Tabs
----
1. Single Entry   – predict a single applicant and display risk metrics.
2. Batch CSV      – upload a CSV, predict all rows, download results.
3. What-If        – vary one field across a range and plot probability sensitivity.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from src.predictor import predict_batch, predict_single, predict_what_if

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d27;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        color: #9ca3af;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #2563eb !important;
        color: white !important;
    }

    .approve-badge {
        display: inline-block;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 12px 32px;
        border-radius: 50px;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 20px rgba(16,185,129,0.35);
    }
    .reject-badge {
        display: inline-block;
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 12px 32px;
        border-radius: 50px;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 20px rgba(239,68,68,0.35);
    }
    .metric-card {
        background: #1a1d27;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #2d3148;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f9fafb;
    }
    .metric-warn { color: #f59e0b !important; }
    .metric-ok   { color: #10b981 !important; }
    .threshold-tag {
        font-size: 0.72rem;
        margin-top: 4px;
        color: #6b7280;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 4px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🏦 Loan Approval Predictor")
st.markdown(
    "<span style='color:#9ca3af;font-size:0.95rem'>"
    "ML-powered credit-risk assessment · GradientBoosting · ROC-AUC 0.91"
    "</span>",
    unsafe_allow_html=True,
)
st.divider()

FIELD_CONFIGS = {
    "ApplicantIncome": {
        "label": "Applicant Monthly Income (₹)",
        "min": 1000, "max": 100000, "default": 8000, "step": 500, "type": "number",
    },
    "CoapplicantIncome": {
        "label": "Co-Applicant Monthly Income (₹)",
        "min": 0, "max": 80000, "default": 0, "step": 500, "type": "number",
    },
    "LoanAmount": {
        "label": "Loan Amount (₹ thousands)",
        "min": 10, "max": 2000, "default": 150, "step": 10, "type": "number",
    },
    "Loan_Amount_Term": {
        "label": "Loan Term (months)",
        "options": [120, 180, 240, 300, 360, 480], "default": 360, "type": "select",
    },
    "Credit_History": {
        "label": "Credit History",
        "options": {"Good (1.0)": 1.0, "Bad (0.0)": 0.0}, "default": "Good (1.0)", "type": "select_map",
    },
    "Employment_Status": {
        "label": "Employment Status",
        "options": ["Salaried", "Self_Employed"], "default": "Salaried", "type": "select",
    },
    "Marital_Status": {
        "label": "Marital Status",
        "options": ["Yes", "No"], "default": "Yes", "type": "select",
    },
    "Dependents": {
        "label": "Dependents",
        "options": ["0", "1", "2", "3+"], "default": "0", "type": "select",
    },
    "Education": {
        "label": "Education",
        "options": ["Graduate", "Not Graduate"], "default": "Graduate", "type": "select",
    },
    "Property_Area": {
        "label": "Property Area",
        "options": ["Urban", "Semiurban", "Rural"], "default": "Semiurban", "type": "select",
    },
}

NUMERIC_VARIABLE_FIELDS = {
    "Loan Amount (₹ thousands)": "LoanAmount",
    "Applicant Income (₹/month)": "ApplicantIncome",
    "Co-Applicant Income (₹/month)": "CoapplicantIncome",
}

DTI_WARN_THRESHOLD = 0.43
LTI_WARN_THRESHOLD = 5.0


def render_applicant_form(form_key: str) -> dict:
    """
    Render all applicant input widgets and return the raw field dict.

    Parameters
    ----------
    form_key : str
        Unique prefix for widget keys to allow multiple forms on one page.

    Returns
    -------
    dict
        Raw applicant fields ready to pass to any predictor function.
    """
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>💰 Financial Details</div>", unsafe_allow_html=True)
        applicant_income = st.number_input(
            FIELD_CONFIGS["ApplicantIncome"]["label"],
            min_value=FIELD_CONFIGS["ApplicantIncome"]["min"],
            max_value=FIELD_CONFIGS["ApplicantIncome"]["max"],
            value=FIELD_CONFIGS["ApplicantIncome"]["default"],
            step=FIELD_CONFIGS["ApplicantIncome"]["step"],
            key=f"{form_key}_applicant_income",
        )
        coapplicant_income = st.number_input(
            FIELD_CONFIGS["CoapplicantIncome"]["label"],
            min_value=FIELD_CONFIGS["CoapplicantIncome"]["min"],
            max_value=FIELD_CONFIGS["CoapplicantIncome"]["max"],
            value=FIELD_CONFIGS["CoapplicantIncome"]["default"],
            step=FIELD_CONFIGS["CoapplicantIncome"]["step"],
            key=f"{form_key}_coapplicant_income",
        )
        loan_amount = st.number_input(
            FIELD_CONFIGS["LoanAmount"]["label"],
            min_value=FIELD_CONFIGS["LoanAmount"]["min"],
            max_value=FIELD_CONFIGS["LoanAmount"]["max"],
            value=FIELD_CONFIGS["LoanAmount"]["default"],
            step=FIELD_CONFIGS["LoanAmount"]["step"],
            key=f"{form_key}_loan_amount",
        )
        loan_term = st.selectbox(
            FIELD_CONFIGS["Loan_Amount_Term"]["label"],
            options=FIELD_CONFIGS["Loan_Amount_Term"]["options"],
            index=FIELD_CONFIGS["Loan_Amount_Term"]["options"].index(360),
            key=f"{form_key}_loan_term",
        )

    with col_b:
        st.markdown("<div class='section-header'>👤 Personal Details</div>", unsafe_allow_html=True)
        credit_history_label = st.selectbox(
            FIELD_CONFIGS["Credit_History"]["label"],
            options=list(FIELD_CONFIGS["Credit_History"]["options"].keys()),
            key=f"{form_key}_credit_history",
        )
        credit_history = FIELD_CONFIGS["Credit_History"]["options"][credit_history_label]

        employment_status = st.selectbox(
            FIELD_CONFIGS["Employment_Status"]["label"],
            options=FIELD_CONFIGS["Employment_Status"]["options"],
            key=f"{form_key}_employment_status",
        )
        marital_status = st.selectbox(
            FIELD_CONFIGS["Marital_Status"]["label"],
            options=FIELD_CONFIGS["Marital_Status"]["options"],
            key=f"{form_key}_marital_status",
        )
        dependents = st.selectbox(
            FIELD_CONFIGS["Dependents"]["label"],
            options=FIELD_CONFIGS["Dependents"]["options"],
            key=f"{form_key}_dependents",
        )
        education = st.selectbox(
            FIELD_CONFIGS["Education"]["label"],
            options=FIELD_CONFIGS["Education"]["options"],
            key=f"{form_key}_education",
        )
        property_area = st.selectbox(
            FIELD_CONFIGS["Property_Area"]["label"],
            options=FIELD_CONFIGS["Property_Area"]["options"],
            key=f"{form_key}_property_area",
        )

    return {
        "ApplicantIncome": float(applicant_income),
        "CoapplicantIncome": float(coapplicant_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": int(loan_term),
        "Credit_History": float(credit_history),
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Education": education,
        "Property_Area": property_area,
    }


def render_risk_metrics(result: dict) -> None:
    """
    Display EMI, DTI, and LTI as styled metric cards with threshold warnings.

    Parameters
    ----------
    result : dict
        Output from any predict_* function containing emi, dti, lti keys.
    """
    col1, col2, col3 = st.columns(3)

    dti_warn = result["dti"] > DTI_WARN_THRESHOLD
    lti_warn = result["lti"] > LTI_WARN_THRESHOLD

    with col1:
        dti_class = "metric-warn" if dti_warn else "metric-ok"
        dti_tag = f"⚠️ Exceeds {DTI_WARN_THRESHOLD} threshold" if dti_warn else f"✓ Within {DTI_WARN_THRESHOLD} limit"
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Debt-to-Income (DTI)</div>
                <div class='metric-value {dti_class}'>{result['dti']:.3f}</div>
                <div class='threshold-tag'>{dti_tag}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        lti_class = "metric-warn" if lti_warn else "metric-ok"
        lti_tag = f"⚠️ Exceeds {LTI_WARN_THRESHOLD}x threshold" if lti_warn else f"✓ Within {LTI_WARN_THRESHOLD}x limit"
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Loan-to-Income (LTI)</div>
                <div class='metric-value {lti_class}'>{result['lti']:.3f}</div>
                <div class='threshold-tag'>{lti_tag}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Est. Monthly Instalment (EMI)</div>
                <div class='metric-value' style='color:#f9fafb'>₹{result['emi']:,.0f}</div>
                <div class='threshold-tag'>Principal-only approximation</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_decision_badge(result: dict) -> None:
    """
    Display the approval decision badge and probability meter.

    Parameters
    ----------
    result : dict
        Output from any predict_* function.
    """
    badge_class = "approve-badge" if result["approved"] else "reject-badge"
    badge_text = "✅ APPROVED" if result["approved"] else "❌ REJECTED"
    prob_pct = result["approval_probability"] * 100
    bar_color = "#10b981" if result["approved"] else "#ef4444"

    st.markdown(
        f"<div style='text-align:center;padding:24px 0 8px'>"
        f"<span class='{badge_class}'>{badge_text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;color:#9ca3af;margin-top:12px;font-size:0.95rem'>"
        f"Approval Probability: <strong style='color:#f9fafb'>{prob_pct:.1f}%</strong></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style='background:#2d3148;border-radius:100px;height:10px;overflow:hidden;margin:0 auto;max-width:400px'>
            <div style='background:{bar_color};width:{prob_pct:.1f}%;height:100%;border-radius:100px;
                        transition:width 0.5s ease'></div>
        </div>""",
        unsafe_allow_html=True,
    )


tab_single, tab_batch, tab_whatif = st.tabs(
    ["📋  Single Entry", "📂  Batch CSV", "🔬  What-If Analysis"]
)


with tab_single:
    st.markdown("### Applicant Details")
    applicant = render_applicant_form("single")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  Predict Approval", type="primary", key="single_predict_btn"):
        with st.spinner("Running model..."):
            result = predict_single(applicant)

        st.markdown("---")
        st.markdown("### Decision")
        render_decision_badge(result)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Risk Metrics")
        render_risk_metrics(result)


with tab_batch:
    st.markdown("### Upload Applicant CSV")
    st.markdown(
        "<span style='color:#9ca3af;font-size:0.88rem'>"
        "CSV must contain these columns: "
        "ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, "
        "Credit_History, Employment_Status, Marital_Status, Dependents, Education, Property_Area"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.markdown(f"**{len(raw_df):,} rows detected.** Running batch prediction...")

            with st.spinner("Processing..."):
                results_df = predict_batch(raw_df)

            approved_count = results_df["approved"].sum()
            rejected_count = len(results_df) - approved_count

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Total Applications", f"{len(results_df):,}")
            metric_col2.metric("Approved", f"{approved_count:,}", delta=f"{approved_count/len(results_df)*100:.1f}%")
            metric_col3.metric("Rejected", f"{rejected_count:,}", delta=f"-{rejected_count/len(results_df)*100:.1f}%", delta_color="inverse")

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                results_df.style.applymap(
                    lambda v: "color: #10b981; font-weight:600" if v is True else
                              ("color: #ef4444; font-weight:600" if v is False else ""),
                    subset=["approved"],
                ),
                use_container_width=True,
                height=420,
            )

            csv_output = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Download Results as CSV",
                data=csv_output,
                file_name="loan_predictions.csv",
                mime="text/csv",
                type="primary",
            )

        except ValueError as validation_error:
            st.error(f"**Schema Error:** {validation_error}")
        except Exception as unexpected_error:
            st.error(f"**Unexpected Error:** {unexpected_error}")


with tab_whatif:
    st.markdown("### Base Applicant")
    st.markdown(
        "<span style='color:#9ca3af;font-size:0.88rem'>"
        "Set a base profile, then choose a field to vary and observe how approval probability shifts."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    base_applicant = render_applicant_form("whatif")

    st.markdown("---")
    st.markdown("### Sensitivity Controls")

    control_col1, control_col2 = st.columns([1, 2])

    with control_col1:
        varied_field_label = st.selectbox(
            "Field to vary",
            options=list(NUMERIC_VARIABLE_FIELDS.keys()),
            key="whatif_varied_field",
        )
        varied_field_key = NUMERIC_VARIABLE_FIELDS[varied_field_label]
        num_points = st.slider("Number of steps", min_value=10, max_value=100, value=40, key="whatif_steps")

    field_ranges = {
        "LoanAmount": (10, 2000),
        "ApplicantIncome": (1000, 100000),
        "CoapplicantIncome": (0, 80000),
    }
    range_min, range_max = field_ranges[varied_field_key]

    with control_col2:
        selected_range = st.slider(
            f"Range for {varied_field_label}",
            min_value=float(range_min),
            max_value=float(range_max),
            value=(float(range_min), float(range_max)),
            step=float((range_max - range_min) / 200),
            key="whatif_range",
        )

    if st.button("📈  Run Sensitivity Analysis", type="primary", key="whatif_run_btn"):
        varied_values = np.linspace(selected_range[0], selected_range[1], num=num_points)

        probabilities = []
        for value in varied_values:
            try:
                result = predict_what_if(base_applicant, {varied_field_key: float(value)})
                probabilities.append(result["approval_probability"])
            except Exception:
                probabilities.append(None)

        sensitivity_df = pd.DataFrame(
            {
                varied_field_label: varied_values,
                "Approval Probability": probabilities,
            }
        ).dropna()

        base_prob = predict_single(base_applicant)["approval_probability"]

        st.markdown(
            f"<p style='color:#9ca3af;font-size:0.88rem;margin-top:8px'>"
            f"Base approval probability (current inputs): "
            f"<strong style='color:#f9fafb'>{base_prob*100:.1f}%</strong></p>",
            unsafe_allow_html=True,
        )

        chart_df = sensitivity_df.set_index(varied_field_label)
        st.line_chart(chart_df, use_container_width=True, height=380)

        crossover_rows = sensitivity_df[
            (sensitivity_df["Approval Probability"] >= 0.499) &
            (sensitivity_df["Approval Probability"] <= 0.501)
        ]
        if not crossover_rows.empty:
            crossover_value = crossover_rows.iloc[0][varied_field_label]
            st.info(
                f"**Decision boundary** for {varied_field_label} ≈ "
                f"{crossover_value:,.0f}  (probability crosses 50%)"
            )
