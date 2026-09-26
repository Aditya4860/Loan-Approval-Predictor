import numpy as np
import pandas as pd
import os

def generate_synthetic_loan_data(num_samples=5000):
    np.random.seed(42)
    
    applicant_income = np.random.lognormal(mean=8.5, sigma=0.8, size=num_samples)
    
    has_coapplicant = np.random.choice([True, False], size=num_samples, p=[0.4, 0.6])
    coapplicant_income = np.where(has_coapplicant, np.random.lognormal(mean=8.0, sigma=0.7, size=num_samples), 0.0)
    
    total_income = applicant_income + coapplicant_income
    base_loan = total_income * np.random.uniform(1.0, 3.5, size=num_samples) / 1000.0
    loan_amount = np.maximum(base_loan + np.random.normal(0, 50, size=num_samples), 10.0)
    
    loan_term = np.random.choice([120, 180, 240, 300, 360, 480], size=num_samples, p=[0.05, 0.1, 0.15, 0.1, 0.55, 0.05])
    credit_history = np.random.choice([1.0, 0.0], size=num_samples, p=[0.85, 0.15])
    employment_status = np.random.choice(["Self_Employed", "Salaried"], size=num_samples, p=[0.2, 0.8])
    marital_status = np.random.choice(["Yes", "No"], size=num_samples, p=[0.65, 0.35])
    dependents = np.random.choice(["0", "1", "2", "3+"], size=num_samples, p=[0.55, 0.2, 0.15, 0.1])
    education = np.random.choice(["Graduate", "Not Graduate"], size=num_samples, p=[0.75, 0.25])
    property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=num_samples, p=[0.35, 0.4, 0.25])
    
    emi_estimate = (loan_amount * 1000) / loan_term
    pti_ratio = emi_estimate / (total_income / 12)
    
    score = np.zeros(num_samples)
    score += credit_history * 3.0
    score -= (pti_ratio > 0.4) * 2.0
    score += (education == "Graduate") * 0.5
    score += (employment_status == "Salaried") * 0.5
    score += (property_area == "Semiurban") * 0.3
    
    noise = np.random.normal(0, 1.0, size=num_samples)
    final_score = score + noise
    
    approval_status = np.where(final_score > 1.5, "Y", "N")
    
    df = pd.DataFrame({
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Education": education,
        "Property_Area": property_area,
        "Loan_Status": approval_status
    })
    
    df = df.round({"ApplicantIncome": 0, "CoapplicantIncome": 0, "LoanAmount": 0})
    
    mask_loan_amount = np.random.choice([True, False], size=num_samples, p=[0.05, 0.95])
    df.loc[mask_loan_amount, "LoanAmount"] = np.nan
    
    mask_credit = np.random.choice([True, False], size=num_samples, p=[0.08, 0.92])
    df.loc[mask_credit, "Credit_History"] = np.nan
    
    return df

if __name__ == "__main__":
    output_path = "data/raw/loan_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_loan_data(5000)
    df.to_csv(output_path, index=False)
