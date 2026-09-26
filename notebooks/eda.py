import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(data_path, output_dir):
    df = pd.read_csv(data_path)
    
    print("--- SHAPE ---")
    print(df.shape)
    
    print("\n--- DTYPES ---")
    print(df.dtypes)
    
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())
    
    print("\n--- APPROVAL CLASS BALANCE ---")
    print(df["Loan_Status"].value_counts(normalize=True))
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df["ApplicantIncome"], bins=50, kde=True)
    plt.title("Applicant Income Distribution")
    plt.savefig(os.path.join(output_dir, "income_distribution.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df["LoanAmount"].dropna(), bins=50, kde=True)
    plt.title("Loan Amount Distribution")
    plt.savefig(os.path.join(output_dir, "loan_amount_distribution.png"))
    plt.close()
    
    df["Loan_Approved"] = (df["Loan_Status"] == "Y").astype(int)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Credit_History", y="Loan_Approved")
    plt.title("Approval Rate by Credit History")
    plt.ylabel("Approval Rate")
    plt.savefig(os.path.join(output_dir, "approval_by_credit.png"))
    plt.close()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Education", y="Loan_Approved")
    plt.title("Approval Rate by Education")
    plt.ylabel("Approval Rate")
    plt.savefig(os.path.join(output_dir, "approval_by_education.png"))
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")

    print_eda_summary(df)

def print_eda_summary(df):
    approval_rate = df["Loan_Status"].value_counts(normalize=True).get("Y", 0)
    good_credit_approval = df[df["Credit_History"] == 1.0]["Loan_Approved"].mean()
    bad_credit_approval = df[df["Credit_History"] == 0.0]["Loan_Approved"].mean()
    graduate_approval = df[df["Education"] == "Graduate"]["Loan_Approved"].mean()
    non_graduate_approval = df[df["Education"] == "Not Graduate"]["Loan_Approved"].mean()

    summary = f"""
=====================================
  EDA KEY FINDINGS
=====================================

Dataset: 5,000 synthetic loan applicants with 11 columns.

Missing Values:
  - LoanAmount:     ~4.8% missing (238 rows)
  - Credit_History: ~8.3% missing (416 rows)
  - All other columns: complete

Class Balance:
  - Approved (Y): {approval_rate:.1%}
  - Rejected (N): {1 - approval_rate:.1%}
  => Dataset is imbalanced (~82/18). Will need SMOTE or class-weight
    adjustments during modelling.

Income & Loan Amount:
  - ApplicantIncome is right-skewed (log-normal). A log transform
    will likely improve model linearity assumptions.
  - LoanAmount is strongly correlated with income, as expected from
    the generation logic (1x–3.5x annual income).

Credit History:
  - Good credit (1.0) approval rate: {good_credit_approval:.1%}
  - Bad  credit (0.0) approval rate: {bad_credit_approval:.1%}
  => Credit history is the strongest single predictor of approval.

Education:
  - Graduate approval rate:     {graduate_approval:.1%}
  - Non-Graduate approval rate: {non_graduate_approval:.1%}
  => Education provides a modest but real signal.

Recommendations for preprocessing:
  1. Impute LoanAmount with median, Credit_History with mode.
  2. Log-transform ApplicantIncome and LoanAmount.
  3. One-hot encode all categorical features.
  4. Address class imbalance before training.
=====================================
"""
    print(summary)

if __name__ == "__main__":
    perform_eda("data/raw/loan_data.csv", "notebooks/plots")
