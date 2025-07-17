import streamlit as st
import pandas as pd
import shap
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load model from MLflow model registry
@st.cache_resource
def load_model():
    try:
        # Load as sklearn pipeline to access preprocessing steps
        model = mlflow.sklearn.load_model("models:/LGBM_Loan_Default_SQLite/latest")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# App UI
st.title("Loan Default Prediction Dashboard")
st.markdown("""
**Instructions:**  
1. Fill in applicant details  
2. View prediction and SHAP explanation  
""")

# Input form
with st.form("applicant_form"):
    st.subheader("Enter Applicant Information")

    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    home_ownership_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    
    col1, col2 = st.columns(2)
    with col1:
        loan_amnt = st.number_input("Loan Amount", 1000, 40000, 15000)
        grade = st.selectbox("Grade", list(grade_map.keys()))
        mort_acc = st.number_input("Number of Mortgage Accounts", 0, 50, 2)
        term = st.selectbox("Loan Term", ["36 months", "60 months"])
        total_acc = st.number_input("Total Credit Accounts", 0, 100, 30)
        home_ownership = st.selectbox("Home Ownership", list(home_ownership_map.keys()))
        open_acc = st.number_input("Open Credit Lines", 0, 100, 10)
        revol_util = st.number_input("Revolving Utilization (%)", 0.0, 150.0, 30.0)
        emp_length = st.number_input("Employment Length (Years)", 0, 50, 5)

    with col2:
        int_rate = st.number_input("Interest Rate (%)", 5.0, 30.0, 12.5)
        annual_inc = st.number_input("Annual Income", 10000, 500000, 60000)
        installment = st.number_input("Installment Amount", 0.0, 2000.0, 400.0)
        revol_bal = st.number_input("Revolving Balance", 0.0, 500000.0, 15000.0)
        dti = st.number_input("Debt-to-Income Ratio", 0.0, 50.0, 15.0)
        income_to_loan = st.number_input("Income-to-Loan Ratio", 0.0, 20.0, 3.0)
        dti_util = st.number_input("DTI Utilization", 0.0, 10.0, 1.5)

    submitted = st.form_submit_button("Get Prediction")

# Prediction and SHAP
if submitted and model is not None:
    # Feature names in correct order
    feature_names = [
        "grade", "mort_acc", "term", "total_acc", "home_ownership",
        "open_acc", "revol_util", "emp_length", "int_rate", "annual_inc",
        "installment", "loan_amnt", "revol_bal", "dti", "income_to_loan", "dti_util"
    ]
    
    # Create input DataFrame with correct feature order
    input_df = pd.DataFrame([[
        grade_map[grade],
        mort_acc,
        36.0 if term == "36 months" else 60.0,
        total_acc,
        home_ownership_map[home_ownership],
        open_acc,
        revol_util,
        emp_length,
        int_rate,
        annual_inc,
        installment,
        loan_amnt,
        revol_bal,
        dti,
        income_to_loan,
        dti_util
    ]], columns=feature_names).astype("float64")

    # Get prediction
    prediction = model.predict(input_df)[0]
    try:
        proba = model.predict_proba(input_df)[0][1]
    except:
        proba = None
    
    # Display results
    st.subheader("Prediction Result")
    if prediction == 0:  # Assuming 1 is default
        st.error(f"CHARGED OFF (Default Probability: {proba:.1%})" if proba else "CHARGED OFF")
    else:
        st.success(f"FULLY PAID (Default Probability: {proba:.1%})" if proba else "FULLY PAID")

    # SHAP explanation with feature names
    st.subheader("Explanation")
    try:
        # Get the LightGBM model from the pipeline
        lgbm_model = model.named_steps['lgbm']
        
        # Apply preprocessing
        scaled_input = model.named_steps['scaler'].transform(input_df)
        
        # Generate SHAP values with feature names
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(scaled_input)
        
        # Create waterfall plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, scaled_input, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
        
    except Exception as e:
        st.error(f"Could not generate explanation: {str(e)}")
