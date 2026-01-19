
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
MODEL_PATH = "models/best_model_lgbm.pkl"
# List of features expected by the model (derived from training data)
EXPECTED_FEATURES = [
    'AMT_CREDIT', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START', 
    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
    # ... (Aggregated features will be defaulted to 0 or median if not collected)
    'CREDIT_TO_ANNUITY_RATIO', 'CREDIT_TO_GOODS_RATIO', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 
    'REGISTRATION_YEARS', 'ID_PUBLISH_YEARS', 'EMPLOYMENT_TO_AGE_RATIO', 
    'EXT_SOURCE_MEAN', 'EXT_SOURCE_STD', 'EXT_SOURCE_MIN', 'EXT_SOURCE_MAX',
    # Categoricals (One-Hot Encoded placeholders)
    'NAME_CONTRACT_TYPE_Revolving loans', 'CODE_GENDER_M', 'FLAG_OWN_CAR_Y', 
    'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Working',
    'NAME_EDUCATION_TYPE_Higher education', 'NAME_EDUCATION_TYPE_Secondary / secondary special',
    'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Single / not married',
    'NAME_HOUSING_TYPE_House / apartment', 'NAME_HOUSING_TYPE_With parents',
    'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers', 'OCCUPATION_TYPE_Low-skill Laborers',
    'ORGANIZATION_TYPE_Business Entity Type 3', 'ORGANIZATION_TYPE_Self-employed', 'ORGANIZATION_TYPE_XNA'
]

# --- Helper Functions ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return None

# --- Main Logic ---
def main():
    st.title("💳 Credit Scoring Prediction System")
    st.markdown("""
    This application predicts the probability of a client defaulting on a loan.
    Please input the applicant's details in the sidebar.
    """)

    model = load_model()
    
    if model is None:
        st.error(f"Could not load model from `{MODEL_PATH}`. Please check if the file exists.")
        st.stop()

    # --- Sidebar Inputs ---
    st.sidebar.header(" Applicant Details")

    # 1. Personal Info
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.slider("Age (Years)", 20, 70, 30)
    education = st.sidebar.selectbox("Education Level", 
                                     ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
    family_status = st.sidebar.selectbox("Family Status", 
                                         ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"])
    housing_type = st.sidebar.selectbox("Housing Type", 
                                        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])

    # 2. Financial Info
    income = st.sidebar.number_input("Annual Income ($)", min_value=10000.0, value=50000.0, step=5000.0)
    credit_amount = st.sidebar.number_input("Credit Amount ($)", min_value=10000.0, value=200000.0, step=5000.0)
    annuity = st.sidebar.number_input("Loan Annuity ($)", min_value=1000.0, value=10000.0, step=500.0)
    goods_price = st.sidebar.number_input("Goods Price ($)", min_value=10000.0, value=180000.0, step=5000.0)
    
    # 3. Employment & Assets
    employment_years = st.sidebar.slider("Years Employed", 0, 50, 5)
    occupation = st.sidebar.selectbox("Occupation", 
                                      ["Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff", "IT staff", "Other"])
    org_type = st.sidebar.selectbox("Organization Type", ["Business Entity Type 3", "Self-employed", "Other", "XNA"])
    own_car = st.sidebar.checkbox("Owns a Car?")
    own_realty = st.sidebar.checkbox("Owns Realty?", value=True)

    # 4. External Sources (Mocking these as they come from credit bureaus)
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ External Credit Data")
    ext_source_1 = st.sidebar.slider("External Source 1 (Normalized)", 0.0, 1.0, 0.5)
    ext_source_2 = st.sidebar.slider("External Source 2 (Normalized)", 0.0, 1.0, 0.5)
    ext_source_3 = st.sidebar.slider("External Source 3 (Normalized)", 0.0, 1.0, 0.5)

    # --- Processing Inputs ---
    if st.button(" Analyze Risk", type="primary"):
        with st.spinner("Calculating Risk Score..."):
            
            # Initialize input dict with ALL model features set to 0 (default)
            # This handles the hundreds of aggregated features by assuming 'average' or 'zero' impact for this demo
            # Ideally, these would be imputed with training set medians.
            input_dict = {feature: 0 for feature in EXPECTED_FEATURES}
            
            # --- Map User Inputs to Features ---
            
            # Numeric Mappings
            input_dict['AMT_CREDIT'] = credit_amount
            input_dict['AMT_GOODS_PRICE'] = goods_price
            input_dict['DAYS_BIRTH'] = age * -365 # Approximate
            input_dict['DAYS_EMPLOYED'] = employment_years * -365 # Approximate
            input_dict['EXT_SOURCE_1'] = ext_source_1
            input_dict['EXT_SOURCE_2'] = ext_source_2
            input_dict['EXT_SOURCE_3'] = ext_source_3
            
            # Derived Domain Features (Replicating feature engineering)
            input_dict['CREDIT_TO_ANNUITY_RATIO'] = credit_amount / annuity if annuity > 0 else 0
            input_dict['CREDIT_TO_GOODS_RATIO'] = credit_amount / goods_price if goods_price > 0 else 0
            input_dict['AGE_YEARS'] = age
            input_dict['EMPLOYMENT_YEARS'] = employment_years
            input_dict['EMPLOYMENT_TO_AGE_RATIO'] = employment_years / age if age > 0 else 0
            
            ext_list = [ext_source_1, ext_source_2, ext_source_3]
            input_dict['EXT_SOURCE_MEAN'] = np.mean(ext_list)
            input_dict['EXT_SOURCE_STD'] = np.std(ext_list)
            input_dict['EXT_SOURCE_MIN'] = np.min(ext_list)
            input_dict['EXT_SOURCE_MAX'] = np.max(ext_list)

            # Categorical Mappings (One-Hot Encoding Manual Set)
            if gender == 'Male': input_dict['CODE_GENDER_M'] = 1
            if own_car: input_dict['FLAG_OWN_CAR_Y'] = 1
            
            if 'Secondary' in education: input_dict['NAME_EDUCATION_TYPE_Secondary / secondary special'] = 1
            elif 'Higher' in education: input_dict['NAME_EDUCATION_TYPE_Higher education'] = 1
            
            if family_status == 'Married': input_dict['NAME_FAMILY_STATUS_Married'] = 1
            elif 'Single' in family_status: input_dict['NAME_FAMILY_STATUS_Single / not married'] = 1
            
            if housing_type == 'House / apartment': input_dict['NAME_HOUSING_TYPE_House / apartment'] = 1
            elif 'parents' in housing_type: input_dict['NAME_HOUSING_TYPE_With parents'] = 1
            
            if occupation == 'Core staff': input_dict['OCCUPATION_TYPE_Core staff'] = 1
            elif occupation == 'Drivers': input_dict['OCCUPATION_TYPE_Drivers'] = 1
            elif 'Laborers' in occupation: input_dict['OCCUPATION_TYPE_Low-skill Laborers'] = 1
            
            if org_type == 'Business Entity Type 3': input_dict['ORGANIZATION_TYPE_Business Entity Type 3'] = 1
            elif org_type == 'Self-employed': input_dict['ORGANIZATION_TYPE_Self-employed'] = 1
            elif org_type == 'XNA': input_dict['ORGANIZATION_TYPE_XNA'] = 1

            # Convert to DataFrame
            # IMPORTANT: We must match the exact feature order expected by the model if it's sensitive to it (GBMs usually identify by name, but pipelines might not)
            # Since we have the list, let's assume we pass all keys. However, the model obj might have .feature_name_
            
            try:
                # Attempt to retrieve feature names from model if available (LightGBM)
                if hasattr(model, 'feature_name_'):
                    model_features = model.feature_name_
                elif hasattr(model, 'feature_names_in_'): # Sklearn
                    model_features = model.feature_names_in_
                else:
                    model_features = EXPECTED_FEATURES # Fallback
                
                # Align input_dict keys with model features
                # Fill missing features with 0 (safe default for one-hot and null-imputed features)
                aligned_input = {feat: input_dict.get(feat, 0) for feat in model_features}
                
                df_predict = pd.DataFrame([aligned_input])
                
                # --- Prediction ---
                # Check for Scaler: If trained on scaled data, we ideally need to scale.
                # Assuming tree-based model (LightGBM) is robust enough or was trained on data that preserved relative magnitude.
                # If scaling is strictly required, we would load the scaler here.
                
                probability = model.predict_proba(df_predict)[:, 1][0]
                
                # --- Display ---
                st.success("Analysis Complete!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                # Credit Score Simulation (e.g., 300-850 scale inverse to risk)
                credit_score = int(850 - (probability * 550))
                
                with col1:
                    st.metric(label="Predicted Probability of Default", value=f"{probability:.2%}")
                
                with col2:
                    st.metric(label="Estimated Credit Score", value=f"{credit_score}")
                    
                with col3:
                    if probability < 0.2:
                        st.balloons()
                        st.success("**Low Risk** (Approved)")
                    elif probability < 0.5:
                        st.warning("**Medium Risk** (Review Required)")
                    else:
                        st.error("**High Risk** (Reject)")

                # Factors
                st.subheader("Factor Analysis")
                st.write("Top influencing factors for this applicant (Simulated based on inputs):")
                
                # Simple bar chart of key inputs
                key_metrics = {
                    "Ext Source Mean": input_dict['EXT_SOURCE_MEAN'],
                    "Employment Years": employment_years,
                    "Credit/Ann Ratio": input_dict['CREDIT_TO_ANNUITY_RATIO']
                }
                st.bar_chart(pd.DataFrame.from_dict(key_metrics, orient='index', columns=['Value']))

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.write("Debug info: Feature mismatch or model format error.")

if __name__ == "__main__":
    main()
