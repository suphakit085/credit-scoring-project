import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --- Constants ---
MODEL_PATH = 'models/best_model_lgbm.pkl'
FEATURE_NAMES_PATH = 'data/features/feature_names.csv'

# Helper to load features
@st.cache_data
def load_feature_names():
    if os.path.exists(FEATURE_NAMES_PATH):
        df = pd.read_csv(FEATURE_NAMES_PATH)
        return df['feature'].tolist()
    return []

EXPECTED_FEATURES = load_feature_names()

# Helper to load medians
@st.cache_data
def load_medians():
    path = 'data/processed/feature_medians.json'
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                import json
                return json.load(f)
        except:
            return {}
    return {}

# --- Helper Functions ---
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing artifacts from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        # Load scaler and imputer if they exist (created by recreate_scaling.py)
        imputer = joblib.load('models/imputer.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return model, imputer, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None
    except Exception as e:
        # Fallback for older sessions where artifacts might not be ready
        st.error(f"Unexpected error loading model: {e}")
        return None, None, None

# --- Main Logic ---
def main():
    st.title("💳 ระบบประเมินความเสี่ยงสินเชื่อ (Credit Scoring)")
    st.markdown("""
    แอปพลิเคชันนี้ใช้สำหรับ **ประเมินความน่าจะเป็นที่ลูกค้าจะผิดนัดชำระหนี้** 
    โปรดกรอกข้อมูลผู้ขอสินเชื่อที่ด้านยซ้าย (หรือเมนู Sidebar) เพื่อดูผลลัพธ์
    """)

    model, imputer, scaler = load_model()
    medians = load_medians()
    
    if model is None:
        st.error(f"ไม่พบไฟล์โมเดลหรือไฟล์ประกอบสำคัญ กรุณาตรวจสอบโฟลเดอร์ `models/`")
        st.stop()

    # --- Sidebar Inputs ---
    st.sidebar.header("📝 ข้อมูลผู้ขอสินเชื่อ")

    # 1. ข้อมูลส่วนตัว
    st.sidebar.subheader("1. ข้อมูลส่วนตัว")
    gender = st.sidebar.selectbox("เพศ", ["หญิง (Female)", "ชาย (Male)"])
    age = st.sidebar.slider("อายุ (ปี)", 20, 70, 30)
    education = st.sidebar.selectbox("ระดับการศึกษา", 
                                     ["มัธยมศึกษา (Secondary)", "ปริญญาตรี (Higher education)", "ไม่จบปริญญาตรี (Incomplete higher)", "มัธยมต้น (Lower secondary)", "ปริญญาโทขึ้นไป (Academic degree)"])
    family_status = st.sidebar.selectbox("สถานะครอบครัว", 
                                         ["แต่งงานแล้ว (Married)", "โสด (Single / not married)", "จดทะเบียนสมรส (Civil marriage)", "หม้าย (Widow)", "หย่าร้าง/แยกกันอยู่ (Separated)"])
    housing_type = st.sidebar.selectbox("ประเภทที่อยู่อาศัย", 
                                        ["บ้าน/อพาร์ทเมนท์ส่วนตัว", "อยู่กับพ่อแม่", "ที่พักของเทศบาล", "เช่าอพาร์ทเมนท์", "ที่พักสวัสดิการ", "คอนโด/สหกรณ์"])

    # 2. ข้อมูลการเงิน
    st.sidebar.subheader("2. ข้อมูลการเงิน")
    income = st.sidebar.number_input("รายได้ต่อปี (บาท)", min_value=10000.0, value=50000.0, step=5000.0)
    credit_amount = st.sidebar.number_input("วงเงินกู้ที่ขอ (บาท)", min_value=10000.0, value=200000.0, step=5000.0)
    annuity = st.sidebar.number_input("ยอดผ่อนชำระต่องวด (บาท)", min_value=1000.0, value=10000.0, step=500.0)
    goods_price = st.sidebar.number_input("ราคาสินค้า (กรณีสินเชื่อสินค้า) (บาท)", min_value=10000.0, value=180000.0, step=5000.0)
    
    # 3. การทำงานและทรัพย์สิน
    st.sidebar.subheader("3. การทำงานและทรัพย์สิน")
    income_type = st.sidebar.selectbox("ประเภทรายได้",
                                       ["มนุษย์เงินเดือน (Working)", "ข้าราชการ/รัฐวิสาหกิจ (State servant)", "ผู้รับบำนาญ (Pensioner)", "พนักงานบริษัทเอกชน (Commercial associate)", "คนว่างงาน/นักศึกษา (Unemployed/Student)"])
    employment_years = st.sidebar.slider("ประสบการณ์ทำงาน (ปี)", 0, 50, 5)
    occupation = st.sidebar.selectbox("อาชีพ", 
                                      ["แรงงานทั่วไป (Laborers)", "พนักงานหลัก/เจ้าหน้าที่ (Core staff)", "บัญชี (Accountants)", "ผู้จัดการ (Managers)", "คนขับรถ (Drivers)", "พนักงานขาย (Sales staff)", "ไอที (IT staff)", "อื่นๆ"])
    org_type = st.sidebar.selectbox("ประเภทองค์กร", ["ธุรกิจส่วนตัว/นิติบุคคล (Business Entity Type 3)", "อาชีพอิสระ (Self-employed)", "อื่นๆ", "ไม่ระบุ (XNA)"])
    own_car = st.sidebar.checkbox("มีรถยนต์ส่วนตัว?")
    own_realty = st.sidebar.checkbox("มีอสังหาริมทรัพย์?", value=True)
    work_phone = st.sidebar.checkbox("มีเบอร์โทรศัพท์ที่ทำงาน?", value=True)
    
    # 4. ข้อมูลพื้นที่และเครดิต
    st.sidebar.subheader("4. ข้อมูลอื่นๆ")
    region_rating = st.sidebar.selectbox("ระดับความเจริญของพื้นที่ (Region Rating)", [1, 2, 3], index=1, help="1=เจริญมาก, 3=เจริญน้อย")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ ข้อมูลเครดิตภายนอก (จำลอง)")
    st.sidebar.caption("ปกติข้อมูลส่วนนี้จะดึงมาจากระบบเครดิตบูโรโดยตรง ในที่นี้ให้ทดลองปรับค่า 0.0 (แย่) ถึง 1.0 (ดีเยี่ยม)")
    ext_source_1 = st.sidebar.slider("คะแนนเครดิตแหล่งที่ 1", 0.0, 1.0, 0.5)
    ext_source_2 = st.sidebar.slider("คะแนนเครดิตแหล่งที่ 2", 0.0, 1.0, 0.5)
    ext_source_3 = st.sidebar.slider("คะแนนเครดิตแหล่งที่ 3", 0.0, 1.0, 0.5)

    # --- Processing Inputs ---
    if st.button("🚀 วิเคราะห์ความเสี่ยง (Analyze Risk)", type="primary"):
        with st.spinner("กำลังประมวลผล... กรุณารอสักครู่"):
            
            # Initialize input dict with MEDIANS for robustness
            # This ensures missing features (like previous application history) use average values instead of 0 (outlier)
            input_dict = {feat: medians.get(feat, 0) for feat in EXPECTED_FEATURES}
            
            # --- Map User Inputs to Features ---
            
            # Numeric Mappings
            input_dict['AMT_CREDIT'] = credit_amount
            input_dict['AMT_GOODS_PRICE'] = goods_price
            input_dict['DAYS_BIRTH'] = age * -365 # Convert to days (negative)
            input_dict['DAYS_EMPLOYED'] = employment_years * -365 # Convert to days (negative)
            input_dict['EXT_SOURCE_1'] = ext_source_1
            input_dict['EXT_SOURCE_2'] = ext_source_2
            input_dict['EXT_SOURCE_3'] = ext_source_3
            
            input_dict['REGION_RATING_CLIENT'] = region_rating
            input_dict['REGION_RATING_CLIENT_W_CITY'] = region_rating
            input_dict['FLAG_WORK_PHONE'] = 1 if work_phone else 0
            
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
            if 'ชาย' in gender: input_dict['CODE_GENDER_M'] = 1
            if own_car: input_dict['FLAG_OWN_CAR_Y'] = 1
            
            if 'มัธยมศึกษา' in education: input_dict['NAME_EDUCATION_TYPE_Secondary / secondary special'] = 1
            elif 'ปริญญาตรี' in education: input_dict['NAME_EDUCATION_TYPE_Higher education'] = 1
            
            if 'แต่งงานแล้ว' in family_status: input_dict['NAME_FAMILY_STATUS_Married'] = 1
            elif 'โสด' in family_status: input_dict['NAME_FAMILY_STATUS_Single / not married'] = 1
            
            if 'บ้าน/อพาร์ทเมนท์ส่วนตัว' in housing_type: input_dict['NAME_HOUSING_TYPE_House / apartment'] = 1
            elif 'อยู่กับพ่อแม่' in housing_type: input_dict['NAME_HOUSING_TYPE_With parents'] = 1
            
            if 'เจ้าหน้าที่' in occupation: input_dict['OCCUPATION_TYPE_Core staff'] = 1
            elif 'คนขับรถ' in occupation: input_dict['OCCUPATION_TYPE_Drivers'] = 1
            elif 'แรงงาน' in occupation: input_dict['OCCUPATION_TYPE_Low-skill Laborers'] = 1
            
            if 'ธุรกิจส่วนตัว' in org_type: input_dict['ORGANIZATION_TYPE_Business Entity Type 3'] = 1
            elif 'อาชีพอิสระ' in org_type: input_dict['ORGANIZATION_TYPE_Self-employed'] = 1
            elif 'ไม่ระบุ' in org_type: input_dict['ORGANIZATION_TYPE_XNA'] = 1
            
            # Income Type Mapping
            if 'มนุษย์เงินเดือน' in income_type: input_dict['NAME_INCOME_TYPE_Working'] = 1
            elif 'ข้าราชการ' in income_type: input_dict['NAME_INCOME_TYPE_State servant'] = 1
            elif 'บำนาญ' in income_type: input_dict['NAME_INCOME_TYPE_Pensioner'] = 1
            elif 'เอกชน' in income_type: input_dict['NAME_INCOME_TYPE_Commercial associate'] = 1

            # Convert to DataFrame
            try:
                # Force usage of EXPECTED_FEATURES (from CSV) to ensure exact match with training data.
                # Do NOT use model.feature_name_ as it may return sanitized names (underscores) 
                # which causes mismatch errors with Sklearn's validation.
                model_features = EXPECTED_FEATURES
                
                # Align input_dict keys with model features
                aligned_input = {feat: input_dict.get(feat, 0) for feat in model_features}
                
                df_predict = pd.DataFrame([aligned_input])
                
                # --- Preprocessing ---
                # 1. Impute
                if imputer:
                    df_predict_imputed = pd.DataFrame(imputer.transform(df_predict), columns=df_predict.columns)
                else:
                    df_predict_imputed = df_predict
                    
                # 2. Scale
                if scaler:
                    try:
                        # Ensure columns match scaler expectations (order matters)
                        # The scaler was fitted on 'expected_features', which matches 'model_features' (ideally)
                        df_predict_scaled = pd.DataFrame(scaler.transform(df_predict_imputed), columns=df_predict.columns)
                    except ValueError as ve:
                        # Only warn in console, attempt to predict anyway if robust
                        print(f"Scaling warning: {ve}") 
                        df_predict_scaled = df_predict_imputed
                else:
                    df_predict_scaled = df_predict_imputed
                
                # --- Prediction ---
                probability = model.predict_proba(df_predict_scaled)[:, 1][0]
                
                # --- Display ---
                # Credit Score Simulation (e.g., 300-850 scale inverse to risk)
                credit_score = int(850 - (probability * 550))
                
                st.write("---")
                st.subheader("📊 ผลการวิเคราะห์")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="โอกาสผิดนัดชำระหนี้ (Default Prob.)", value=f"{probability:.2%}")
                
                with col2:
                    st.metric(label="เครดิตสกอร์ (Credit Score)", value=f"{credit_score}")
                    
                with col3:
                    if probability < 0.2:
                        st.balloons()
                        st.success("**ความเสี่ยงต่ำ (Low Risk)**\n\n✅ อนุมัติสินเชื่อเบื้องต้น")
                    elif probability < 0.5:
                        st.warning("**ความเสี่ยงปานกลาง (Medium Risk)**\n\n⚠️ ต้องพิจารณาเพิ่มเติม")
                    else:
                        st.error("**ความเสี่ยงสูง (High Risk)**\n\n❌ ไม่ผ่านเกณฑ์เบื้องต้น")

                # Factors
                st.write("")
                with st.expander("ดูปัจจัยที่มีผลต่อการคำนวณ (Analysis Details)"):
                    st.write("ค่าสำคัญที่ใช้ในการคำนวณ:")
                    key_metrics = {
                        "คะแนนเครดิตเฉลี่ย (Ext Source Mean)": input_dict['EXT_SOURCE_MEAN'],
                        "อายุงาน (ปี)": employment_years,
                        "สัดส่วนวงเงินกู้ต่อรายได้": input_dict['CREDIT_TO_ANNUITY_RATIO']
                    }
                    st.bar_chart(pd.DataFrame.from_dict(key_metrics, orient='index', columns=['Value']))

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
                st.write("คำแนะนำ: โปรดตรวจสอบว่าข้อมูลที่กรอกถูกต้องและครบถ้วน")


if __name__ == "__main__":
    main()
