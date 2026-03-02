import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(
    page_title="Credit Scoring | ระบบประเมินสินเชื่อ",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern Clean Design ---
st.markdown("""
<style>
    /* --- Google Font Import --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* --- Global --- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* --- Main Content Area --- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 960px;
    }

    /* --- Header Styling --- */
    h1 {
        font-weight: 700 !important;
        font-size: 1.75rem !important;
        color: #E8EAED !important;
        letter-spacing: -0.02em !important;
        border-bottom: 2px solid #3D4043 !important;
        padding-bottom: 0.75rem !important;
        margin-bottom: 1.5rem !important;
    }

    h2, h3 {
        font-weight: 600 !important;
        color: #BDC1C6 !important;
        letter-spacing: -0.01em !important;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #1E1E2E !important;
        border-right: 1px solid #2D2D3D !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #CDD6F4 !important;
        border-bottom: 1px solid #313244 !important;
        padding-bottom: 0.5rem !important;
    }

    [data-testid="stSidebar"] label {
        color: #A6ADC8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* --- Metric Cards --- */
    [data-testid="stMetric"] {
        background-color: #1E1E2E;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.25rem;
    }

    [data-testid="stMetric"] label {
        color: #A6ADC8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #CDD6F4 !important;
    }

    /* --- Buttons --- */
    .stButton > button {
        background-color: #89B4FA !important;
        color: #1E1E2E !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em !important;
    }

    .stButton > button:hover {
        background-color: #74C7EC !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(137, 180, 250, 0.25) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* --- Number Input & Select --- */
    [data-testid="stNumberInput"] input,
    .stSelectbox > div > div {
        border-radius: 8px !important;
    }

    /* --- Expander --- */
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        color: #A6ADC8 !important;
    }

    /* --- Divider --- */
    hr {
        border-color: #313244 !important;
        margin: 1.5rem 0 !important;
    }

    /* --- Result Cards --- */
    .result-card {
        background-color: #1E1E2E;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }

    .risk-low {
        border-left: 4px solid #A6E3A1;
    }

    .risk-medium {
        border-left: 4px solid #F9E2AF;
    }

    .risk-high {
        border-left: 4px solid #F38BA8;
    }

    .risk-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }

    .risk-label-low { color: #A6E3A1; }
    .risk-label-medium { color: #F9E2AF; }
    .risk-label-high { color: #F38BA8; }

    .risk-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #CDD6F4;
        margin-bottom: 0.25rem;
    }

    .risk-desc {
        font-size: 0.85rem;
        color: #A6ADC8;
    }

    /* --- Subtitle text --- */
    .subtitle {
        color: #A6ADC8;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }

    /* --- Score Display --- */
    .score-display {
        text-align: center;
        padding: 1.5rem;
        background-color: #1E1E2E;
        border: 1px solid #313244;
        border-radius: 12px;
    }

    .score-number {
        font-size: 3rem;
        font-weight: 700;
        color: #CDD6F4;
        line-height: 1;
    }

    .score-label {
        font-size: 0.8rem;
        color: #A6ADC8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.5rem;
    }

    /* --- Hide Streamlit Branding --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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
        imputer = joblib.load('models/imputer.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return model, imputer, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        return None, None, None


def get_risk_html(probability):
    """Return styled HTML card based on risk level."""
    if probability < 0.2:
        return f"""
        <div class="result-card risk-low">
            <div class="risk-label risk-label-low">Low Risk</div>
            <div class="risk-title">ความเสี่ยงต่ำ</div>
            <div class="risk-desc">ผ่านเกณฑ์การอนุมัติสินเชื่อเบื้องต้น</div>
        </div>
        """
    elif probability < 0.5:
        return f"""
        <div class="result-card risk-medium">
            <div class="risk-label risk-label-medium">Medium Risk</div>
            <div class="risk-title">ความเสี่ยงปานกลาง</div>
            <div class="risk-desc">ต้องพิจารณาเพิ่มเติมก่อนอนุมัติ</div>
        </div>
        """
    else:
        return f"""
        <div class="result-card risk-high">
            <div class="risk-label risk-label-high">High Risk</div>
            <div class="risk-title">ความเสี่ยงสูง</div>
            <div class="risk-desc">ไม่ผ่านเกณฑ์การอนุมัติเบื้องต้น</div>
        </div>
        """


# --- Main Logic ---
def main():
    st.title("Credit Scoring System")
    st.markdown('<p class="subtitle">ระบบประเมินความน่าจะเป็นที่ลูกค้าจะผิดนัดชำระหนี้ &mdash; กรอกข้อมูลผู้ขอสินเชื่อที่แถบด้านซ้ายเพื่อดูผลลัพธ์</p>', unsafe_allow_html=True)

    model, imputer, scaler = load_model()
    medians = load_medians()
    
    if model is None:
        st.error("ไม่พบไฟล์โมเดลหรือไฟล์ประกอบสำคัญ กรุณาตรวจสอบโฟลเดอร์ models/")
        st.stop()

    # --- Sidebar Inputs ---
    st.sidebar.header("ข้อมูลผู้ขอสินเชื่อ")

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
    own_car = st.sidebar.checkbox("มีรถยนต์ส่วนตัว")
    own_realty = st.sidebar.checkbox("มีอสังหาริมทรัพย์", value=True)
    work_phone = st.sidebar.checkbox("มีเบอร์โทรศัพท์ที่ทำงาน", value=True)
    
    # 4. ข้อมูลพื้นที่และเครดิต
    st.sidebar.subheader("4. ข้อมูลอื่นๆ")
    region_rating = st.sidebar.selectbox("ระดับความเจริญของพื้นที่", [1, 2, 3], index=1, help="1 = เจริญมาก, 3 = เจริญน้อย")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ข้อมูลเครดิตภายนอก")
    st.sidebar.caption("ปกติข้อมูลส่วนนี้จะดึงจากระบบเครดิตบูโรโดยตรง ในที่นี้ให้ทดลองปรับค่า 0.0 (แย่) ถึง 1.0 (ดีเยี่ยม)")
    ext_source_1 = st.sidebar.slider("คะแนนเครดิตแหล่งที่ 1", 0.0, 1.0, 0.5)
    ext_source_2 = st.sidebar.slider("คะแนนเครดิตแหล่งที่ 2", 0.0, 1.0, 0.5)
    ext_source_3 = st.sidebar.slider("คะแนนเครดิตแหล่งที่ 3", 0.0, 1.0, 0.5)

    # --- Processing Inputs ---
    if st.button("วิเคราะห์ความเสี่ยง", type="primary", use_container_width=True):
        with st.spinner("กำลังประมวลผล..."):
            
            # Initialize input dict with MEDIANS for robustness
            input_dict = {feat: medians.get(feat, 0) for feat in EXPECTED_FEATURES}
            
            # --- Map User Inputs to Features ---
            
            # Numeric Mappings
            input_dict['AMT_CREDIT'] = credit_amount
            input_dict['AMT_GOODS_PRICE'] = goods_price
            input_dict['DAYS_BIRTH'] = age * -365
            input_dict['DAYS_EMPLOYED'] = employment_years * -365
            input_dict['EXT_SOURCE_1'] = ext_source_1
            input_dict['EXT_SOURCE_2'] = ext_source_2
            input_dict['EXT_SOURCE_3'] = ext_source_3
            
            input_dict['REGION_RATING_CLIENT'] = region_rating
            input_dict['REGION_RATING_CLIENT_W_CITY'] = region_rating
            input_dict['FLAG_WORK_PHONE'] = 1 if work_phone else 0
            
            # Derived Domain Features
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

            # Categorical Mappings (One-Hot)
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
            
            if 'มนุษย์เงินเดือน' in income_type: input_dict['NAME_INCOME_TYPE_Working'] = 1
            elif 'ข้าราชการ' in income_type: input_dict['NAME_INCOME_TYPE_State servant'] = 1
            elif 'บำนาญ' in income_type: input_dict['NAME_INCOME_TYPE_Pensioner'] = 1
            elif 'เอกชน' in income_type: input_dict['NAME_INCOME_TYPE_Commercial associate'] = 1

            # Convert to DataFrame
            try:
                model_features = EXPECTED_FEATURES
                aligned_input = {feat: input_dict.get(feat, 0) for feat in model_features}
                df_predict = pd.DataFrame([aligned_input])
                
                # Preprocessing
                if imputer:
                    df_predict_imputed = pd.DataFrame(imputer.transform(df_predict), columns=df_predict.columns)
                else:
                    df_predict_imputed = df_predict
                    
                if scaler:
                    try:
                        df_predict_scaled = pd.DataFrame(scaler.transform(df_predict_imputed), columns=df_predict.columns)
                    except ValueError as ve:
                        print(f"Scaling warning: {ve}") 
                        df_predict_scaled = df_predict_imputed
                else:
                    df_predict_scaled = df_predict_imputed
                
                # Prediction
                probability = model.predict_proba(df_predict_scaled)[:, 1][0]
                
                # Credit Score (300-850 scale, inverse to risk)
                credit_score = int(850 - (probability * 550))
                
                # --- Display Results ---
                st.markdown("---")
                st.subheader("ผลการวิเคราะห์")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="Default Probability", value=f"{probability:.2%}")
                
                with col2:
                    st.markdown(f"""
                    <div class="score-display">
                        <div class="score-number">{credit_score}</div>
                        <div class="score-label">Credit Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(get_risk_html(probability), unsafe_allow_html=True)

                # Key Factors
                st.write("")
                with st.expander("ดูปัจจัยที่มีผลต่อการคำนวณ"):
                    st.caption("ค่าสำคัญที่ใช้ในการคำนวณ")
                    
                    factor_col1, factor_col2, factor_col3 = st.columns(3)
                    with factor_col1:
                        st.metric("Ext Source Mean", f"{input_dict['EXT_SOURCE_MEAN']:.3f}")
                    with factor_col2:
                        st.metric("Employment Years", f"{employment_years}")
                    with factor_col3:
                        st.metric("Credit/Annuity Ratio", f"{input_dict['CREDIT_TO_ANNUITY_RATIO']:.1f}")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
                st.caption("โปรดตรวจสอบว่าข้อมูลที่กรอกถูกต้องและครบถ้วน")


if __name__ == "__main__":
    main()
