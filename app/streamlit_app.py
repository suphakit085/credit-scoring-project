"""Premium Streamlit interface for the calibrated credit-risk model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    DEFAULT_BUNDLE_PATH,
    load_bundle,
    predict_default_probability,
    probability_to_score,
    risk_level,
)


st.set_page_config(
    page_title="Aurelis | Credit Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


LABELS = {
    "gender": {"F": "หญิง", "M": "ชาย", "XNA": "ไม่ระบุ"},
    "education": {
        "Secondary / secondary special": "มัธยมศึกษา",
        "Higher education": "ปริญญาตรี",
        "Incomplete higher": "ไม่จบปริญญาตรี",
        "Lower secondary": "มัธยมต้น",
        "Academic degree": "ปริญญาโทขึ้นไป",
    },
    "family": {
        "Married": "สมรส",
        "Single / not married": "โสด",
        "Civil marriage": "สมรสตามกฎหมาย",
        "Separated": "แยกกันอยู่",
        "Widow": "หม้าย",
    },
    "housing": {
        "House / apartment": "บ้านหรืออพาร์ตเมนต์",
        "With parents": "อาศัยกับครอบครัว",
        "Municipal apartment": "ที่พักเทศบาล",
        "Rented apartment": "เช่าอพาร์ตเมนต์",
        "Office apartment": "ที่พักสวัสดิการ",
        "Co-op apartment": "สหกรณ์เคหสถาน",
    },
    "income": {
        "Working": "พนักงานประจำ",
        "Commercial associate": "พนักงานเอกชน",
        "State servant": "ข้าราชการ / รัฐวิสาหกิจ",
        "Pensioner": "ผู้รับบำนาญ",
        "Unemployed": "ไม่ได้ทำงาน",
    },
    "occupation": {
        "Laborers": "แรงงานทั่วไป",
        "Core staff": "เจ้าหน้าที่หลัก",
        "Sales staff": "พนักงานขาย",
        "Managers": "ผู้จัดการ",
        "Drivers": "คนขับรถ",
        "Accountants": "นักบัญชี",
        "IT staff": "เจ้าหน้าที่ไอที",
    },
    "organization": {
        "Business Entity Type 3": "ภาคธุรกิจ",
        "Self-employed": "อาชีพอิสระ",
        "Government": "หน่วยงานรัฐ",
        "School": "สถานศึกษา",
        "XNA": "ไม่ระบุ",
    },
}

RISK_PRESENTATION = {
    "low": {
        "title": "ความเสี่ยงต่ำ",
        "eyebrow": "LOW RISK PROFILE",
        "description": "ต่ำกว่าเกณฑ์ติดตามของโมเดล",
        "class_name": "low",
        "icon": "✓",
    },
    "medium": {
        "title": "ความเสี่ยงปานกลาง",
        "eyebrow": "REVIEW RECOMMENDED",
        "description": "ควรตรวจสอบข้อมูลและเอกสารเพิ่มเติม",
        "class_name": "medium",
        "icon": "!",
    },
    "high": {
        "title": "ความเสี่ยงสูง",
        "eyebrow": "DEEPER REVIEW",
        "description": "ถึงเกณฑ์ที่โมเดลระบุให้ตรวจสอบเชิงลึก",
        "class_name": "high",
        "icon": "!",
    },
}


st.markdown(
    """
    <style>
      :root {
        --bg: #09131f;
        --surface: #101f2f;
        --surface-2: #13283b;
        --line: rgba(176, 211, 224, .16);
        --ink: #f1f7f8;
        --muted: #91aab8;
        --mint: #76e6c2;
        --cyan: #61c7f5;
        --amber: #f8c66b;
        --rose: #ff8c9f;
      }

      .stApp {
        background:
          radial-gradient(circle at 85% -5%, rgba(49, 148, 179, .22), transparent 32rem),
          radial-gradient(circle at 20% 0%, rgba(55, 190, 157, .14), transparent 29rem),
          var(--bg);
        color: var(--ink);
      }

      header[data-testid="stHeader"] {
        background: rgba(9, 19, 31, .92) !important;
        border-bottom: 1px solid rgba(176, 211, 224, .08);
        backdrop-filter: blur(14px);
      }

      [data-testid="stToolbar"] { background: transparent !important; }
      header[data-testid="stHeader"] button { color: #b9cdd4 !important; }

      .block-container {
        max-width: 1250px;
        padding: 2.1rem 2.7rem 3.5rem;
      }

      [data-testid="stSidebar"] {
        background: #0b1928;
        border-right: 1px solid var(--line);
      }

      [data-testid="stSidebar"] > div:first-child {
        background:
          radial-gradient(circle at 5% 0%, rgba(67, 203, 177, .17), transparent 22rem),
          #0b1928;
      }

      [data-testid="stSidebar"] .block-container {
        padding: 1.35rem 1rem 2rem;
      }

      h1, h2, h3, p, span, label, div { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      h1, h2, h3 { color: var(--ink) !important; letter-spacing: -.03em; }
      h2 { font-size: 1.12rem !important; }

      .brand-lockup {
        padding: .35rem .25rem 1.25rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1rem;
      }

      .brand-mark {
        align-items: center;
        background: linear-gradient(135deg, var(--mint), var(--cyan));
        border-radius: 10px;
        color: #09202a;
        display: inline-flex;
        font-size: 1.25rem;
        font-weight: 900;
        height: 34px;
        justify-content: center;
        margin-right: .55rem;
        width: 34px;
      }

      .brand-name { color: var(--ink); font-size: .95rem; font-weight: 800; letter-spacing: .16em; vertical-align: 7px; }
      .brand-subtitle { color: var(--muted); font-size: .66rem; letter-spacing: .15em; margin-top: .6rem; text-transform: uppercase; }

      .sidebar-note {
        background: rgba(118, 230, 194, .07);
        border: 1px solid rgba(118, 230, 194, .16);
        border-radius: 12px;
        color: #bad1d8;
        font-size: .76rem;
        line-height: 1.55;
        margin: .95rem 0 1.15rem;
        padding: .8rem .85rem;
      }

      [data-testid="stSidebar"] .stExpander {
        background: rgba(255, 255, 255, .025);
        border: 1px solid var(--line);
        border-radius: 12px;
        margin: .55rem 0;
        overflow: hidden;
      }

      [data-testid="stSidebar"] [data-testid="stExpanderDetails"] { border: 0; }
      [data-testid="stSidebar"] summary { color: #dbe9ec !important; font-size: .83rem; font-weight: 700; letter-spacing: .01em; }
      [data-testid="stSidebar"] label { color: #b9cbd3 !important; font-size: .76rem !important; font-weight: 600 !important; }
      [data-testid="stSidebar"] .stCaption { color: #7f9aa8 !important; font-size: .7rem !important; }

      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        background: rgba(255,255,255,.055) !important;
        border-color: rgba(181, 218, 230, .16) !important;
        color: #edf7f8 !important;
      }

      [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--mint); }
      [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button,
      .stButton > button {
        background: linear-gradient(100deg, #78e7c3, #67c7ee) !important;
        border: 0 !important;
        border-radius: 10px !important;
        box-shadow: 0 12px 26px rgba(73, 195, 176, .18);
        color: #08202a !important;
        font-size: .9rem !important;
        font-weight: 800 !important;
        min-height: 2.8rem;
      }

      [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover,
      .stButton > button:hover {
        box-shadow: 0 16px 32px rgba(73, 195, 176, .28);
        transform: translateY(-1px);
      }

      .hero {
        background:
          linear-gradient(115deg, rgba(16, 40, 55, .96), rgba(13, 31, 47, .84)),
          linear-gradient(145deg, rgba(110, 235, 198, .13), transparent);
        border: 1px solid var(--line);
        border-radius: 22px;
        margin-bottom: 1.4rem;
        overflow: hidden;
        padding: 2.2rem 2.25rem;
        position: relative;
      }

      .hero:after {
        background: radial-gradient(circle, rgba(114, 230, 194, .17), transparent 67%);
        content: "";
        height: 280px;
        position: absolute;
        right: -65px;
        top: -140px;
        width: 280px;
      }

      .eyebrow { color: var(--mint); font-size: .71rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
      .hero-title { color: var(--ink); font-size: clamp(2rem, 4vw, 3.05rem); font-weight: 790; letter-spacing: -.055em; line-height: 1.07; margin: .55rem 0 .72rem; max-width: 710px; position: relative; z-index: 1; }
      .hero-copy { color: #aabec8; font-size: .98rem; line-height: 1.65; max-width: 660px; position: relative; z-index: 1; }
      .hero-pills { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.35rem; position: relative; z-index: 1; }
      .hero-pill { background: rgba(130, 230, 202, .08); border: 1px solid rgba(130, 230, 202, .17); border-radius: 100px; color: #cae7e1; font-size: .73rem; font-weight: 650; padding: .37rem .7rem; }
      .hero-pill b { color: var(--mint); font-weight: 800; margin-right: .3rem; }

      .section-kicker { color: var(--muted); font-size: .72rem; font-weight: 800; letter-spacing: .14em; margin: .15rem 0 .4rem; text-transform: uppercase; }
      .section-title { color: var(--ink); font-size: 1.38rem; font-weight: 760; letter-spacing: -.03em; margin: 0 0 1rem; }

      .empty-board, .model-board {
        background: rgba(16, 37, 54, .74);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem;
      }

      .empty-icon {
        align-items: center;
        background: linear-gradient(135deg, rgba(118,230,194,.2), rgba(97,199,245,.15));
        border: 1px solid rgba(129, 230, 205, .22);
        border-radius: 13px;
        color: var(--mint);
        display: inline-flex;
        font-size: 1.2rem;
        height: 43px;
        justify-content: center;
        margin-bottom: .85rem;
        width: 43px;
      }

      .empty-title { color: #edf7f8; font-size: 1.04rem; font-weight: 750; margin: 0 0 .4rem; }
      .empty-copy { color: var(--muted); font-size: .85rem; line-height: 1.55; margin: 0; }
      .micro-card {
        background: rgba(255,255,255,.025);
        border: 1px solid var(--line);
        border-radius: 14px;
        height: 100%;
        padding: 1rem;
      }
      .micro-label { color: #7893a1; font-size: .68rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
      .micro-value { color: #e8f5f2; font-size: 1.15rem; font-weight: 780; margin-top: .42rem; }
      .micro-copy { color: #91a9b5; font-size: .76rem; line-height: 1.45; margin-top: .35rem; }

      .result-shell {
        background: linear-gradient(125deg, rgba(19, 44, 60, .95), rgba(14, 31, 47, .88));
        border: 1px solid rgba(126, 213, 204, .22);
        border-radius: 22px;
        margin-bottom: 1rem;
        overflow: hidden;
        padding: 1.35rem;
      }

      .score-orb {
        align-items: center;
        background: #0a1926;
        border: 9px solid #6ee3c0;
        border-radius: 50%;
        box-shadow: 0 0 0 8px rgba(110, 227, 192, .08), 0 18px 45px rgba(0,0,0,.22);
        display: flex;
        flex-direction: column;
        height: 148px;
        justify-content: center;
        margin: .2rem auto .95rem;
        text-align: center;
        width: 148px;
      }
      .score-orb.medium { border-color: var(--amber); box-shadow: 0 0 0 8px rgba(248,198,107,.08), 0 18px 45px rgba(0,0,0,.22); }
      .score-orb.high { border-color: var(--rose); box-shadow: 0 0 0 8px rgba(255,140,159,.08), 0 18px 45px rgba(0,0,0,.22); }
      .score-number { color: #f3faf8; font-size: 2.55rem; font-weight: 820; letter-spacing: -.08em; line-height: 1; }
      .score-label { color: #87a4ae; font-size: .62rem; font-weight: 800; letter-spacing: .16em; margin-top: .36rem; text-transform: uppercase; }

      .risk-badge { align-items: center; border: 1px solid; border-radius: 100px; display: inline-flex; font-size: .7rem; font-weight: 800; gap: .35rem; letter-spacing: .1em; padding: .38rem .65rem; text-transform: uppercase; }
      .risk-badge.low { background: rgba(118,230,194,.1); border-color: rgba(118,230,194,.26); color: var(--mint); }
      .risk-badge.medium { background: rgba(248,198,107,.1); border-color: rgba(248,198,107,.28); color: var(--amber); }
      .risk-badge.high { background: rgba(255,140,159,.1); border-color: rgba(255,140,159,.28); color: var(--rose); }
      .risk-title { color: #eff8f8; font-size: 1.35rem; font-weight: 780; letter-spacing: -.035em; margin: .7rem 0 .25rem; }
      .risk-copy { color: #9eb6c1; font-size: .86rem; line-height: 1.55; margin: 0; }
      .probability-value { color: #f3faf8; font-size: 3.1rem; font-weight: 800; letter-spacing: -.08em; line-height: 1; margin: .55rem 0 .55rem; }
      .probability-label { color: #8ca7b3; font-size: .68rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }

      .risk-track { background: rgba(255,255,255,.09); border-radius: 100px; height: 9px; margin: .65rem 0 .5rem; overflow: hidden; }
      .risk-fill { background: linear-gradient(90deg, #6ee3c0 0%, #f8c66b 59%, #ff8c9f 100%); border-radius: inherit; height: 100%; }
      .risk-scale { color: #809ba8; display: flex; font-size: .67rem; justify-content: space-between; }

      [data-testid="stMetric"] {
        background: rgba(255,255,255,.025);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: .85rem 1rem;
      }
      [data-testid="stMetricLabel"] { color: #87a2ae !important; font-size: .7rem !important; font-weight: 800 !important; letter-spacing: .08em; text-transform: uppercase; }
      [data-testid="stMetricValue"] { color: #edf8f6 !important; font-size: 1.42rem !important; font-weight: 760 !important; }

      .disclaimer {
        border-top: 1px solid var(--line);
        color: #7f99a7;
        font-size: .74rem;
        line-height: 1.55;
        margin-top: 1.3rem;
        padding-top: .95rem;
      }

      [data-testid="stExpander"] {
        background: rgba(16,37,54,.62);
        border: 1px solid var(--line);
        border-radius: 14px;
      }
      [data-testid="stExpander"] summary { color: #dcecf0 !important; font-size: .83rem; font-weight: 700; }
      [data-testid="stAlert"] { border-radius: 12px; }

      #MainMenu, footer { visibility: hidden; }
      @media (max-width: 800px) {
        .block-container { padding: 1rem 1rem 2.5rem; }
        .hero { padding: 1.5rem; }
        .hero-title { font-size: 2.1rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_bundle():
    return load_bundle(DEFAULT_BUNDLE_PATH)


UNKNOWN_CHOICE = "__unknown__"


def format_choice(group: str):
    """Thai label where one exists, otherwise the Home Credit value itself.

    Categories come from the bundle, so most of them have no hand-written Thai
    label. Showing the raw value is better than hiding the option: hiding it is
    what forced officers to pick a wrong category.
    """

    def render(value: object) -> str:
        if value == UNKNOWN_CHOICE:
            return "ไม่ระบุ / ไม่พบข้อมูล"
        return LABELS.get(group, {}).get(value, str(value))

    return render


def category_options(bundle: dict, column: str, group: str) -> list[str]:
    options = list(bundle.get("input_categories", {}).get(column, ()))
    if not options:
        options = list(LABELS.get(group, {}))
    if bundle.get("input_missing_rates", {}).get(column, 0.0) > 0:
        options = [UNKNOWN_CHOICE, *options]
    return options


def category_value(choice: object) -> object:
    return np.nan if choice == UNKNOWN_CHOICE else choice


def render_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="brand-lockup">
          <span class="brand-mark">◈</span><span class="brand-name">AURELIS</span>
          <div class="brand-subtitle">Credit intelligence platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(bundle: dict) -> None:
    validation_auc = bundle.get("metrics", {}).get("roc_auc")
    auc_text = f"{validation_auc:.3f}" if validation_auc is not None else "—"
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Calibrated decision support</div>
          <div class="hero-title">ตัดสินใจสินเชื่อ<br>อย่างมั่นใจและมีบริบท</div>
          <div class="hero-copy">
            ประเมินความเสี่ยงจากข้อมูลผู้สมัครด้วยโมเดลที่ผ่านการ calibration
            เพื่อให้ทีมสินเชื่อเห็นสัญญาณที่จำเป็นก่อนเข้าสู่กระบวนการพิจารณา
          </div>
          <div class="hero-pills">
            <span class="hero-pill"><b>◉</b> Calibrated probability</span>
            <span class="hero-pill"><b>◈</b> ROC-AUC {auc_text}</span>
            <span class="hero-pill"><b>✓</b> Human review ready</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(bundle: dict) -> None:
    st.markdown('<div class="section-kicker">Assessment workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">พร้อมวิเคราะห์ผู้สมัคร</div>', unsafe_allow_html=True)
    left, middle, right = st.columns([1.5, 1, 1])
    with left:
        st.markdown(
            """
            <div class="empty-board">
              <div class="empty-icon">↗</div>
              <div class="empty-title">เริ่มจากกรอกข้อมูลผู้สมัคร</div>
              <p class="empty-copy">
                ใช้แถบด้านซ้ายเพื่อระบุข้อมูลพื้นฐาน ภาระสินเชื่อ และข้อมูลเครดิตภายนอก
                จากนั้นกดปุ่มประเมินเพื่อสร้าง risk profile
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with middle:
        st.markdown(
            """
            <div class="micro-card">
              <div class="micro-label">Output</div>
              <div class="micro-value">PD + Score</div>
              <div class="micro-copy">Default probability พร้อมดัชนีความเสี่ยงแบบ odds-based</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        selected = len(bundle.get("selected_feature_names", []))
        st.markdown(
            f"""
            <div class="micro-card">
              <div class="micro-label">Model coverage</div>
              <div class="micro-value">{selected} features</div>
              <div class="micro-copy">ฟีเจอร์ที่ผ่านการคัดเลือกสำหรับการประเมิน</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result(assessment: dict, bundle: dict) -> None:
    probability = assessment["probability"]
    score = assessment["score"]
    level = assessment["level"]
    risk = RISK_PRESENTATION[level]
    threshold = assessment["threshold"]
    meter_width = min(max(probability / max(threshold * 1.55, 0.01) * 100, 3), 100)

    st.markdown('<div class="section-kicker">Decision snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ผลการประเมินความเสี่ยง</div>', unsafe_allow_html=True)
    col_score, col_risk, col_probability = st.columns([1, 1.25, 1.1], gap="large")
    with col_score:
        st.markdown(
            f"""
            <div class="result-shell">
              <div class="score-orb {risk["class_name"]}">
                <div class="score-number">{score}</div>
                <div class="score-label">Risk index</div>
              </div>
              <div class="probability-label" style="text-align:center">300–850 odds-based index</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_risk:
        st.markdown(
            f"""
            <div class="result-shell">
              <span class="risk-badge {risk["class_name"]}"><span>{risk["icon"]}</span>{risk["eyebrow"]}</span>
              <div class="risk-title">{risk["title"]}</div>
              <p class="risk-copy">{risk["description"]}</p>
              <div class="disclaimer">เกณฑ์ตรวจสอบของโมเดลอยู่ที่ {threshold:.2%}
              และผลลัพธ์ควรใช้ประกอบ human review เสมอ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_probability:
        st.markdown(
            f"""
            <div class="result-shell">
              <div class="probability-label">Estimated default probability</div>
              <div class="probability-value">{probability:.2%}</div>
              <div class="risk-track"><div class="risk-fill" style="width:{meter_width:.1f}%"></div></div>
              <div class="risk-scale"><span>ต่ำ</span><span>เกณฑ์ {threshold:.1%}</span><span>สูง</span></div>
              <div class="disclaimer">Probability นี้ผ่าน calibration ด้วยข้อมูล validation
              เพื่อให้ตีความความเสี่ยงได้ดีขึ้น</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metrics = bundle.get("metrics", {})
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Decision threshold", f"{threshold:.2%}")
    metric_2.metric("Validation ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    metric_3.metric("Validation Brier score", f"{metrics.get('brier_score', 0):.4f}")

    with st.expander("ดูรายละเอียดโมเดลและข้อควรทราบ"):
        detail_col, note_col = st.columns([1, 1.25])
        with detail_col:
            st.caption("Model governance")
            st.write(
                {
                    "bundle_version": bundle.get("bundle_version"),
                    "selected_features": len(bundle.get("selected_feature_names", [])),
                    "trained_at": bundle.get("created_at"),
                }
            )
        with note_col:
            st.caption("Deployment note")
            st.write(
                "ฟีเจอร์เครดิตบูโรหรือประวัติการชำระที่ไม่ได้รับผ่านแบบฟอร์ม "
                "จะใช้ค่ากลางจาก training split ใน demo นี้ "
                "ระบบ production ควรเชื่อมต่อข้อมูลจริงที่ได้รับอนุญาต"
            )


def collect_sidebar_inputs(bundle: dict) -> tuple[bool, dict]:
    render_brand()
    st.sidebar.markdown(
        '<div class="sidebar-note">กรอกข้อมูลให้ใกล้เคียงความเป็นจริงมากที่สุด '
        'เพื่อให้แบบจำลองสร้างภาพรวมความเสี่ยงที่มีประโยชน์ต่อการพิจารณา</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar.form("applicant_form", border=False):
        with st.expander("01  ผู้สมัครและครอบครัว", expanded=True):
            gender = st.selectbox(
                "เพศ",
                category_options(bundle, "CODE_GENDER", "gender"),
                format_func=format_choice("gender"),
            )
            age = st.slider("อายุ (ปี)", 20, 70, 35)
            education = st.selectbox(
                "ระดับการศึกษา",
                category_options(bundle, "NAME_EDUCATION_TYPE", "education"),
                format_func=format_choice("education"),
            )
            family_status = st.selectbox(
                "สถานะครอบครัว",
                category_options(bundle, "NAME_FAMILY_STATUS", "family"),
                format_func=format_choice("family"),
            )
            housing_type = st.selectbox(
                "ประเภทที่อยู่อาศัย",
                category_options(bundle, "NAME_HOUSING_TYPE", "housing"),
                format_func=format_choice("housing"),
            )
            children = st.number_input("จำนวนบุตร", min_value=0, max_value=15, value=0)

        with st.expander("02  ข้อมูลการเงิน", expanded=True):
            income = st.number_input(
                "รายได้ต่อปี", min_value=1_000.0, value=150_000.0, step=5_000.0, format="%.0f"
            )
            credit = st.number_input(
                "วงเงินสินเชื่อ", min_value=1_000.0, value=300_000.0, step=5_000.0, format="%.0f"
            )
            annuity = st.number_input(
                "ยอดผ่อนต่อปี", min_value=1.0, value=20_000.0, step=500.0, format="%.0f"
            )
            goods_price = st.number_input(
                "ราคาสินค้า", min_value=1.0, value=270_000.0, step=5_000.0, format="%.0f"
            )

        with st.expander("03  การทำงานและทรัพย์สิน"):
            income_type = st.selectbox(
                "ประเภทรายได้",
                category_options(bundle, "NAME_INCOME_TYPE", "income"),
                format_func=format_choice("income"),
            )
            # ในข้อมูลฝึก DAYS_EMPLOYED ของ Pensioner และ Unemployed เป็น sentinel
            # 365243 ครบ 100% และไม่พบในประเภทอื่นเลย preprocessing แปลงเป็นค่าว่าง
            # พร้อมตั้ง DAYS_EMPLOYED_ANOM = 1 ฟอร์มจึงต้องส่งค่าแบบเดียวกัน
            not_employed = income_type in {"Pensioner", "Unemployed"}
            employment_years = st.slider(
                "อายุงาน (ปี)",
                0,
                50,
                5,
                disabled=not_employed,
                help=(
                    "ผู้รับบำนาญและผู้ไม่ได้ทำงานถูกบันทึกเป็นไม่มีข้อมูลการจ้างงาน"
                    if not_employed
                    else None
                ),
            )
            occupation = st.selectbox(
                "อาชีพ",
                category_options(bundle, "OCCUPATION_TYPE", "occupation"),
                format_func=format_choice("occupation"),
                help="เลือกไม่ระบุได้ ข้อมูลฝึกไม่มีอาชีพถึง 31% ของผู้สมัคร",
            )
            organization = st.selectbox(
                "ประเภทองค์กร",
                category_options(bundle, "ORGANIZATION_TYPE", "organization"),
                format_func=format_choice("organization"),
            )
            flag_col_1, flag_col_2 = st.columns(2)
            own_car = flag_col_1.checkbox("มีรถยนต์")
            own_realty = flag_col_2.checkbox("มีอสังหาฯ", value=True)
            work_phone = st.checkbox("มีเบอร์โทรศัพท์ที่ทำงาน", value=True)

        with st.expander("04  คะแนนเครดิตภายนอก"):
            st.caption(
                "เจ้าหน้าที่สินเชื่อกรอกค่าที่ดึงจากระบบเครดิตภายนอกเท่านั้น ห้ามประมาณค่าเอง "
                "หากระบบไม่มีข้อมูลของผู้สมัครรายนี้ ให้ติ๊ก ไม่พบในระบบ "
                "แบบจำลองจะได้รับค่าว่างซึ่งตรงกับที่ข้อมูลฝึกบันทึกไว้ ไม่ใช่ค่าที่เดาขึ้นมา"
            )
            ext_sources: dict[str, object] = {}
            for number in (1, 2, 3):
                value_column, missing_column = st.columns([2, 1])
                unavailable = missing_column.checkbox(
                    "ไม่พบในระบบ", key=f"ext_source_missing_{number}"
                )
                entered = value_column.number_input(
                    f"แหล่งข้อมูล {number}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.01,
                    format="%.4f",
                    disabled=unavailable,
                    key=f"ext_source_value_{number}",
                )
                ext_sources[f"EXT_SOURCE_{number}"] = (
                    np.nan if unavailable else float(entered)
                )

        submitted = st.form_submit_button("ประเมินความเสี่ยง", type="primary", use_container_width=True)

    values = {
        "CODE_GENDER": category_value(gender),
        "DAYS_BIRTH": -age * 365.25,
        "NAME_EDUCATION_TYPE": category_value(education),
        "NAME_FAMILY_STATUS": category_value(family_status),
        "NAME_HOUSING_TYPE": category_value(housing_type),
        "CNT_CHILDREN": children,
        "CNT_FAM_MEMBERS": children + (2 if family_status in {"Married", "Civil marriage"} else 1),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "NAME_INCOME_TYPE": category_value(income_type),
        "DAYS_EMPLOYED": np.nan if not_employed else -employment_years * 365.25,
        "DAYS_EMPLOYED_ANOM": 1 if not_employed else 0,
        "OCCUPATION_TYPE": category_value(occupation),
        "ORGANIZATION_TYPE": category_value(organization),
        "FLAG_OWN_CAR": "Y" if own_car else "N",
        "FLAG_OWN_REALTY": "Y" if own_realty else "N",
        "FLAG_WORK_PHONE": int(work_phone),
        **ext_sources,
    }
    return submitted, values


def main() -> None:
    try:
        bundle = get_bundle()
    except FileNotFoundError:
        st.error("ยังไม่มี model bundle รุ่นใหม่")
        st.code("python src/model_training.py", language="bash")
        st.stop()
    except Exception as exc:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {exc}")
        st.stop()

    submitted, values = collect_sidebar_inputs(bundle)

    if submitted:
        # The form and the bundle must agree exactly. A drift here used to be
        # invisible: unknown keys were dropped and the applicant was scored on
        # training defaults instead of on what the officer entered.
        expected = set(bundle.get("served_columns", ()))
        supplied = set(values)
        if expected != supplied:
            st.error("แบบฟอร์มกับโมเดลไม่ตรงกัน กรุณาเทรนโมเดลใหม่ก่อนใช้งาน")
            st.write(
                {
                    "ฟอร์มส่งมาแต่โมเดลไม่รู้จัก": sorted(supplied - expected),
                    "โมเดลต้องการแต่ฟอร์มไม่ส่ง": sorted(expected - supplied),
                }
            )
            st.stop()

    if submitted:
        try:
            probability = predict_default_probability(values, bundle)
            st.session_state["assessment"] = {
                "probability": probability,
                "score": probability_to_score(probability),
                "level": risk_level(probability, bundle),
                "threshold": float(bundle["decision_threshold"]),
            }
        except Exception as exc:
            st.error(f"ประมวลผลไม่สำเร็จ: {exc}")
            st.stop()

    render_hero(bundle)
    assessment = st.session_state.get("assessment")
    if assessment:
        render_result(assessment, bundle)
    else:
        render_empty_state(bundle)

    st.markdown(
        '<div class="disclaimer">AURELIS is a decision-support interface. '
        'ผลลัพธ์ใช้เพื่อประกอบการพิจารณา ไม่ใช่การอนุมัติหรือปฏิเสธสินเชื่อโดยอัตโนมัติ</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
