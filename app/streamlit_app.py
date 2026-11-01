"""Premium Streamlit interface for the calibrated credit-risk model."""

from __future__ import annotations

import json
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
    "education": {
        "Secondary / secondary special": "มัธยมศึกษา",
        "Higher education": "ปริญญาตรี",
        "Incomplete higher": "ไม่จบปริญญาตรี",
        "Lower secondary": "มัธยมต้น",
        "Academic degree": "ปริญญาโทขึ้นไป",
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


# Design rationale for the CSS below (kept as a real comment, not injected into
# the page, because a numbered list inside an HTML comment gets parsed as
# Markdown by st.markdown and breaks out of the raw <style> block):
#
# - Thai script stacks tone and vowel marks tightly above and below each
#   consonant. Negative letter-spacing on Thai headings (the previous design
#   used -0.03em to -0.08em) crowds those marks and makes the text genuinely
#   harder to read, not just less elegant. Every Thai text class below uses
#   normal letter-spacing; tracking is reserved for short, all-caps English
#   labels only.
# - "Inter" has no Thai glyphs, so Thai text was silently falling back to
#   whatever the OS supplies, with no control over shaping quality. Noto Sans
#   Thai is loaded explicitly and listed first for anything that renders
#   Thai copy.
# - Body and label text previously ranged from 0.62rem to 0.86rem (10-14px)
#   in muted blue-grey on a dark navy background. Nothing below 0.8rem (13px)
#   is used now, and secondary text uses a lighter, higher-contrast tone.
# - Decorative radial-gradient glows, double gradient layers and the
#   thick-ringed "score orb" added visual noise without adding information.
#   They are replaced with flat surfaces and a single colour accent per risk
#   level.
# <style> must be the very first tag in this markdown string. Streamlit's
# markdown parser treats a block starting with <style>/<script>/<pre> as raw
# text up to the literal </style>, blank lines included. A block starting
# with <link> instead (as this used to) is parsed under different HTML-block
# rules where an interior blank line ends the block early, dumping the rest
# of the CSS as visible text on the page. Font <link> tags are appended after
# </style> instead, in their own blank-line-free block.
st.markdown(
    """
    <style>
      :root {
        --bg: #0d1420;
        --surface: #161f2e;
        --surface-2: #1d2839;
        --border: #2e3b4f;
        --text: #f5f8fb;
        --text-secondary: #c9d3e0;
        --text-muted: #96a7bc;
        --accent: #3ecf8e;
        --accent-strong: #2fae77;
        --low-bg: #123527;
        --low-border: #1f6b48;
        --low-text: #7ee9b8;
        --medium-bg: #3a2c10;
        --medium-border: #8a6a1f;
        --medium-text: #f5c451;
        --high-bg: #3a1620;
        --high-border: #8a2c3f;
        --high-text: #ff93a6;
      }

      html, body, .stApp {
        background: var(--bg) !important;
        color: var(--text);
      }

      /* Excludes Streamlit's Material icon spans: forcing a text font onto
         them turns icon glyphs (e.g. an expander's chevron) into the raw
         ligature text "keyboard_arrow_down" instead of an arrow. */
      *:not([data-testid="stIconMaterial"]) {
        font-family: "Noto Sans Thai", "Inter", ui-sans-serif, system-ui, -apple-system,
          "Segoe UI", sans-serif !important;
      }

      body, p, li, .stMarkdown { line-height: 1.7; }

      header[data-testid="stHeader"] {
        background: var(--bg) !important;
        border-bottom: 1px solid var(--border);
      }
      [data-testid="stToolbar"] { background: transparent !important; }
      header[data-testid="stHeader"] button { color: var(--text-secondary) !important; }

      .block-container { max-width: 1180px; padding: 2rem 2.2rem 3.5rem; }

      [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
      }
      [data-testid="stSidebar"] .block-container { padding: 1.3rem 1rem 2rem; }

      h1, h2, h3 { color: var(--text) !important; letter-spacing: normal; }

      .brand-lockup {
        padding: .35rem .25rem 1.1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
      }
      .brand-mark {
        align-items: center;
        background: var(--accent);
        border-radius: 10px;
        color: #08201a;
        display: inline-flex;
        font-size: 1.25rem;
        font-weight: 800;
        height: 34px;
        justify-content: center;
        margin-right: .55rem;
        width: 34px;
      }
      .brand-name { color: var(--text); font-size: 1rem; font-weight: 800; letter-spacing: .1em; vertical-align: 7px; }
      .brand-subtitle { color: var(--text-muted); font-size: .8rem; letter-spacing: .04em; margin-top: .5rem; }

      .sidebar-note {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        color: var(--text-secondary);
        font-size: .9rem;
        line-height: 1.65;
        margin: .9rem 0 1.1rem;
        padding: .85rem .9rem;
      }

      [data-testid="stSidebar"] .stExpander {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 12px;
        margin: .55rem 0;
        overflow: hidden;
      }
      [data-testid="stSidebar"] [data-testid="stExpanderDetails"] { border: 0; }
      /* Streamlit renders the expander header as a plain <summary> with its
         own light background (~#f8f9fb) regardless of app theme. Overriding
         only the text colour left near-white text on that near-white
         background, which is why section headers were invisible. */
      [data-testid="stSidebar"] summary {
        background: var(--surface-2) !important;
        border-radius: 8px;
        color: var(--text) !important;
        font-size: .95rem;
        font-weight: 700;
        padding: .5rem .6rem !important;
      }
      [data-testid="stSidebar"] summary svg,
      [data-testid="stSidebar"] summary [data-testid="stIconMaterial"] {
        color: var(--text-muted) !important;
      }
      [data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
        font-size: .88rem !important;
        font-weight: 600 !important;
      }
      [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: .82rem !important;
        line-height: 1.6 !important;
      }

      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        background: var(--surface) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        font-size: .92rem !important;
      }
      [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: var(--accent);
        font-size: .88rem;
        font-weight: 700;
      }

      [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button,
      .stButton > button {
        background: var(--accent) !important;
        border: 0 !important;
        border-radius: 10px !important;
        color: #08201a !important;
        font-size: .98rem !important;
        font-weight: 700 !important;
        min-height: 2.9rem;
      }
      [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover,
      .stButton > button:hover { background: var(--accent-strong) !important; }

      .hero {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: 3px solid var(--accent);
        border-radius: 16px;
        margin-bottom: 1.4rem;
        padding: 1.9rem 2rem;
      }
      .eyebrow {
        color: var(--accent);
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
      }
      .hero-title {
        color: var(--text);
        font-size: clamp(1.7rem, 3.4vw, 2.35rem);
        font-weight: 700;
        letter-spacing: normal;
        line-height: 1.35;
        margin: .6rem 0 .8rem;
        max-width: 44rem;
      }
      .hero-copy {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.75;
        max-width: 44rem;
      }
      .hero-stats {
        border-top: 1px solid var(--border);
        color: var(--text-muted);
        display: flex;
        flex-wrap: wrap;
        font-size: .88rem;
        gap: .5rem 1.3rem;
        margin-top: 1.2rem;
        padding-top: 1.1rem;
      }
      .hero-stats b { color: var(--text); font-weight: 700; }

      .section-kicker {
        color: var(--text-muted);
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .08em;
        margin: .15rem 0 .35rem;
        text-transform: uppercase;
      }
      .section-title { color: var(--text); font-size: 1.4rem; font-weight: 700; letter-spacing: normal; margin: 0 0 1rem; }

      .board {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        height: 100%;
        padding: 1.15rem;
      }
      .board-title { color: var(--text); font-size: 1.05rem; font-weight: 700; margin: 0 0 .5rem; }
      .board-copy { color: var(--text-secondary); font-size: .92rem; line-height: 1.7; margin: 0; }
      .board-label { color: var(--text-muted); font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
      .board-value { color: var(--text); font-size: 1.2rem; font-weight: 700; margin-top: .4rem; }

      .result-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        border-top: 4px solid var(--border);
        height: 100%;
        margin-bottom: 1rem;
        padding: 1.3rem 1.35rem;
      }
      .result-card.low { border-top-color: var(--low-text); }
      .result-card.medium { border-top-color: var(--medium-text); }
      .result-card.high { border-top-color: var(--high-text); }

      .card-label {
        color: var(--text-muted);
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
      }

      .score-number { color: var(--text); font-size: 2.6rem; font-weight: 800; letter-spacing: -.01em; line-height: 1.15; margin-top: .3rem; }
      .score-caption { color: var(--text-muted); font-size: .85rem; margin-top: .3rem; }

      .risk-badge {
        border: 1px solid;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        font-size: .82rem;
        font-weight: 700;
        letter-spacing: .04em;
        padding: .3rem .7rem;
      }
      .risk-badge.low { background: var(--low-bg); border-color: var(--low-border); color: var(--low-text); }
      .risk-badge.medium { background: var(--medium-bg); border-color: var(--medium-border); color: var(--medium-text); }
      .risk-badge.high { background: var(--high-bg); border-color: var(--high-border); color: var(--high-text); }

      .risk-title { color: var(--text); font-size: 1.3rem; font-weight: 700; letter-spacing: normal; line-height: 1.4; margin: .7rem 0 .3rem; }
      .risk-copy { color: var(--text-secondary); font-size: .92rem; line-height: 1.7; margin: 0; }

      .probability-value { color: var(--text); font-size: 2.6rem; font-weight: 800; letter-spacing: -.01em; line-height: 1.15; margin: .3rem 0 .7rem; }

      .risk-track { background: var(--surface-2); border: 1px solid var(--border); border-radius: 100px; height: 10px; margin: .6rem 0 .5rem; overflow: hidden; }
      .risk-fill { background: linear-gradient(90deg, var(--low-text), var(--medium-text), var(--high-text)); border-radius: inherit; height: 100%; }
      .risk-scale { color: var(--text-muted); display: flex; font-size: .8rem; justify-content: space-between; margin-bottom: .8rem; }

      .note { color: var(--text-muted); font-size: .84rem; line-height: 1.65; margin-top: .9rem; }

      [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: .9rem 1.1rem;
      }
      [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: .78rem !important;
        font-weight: 700 !important;
        letter-spacing: .04em;
        text-transform: uppercase;
      }
      [data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.5rem !important; font-weight: 700 !important; }

      .disclaimer {
        border-top: 1px solid var(--border);
        color: var(--text-muted);
        font-size: .85rem;
        line-height: 1.7;
        margin-top: 1.3rem;
        padding-top: 1rem;
      }

      [data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
      }
      [data-testid="stExpander"] summary {
        background: var(--surface) !important;
        color: var(--text) !important;
        font-size: .92rem;
        font-weight: 700;
        padding: .6rem .8rem !important;
      }
      [data-testid="stExpander"] summary svg,
      [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
        color: var(--text-muted) !important;
      }
      [data-testid="stAlert"] { border-radius: 10px; font-size: .92rem; }

      #MainMenu, footer { visibility: hidden; }

      /* Streamlit inherits overflow-wrap: break-word, which splits a word
         mid-character in a narrow column ("features" became "feature" + "s").
         Numbers and short values should wrap between words or not at all. */
      .score-number, .probability-value, .board-value, .card-label {
        overflow-wrap: normal;
      }

      /* The three result columns share the content area beside a 300px
         sidebar. Below roughly 1100px that leaves each card under 250px,
         where "RISK INDEX (300-850)" wraps onto four lines and the score
         digits stack. Stack the cards full-width instead. Scoped to the main
         area: the sidebar has its own two-column rows (external-score value
         beside its "not found" checkbox) that must stay side by side. */
      @media (max-width: 1100px) {
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
          flex: 1 1 100% !important;
          min-width: 100% !important;
        }
        .result-card { margin-bottom: .75rem; }
      }

      @media (max-width: 800px) {
        .block-container { padding: 1rem 1rem 2.5rem; }
        .hero { padding: 1.4rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600;700;800'
    '&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
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


def input_fingerprint(values: dict) -> str:
    """Stable identity for a set of form inputs, NaN included.

    A plain dict comparison would never match once an input is missing,
    because NaN != NaN; serialising maps every NaN to the same token.
    """
    return json.dumps(values, sort_keys=True, default=str)


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
    # Two decimals: the bootstrap 95% interval is about 0.019 wide, so a third
    # digit would claim precision one validation split cannot support.
    auc_text = f"{validation_auc:.2f}" if validation_auc is not None else "—"
    n_features = len(bundle.get("selected_feature_names", []))
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Calibrated decision support</div>
          <div class="hero-title">ตัดสินใจสินเชื่ออย่างมั่นใจและมีบริบท</div>
          <div class="hero-copy">
            ประเมินความเสี่ยงจากข้อมูลผู้สมัครด้วยโมเดลที่ผ่านการ calibration
            เพื่อให้ทีมสินเชื่อเห็นสัญญาณที่จำเป็นก่อนเข้าสู่กระบวนการพิจารณา
          </div>
          <div class="hero-stats">
            <span><b>Calibrated</b> probability</span>
            <span><b>{auc_text}</b> ROC-AUC</span>
            <span><b>{n_features}</b> features</span>
            <span><b>✓</b> Human review ready</span>
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
            <div class="board">
              <div class="board-title">เริ่มจากกรอกข้อมูลผู้สมัคร</div>
              <p class="board-copy">
                ใช้แถบด้านซ้ายเพื่อระบุข้อมูลพื้นฐาน ภาระสินเชื่อ และคะแนนเครดิตภายนอก
                จากนั้นกดปุ่มประเมินความเสี่ยงเพื่อสร้าง risk profile
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with middle:
        st.markdown(
            """
            <div class="board">
              <div class="board-label">Output</div>
              <div class="board-value">PD + Score</div>
              <p class="board-copy">Default probability พร้อมดัชนีความเสี่ยงแบบ odds-based</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        selected = len(bundle.get("selected_feature_names", []))
        st.markdown(
            f"""
            <div class="board">
              <div class="board-label">Model coverage</div>
              <div class="board-value">{selected} features</div>
              <p class="board-copy">ฟีเจอร์ที่ผ่านการคัดเลือกสำหรับการประเมิน</p>
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
            <div class="result-card {risk["class_name"]}">
              <div class="card-label">Risk index (300–850)</div>
              <div class="score-number">{score}</div>
              <div class="score-caption">ดัชนีแบบ odds-based ไม่ใช่คะแนนเครดิตบูโร</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_risk:
        st.markdown(
            f"""
            <div class="result-card {risk["class_name"]}">
              <span class="risk-badge {risk["class_name"]}"><span>{risk["icon"]}</span>{risk["eyebrow"]}</span>
              <div class="risk-title">{risk["title"]}</div>
              <p class="risk-copy">{risk["description"]}</p>
              <p class="note">เกณฑ์ตรวจสอบของโมเดลอยู่ที่ {threshold:.2%}
              ผลลัพธ์นี้ควรใช้ประกอบ human review เสมอ</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_probability:
        st.markdown(
            f"""
            <div class="result-card {risk["class_name"]}">
              <div class="card-label">Estimated default probability</div>
              <div class="probability-value">{probability:.2%}</div>
              <div class="risk-track"><div class="risk-fill" style="width:{meter_width:.1f}%"></div></div>
              <div class="risk-scale"><span>ต่ำ</span><span>เกณฑ์ {threshold:.1%}</span><span>สูง</span></div>
              <p class="note">Probability นี้ผ่าน calibration ด้วยข้อมูล validation
              เพื่อให้ตีความความเสี่ยงได้ดีขึ้น</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metrics = bundle.get("metrics", {})
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Decision threshold", f"{threshold:.2%}")
    metric_2.metric("Validation ROC-AUC", f"{metrics.get('roc_auc', 0):.2f}")
    metric_3.metric("Validation Brier score", f"{metrics.get('brier_score', 0):.3f}")

    with st.expander("ดูรายละเอียดโมเดลและข้อควรทราบ"):
        detail_col, note_col = st.columns([1, 1.25])
        bands = bundle.get("risk_thresholds", {})
        with detail_col:
            st.caption("Model governance")
            st.write(
                {
                    "bundle_version": bundle.get("bundle_version"),
                    "feature_scope": bundle.get("feature_scope"),
                    "selected_features": len(bundle.get("selected_feature_names", [])),
                    "cost_ratio": bundle.get("cost_ratio"),
                    "trained_at": bundle.get("created_at"),
                }
            )
        with note_col:
            st.caption("เกณฑ์ตัดสินใจ")
            ratio = bundle.get("cost_ratio")
            if ratio:
                st.write(
                    f"เกณฑ์ตรวจสอบ {threshold:.2%} คำนวณจากสมมติฐานว่าการอนุมัติผู้ที่จะผิดนัด "
                    f"เสียหายมากกว่าการปฏิเสธผู้ที่จะชำระคืนได้ {ratio:g} เท่า "
                    f"ความเสี่ยงต่ำหมายถึงต่ำกว่าอัตราผิดนัดเฉลี่ยของพอร์ต {bands.get('low', 0):.2%}"
                )
            st.caption("ข้อควรทราบ")
            st.write(
                "โมเดลเรียนรู้เฉพาะจากช่องที่แบบฟอร์มนี้เก็บได้จริง และไม่ใช้เพศหรือสถานภาพสมรส "
                "ช่องที่ไม่ได้กรอกหรือติ๊กไม่พบในระบบจะถูกส่งเป็นค่าว่าง ไม่ใช่ค่ากลางของข้อมูลฝึก "
                "ระบบ production ยังควรบันทึก model และ data version ของทุกคำทำนาย"
            )


def collect_sidebar_inputs(bundle: dict) -> tuple[bool, dict]:
    render_brand()
    st.sidebar.markdown(
        '<div class="sidebar-note">กรอกข้อมูลให้ใกล้เคียงความเป็นจริงมากที่สุด '
        'เพื่อให้แบบจำลองสร้างภาพรวมความเสี่ยงที่มีประโยชน์ต่อการพิจารณา</div>',
        unsafe_allow_html=True,
    )

    # Deliberately not an st.form. Inside a form no widget reruns the script
    # until submit, so ticking "ไม่พบในระบบ" left its score box editable until
    # the officer had already pressed the button once. main() instead detects
    # when the inputs drift from the last assessment and says so.
    with st.sidebar:
        # เพศและสถานภาพสมรสถูกตัดออกจากโมเดลโดยเจตนา (PROTECTED_ATTRIBUTES ใน
        # src/model_training.py) จำนวนสมาชิกครัวเรือนจึงถามตรงๆ จากจำนวนผู้ใหญ่
        # แทนการอนุมานจากสถานภาพสมรสแบบเดิม
        with st.expander("01  ผู้สมัครและครัวเรือน", expanded=True):
            age = st.slider("อายุ (ปี)", 20, 70, 35)
            education = st.selectbox(
                "ระดับการศึกษา",
                category_options(bundle, "NAME_EDUCATION_TYPE", "education"),
                format_func=format_choice("education"),
            )
            housing_type = st.selectbox(
                "ประเภทที่อยู่อาศัย",
                category_options(bundle, "NAME_HOUSING_TYPE", "housing"),
                format_func=format_choice("housing"),
            )
            adult_column, child_column = st.columns(2)
            adults = adult_column.number_input(
                "ผู้ใหญ่ในครัวเรือน", min_value=1, max_value=10, value=2,
                help="นับรวมผู้สมัครด้วย",
            )
            children = child_column.number_input("จำนวนบุตร", min_value=0, max_value=15, value=0)

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

        submitted = st.button("ประเมินความเสี่ยง", type="primary", use_container_width=True)

    values = {
        "DAYS_BIRTH": -age * 365.25,
        "NAME_EDUCATION_TYPE": category_value(education),
        "NAME_HOUSING_TYPE": category_value(housing_type),
        "CNT_CHILDREN": children,
        "CNT_FAM_MEMBERS": adults + children,
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

    fingerprint = input_fingerprint(values)

    if submitted:
        try:
            probability = predict_default_probability(values, bundle)
            st.session_state["assessment"] = {
                "probability": probability,
                "score": probability_to_score(probability, bundle),
                "level": risk_level(probability, bundle),
                "threshold": float(bundle["decision_threshold"]),
                "fingerprint": fingerprint,
            }
        except Exception as exc:
            st.error(f"ประมวลผลไม่สำเร็จ: {exc}")
            st.stop()

    render_hero(bundle)
    assessment = st.session_state.get("assessment")
    if assessment:
        if assessment.get("fingerprint") != fingerprint:
            # Without a form, edits rerun the page immediately while the result
            # on screen still belongs to the previous inputs.
            st.warning(
                "ข้อมูลในแบบฟอร์มถูกแก้ไขหลังการประเมินครั้งล่าสุด "
                "ผลด้านล่างเป็นของข้อมูลชุดก่อน กดประเมินความเสี่ยงอีกครั้งเพื่ออัปเดต"
            )
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
