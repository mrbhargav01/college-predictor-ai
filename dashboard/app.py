"""
dashboard/app.py  ── EduPath v3.0  ── Ultra-Advanced College Prediction System
Run from project root:
                                                                                             
  Local URL: http://localhost:8501
  Network URL: http://10.70.248.119:8501
    streamlit run dashboard/app.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ── Project imports ────────────────────────────────────────────────────────────
from src.preprocess import load_college_data, get_unique_values, get_cities_for_state, aggregate_stats
from src.model import score_colleges, apply_budget_filter, train_rf_model, load_rf_model
from src.charts import (
    chart_classification_donut, chart_state_distribution,
    chart_exam_cutoff, chart_category_probability,
    chart_probability_histogram, chart_marks_vs_prob,
    chart_course_stacked, chart_correlation_heatmap,
    chart_state_city_sunburst, chart_probability_gauge,
    chart_cutoff_trend, chart_placement_by_course,
)
from src.report_gen import generate_html_report, get_html_download_link, get_csv_download
from dashboard.components.ui_components import (
    render_hero, render_kpi_row, section_header,
    render_college_card, render_stepper,
    render_chat_messages, alert, render_detail_hero,
    render_stat_grid, render_report_banner, render_theme_toggle,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EduPath – AI College Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:scamrebhai@gmail.com",
        "About": "EduPath v3.0 AI College Prediction System",
    },
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(CSS_PATH):
    try:
        with open(CSS_PATH, 'r', encoding='utf-8') as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load custom styles: {e}")

# ── Extra inline CSS for fine details and dark mode ───────────────────────────
st.markdown("""
<style>
/* Sidebar active button override */
[data-testid="stSidebarContent"] .element-container:has(button) button {
  width: 100% !important;
}
/* Hide Streamlit branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
/* Full-width hero */
div[data-testid="stVerticalBlock"]:has(div.hero-wrap) {
  padding-top: 0 !important;
}

/* ════════════════════════════════════════════════════ */
/* DARK MODE OVERRIDES - Ensures text visibility */
/* ════════════════════════════════════════════════════ */

[data-theme="dark"] .main,
[data-theme="dark"] .stApp {
  background-color: #0b0720 !important;
}

[data-theme="dark"] .main .block-container {
  color: #c4bfdf !important;
}

[data-theme="dark"] h1,
[data-theme="dark"] h2,
[data-theme="dark"] h3,
[data-theme="dark"] h4,
[data-theme="dark"] h5,
[data-theme="dark"] h6,
[data-theme="dark"] p,
[data-theme="dark"] span,
[data-theme="dark"] div,
[data-theme="dark"] label {
  color: #c4bfdf !important;
}

[data-theme="dark"] .stMarkdown p,
[data-theme="dark"] .stMarkdown span,
[data-theme="dark"] [data-testid="stMarkdownContainer"] p {
  color: #c4bfdf !important;
}

[data-theme="dark"] table {
  color: #c4bfdf !important;
}

[data-theme="dark"] table td,
[data-theme="dark"] table th {
  color: #c4bfdf !important;
  border-color: rgba(108,99,255,0.15) !important;
}

[data-theme="dark"] .stTextInput input,
[data-theme="dark"] .stNumberInput input,
[data-theme="dark"] .stSelectbox div[data-baseweb="select"] {
  color: #f0eeff !important;
  background-color: rgba(255,255,255,0.05) !important;
}

[data-theme="dark"] .stSelectbox span {
  color: #f0eeff !important;
}

[data-theme="dark"] .stDataFrame {
  color: #c4bfdf !important;
}

[data-theme="dark"] .stDataFrame th {
  color: #f0eeff !important;
}

[data-theme="dark"] .stDataFrame td {
  color: #c4bfdf !important;
}

[data-theme="dark"] .streamlit-expanderHeader {
  color: #f0eeff !important;
}

[data-theme="dark"] .streamlit-expanderContent {
  color: #c4bfdf !important;
}

[data-theme="dark"] .stRadio label,
[data-theme="dark"] .stCheckbox label {
  color: #c4bfdf !important;
}

[data-theme="dark"] [data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0618 0%, #1a0e2e 50%, #0b1a24 100%) !important;
}

[data-theme="dark"] [data-testid="stSidebar"] * {
  color: rgba(255,255,255,0.85) !important;
}

[data-theme="dark"] [data-testid="metric-container"] {
  background: rgba(255,255,255,0.05) !important;
}

[data-theme="dark"] [data-testid="metric-container"] label {
  color: #7b749e !important;
}

[data-theme="dark"] .alert-info {
  color: #c4bfdf !important;
}

[data-theme="dark"] .alert-success {
  color: #c4bfdf !important;
}

[data-theme="dark"] .alert-warn {
  color: #c4bfdf !important;
}

[data-theme="dark"] .glass-card {
  background: rgba(255,255,255,0.05) !important;
}

[data-theme="dark"] .glass-card h3,
[data-theme="dark"] .glass-card h4,
[data-theme="dark"] .glass-card p,
[data-theme="dark"] .glass-card li,
[data-theme="dark"] .glass-card span,
[data-theme="dark"] .glass-card div {
  color: #c4bfdf !important;
}

[data-theme="dark"] .glass-card table td {
  color: #c4bfdf !important;
}

[data-theme="dark"] .college-card {
  background: rgba(255,255,255,0.05) !important;
}

[data-theme="dark"] .college-name {
  color: #f0eeff !important;
}

[data-theme="dark"] .sec-hdr {
  color: #f0eeff !important;
}

[data-theme="dark"] .kpi-card {
  background: rgba(255,255,255,0.05) !important;
}

[data-theme="dark"] .detail-stat {
  background: rgba(255,255,255,0.04) !important;
}

[data-theme="dark"] .detail-stat p {
  color: #7b749e !important;
}

[data-theme="dark"] .stTabs [data-baseweb="tab"] {
  color: #7b749e !important;
}

[data-theme="dark"] .stTabs [aria-selected="true"] {
  color: white !important;
}

[data-theme="dark"] .msg-bot {
  color: #c4bfdf !important;
}

[data-theme="dark"] .hero-title,
[data-theme="dark"] .hero-sub {
  color: white !important;
}

[data-theme="dark"] .report-cta h3 {
  color: #f0eeff !important;
}

[data-theme="dark"] .report-cta p {
  color: #7b749e !important;
}

[data-theme="dark"] .auth-card {
  background: rgba(255,255,255,0.05) !important;
}

[data-theme="dark"] .auth-logo p {
  color: #7b749e !important;
}

[data-theme="dark"] .stMultiSelect span {
  color: #f0eeff !important;
}

[data-theme="dark"] .stSlider label {
  color: #c4bfdf !important;
}

[data-theme="dark"] .prob-header {
  color: #c4bfdf !important;
}

[data-theme="dark"] .college-meta {
  color: #7b749e !important;
}

[data-theme="dark"] .step-label {
  color: #7b749e !important;
}

[data-theme="dark"] .kpi-label {
  color: #7b749e !important;
}

[data-theme="dark"] .chip {
  background: rgba(108,99,255,0.15) !important;
}

[data-theme="dark"] .badge-neutral {
  background: rgba(107,104,144,0.15) !important;
}

[data-theme="dark"] strong,
[data-theme="dark"] b {
  color: #f0eeff !important;
}

[data-theme="dark"] em,
[data-theme="dark"] i {
  color: #9b95c4 !important;
}

[data-theme="dark"] a {
  color: #6C63FF !important;
}

[data-theme="dark"] code {
  color: #4ECDC4 !important;
  background: rgba(255,255,255,0.08) !important;
}

[data-theme="dark"] ul li,
[data-theme="dark"] ol li {
  color: #c4bfdf !important;
}

[data-theme="dark"] .stAlert {
  color: #c4bfdf !important;
}

[data-theme="dark"] .stSuccess {
  color: #065f46 !important;
}

[data-theme="dark"] .stProgress > div > div > div {
  color: #f0eeff !important;
}

[data-theme="dark"] [data-testid="stAppViewContainer"] {
  background-color: #0b0720 !important;
}

[data-theme="dark"] .stApp {
  background-color: #0b0720 !important;
}

[data-theme="dark"] .js-plotly-plot .plotly .main-svg {
  background: transparent !important;
}

/* Login page stat cards */
.login-stat-card {
  text-align: center;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 14px;
  transition: all 0.3s ease;
}

.login-stat-card:hover {
  background: rgba(255, 255, 255, 0.25);
  transform: translateY(-3px);
}

.login-stat-icon {
  font-size: 1.5rem;
  margin-bottom: 0.3rem;
}

.login-stat-value {
  font-size: 1.4rem;
  font-weight: 800;
  color: #1a1040 !important;
  font-family: 'Syne', sans-serif;
  text-shadow: 0 1px 2px rgba(255, 255, 255, 0.3);
}

.login-stat-label {
  font-size: 0.78rem;
  color: #4a4570 !important;
  font-weight: 500;
  margin-top: 0.2rem;
}

[data-theme="dark"] .login-stat-value {
  color: #ffffff !important;
  text-shadow: 0 2px 8px rgba(108, 99, 255, 0.4);
}

[data-theme="dark"] .login-stat-label {
  color: rgba(255, 255, 255, 0.65) !important;
}

[data-theme="dark"] .login-stat-card {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

[data-theme="dark"] .login-stat-card:hover {
  background: rgba(108, 99, 255, 0.15);
}

/* Copyright footer - always visible */
.copyright-footer {
  text-align: center;
  padding: 1.5rem 1rem;
  color: #333333;
  font-size: 0.78rem;
  border-top: 1px solid rgba(0,0,0,0.15);
  margin-top: 2rem;
}

[data-theme="dark"] .copyright-footer {
  color: rgba(255,255,255,0.5);
  border-top: 1px solid rgba(255,255,255,0.1);
}

[data-theme="dark"] .copyright-footer .footer-title {
  color: rgba(255,255,255,0.7);
}

[data-theme="dark"] .copyright-footer .footer-name {
  color: rgba(255,255,255,0.6);
}

[data-theme="dark"] .copyright-footer .footer-subtitle {
  color: rgba(255,255,255,0.4);
}

.footer-title {
  font-weight: 600;
  margin-bottom: .2rem;
  color: #000000;
}

.footer-name {
  color: #000000;
}

.footer-subtitle {
  margin-top: .15rem;
  font-size: .7rem;
  color: #555555;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DARK MODE INITIALIZATION SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<script>
(function() {
    function getCookie(name) {
        var value = "; " + document.cookie;
        var parts = value.split("; " + name + "=");
        if (parts.length == 2) return parts.pop().split(";").shift();
    }
    
    function setCookie(name, value, days) {
        var expires = "";
        if (days) {
            var date = new Date();
            date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
            expires = "; expires=" + date.toUTCString();
        }
        document.cookie = name + "=" + (value || "") + expires + "; path=/";
    }
    
    function applyTheme(theme) {
        var appElement = window.parent.document.querySelector('.stApp') || 
                         window.parent.document.querySelector('[data-testid="stAppViewContainer"]') ||
                         document.querySelector('.stApp');
        
        if (appElement) {
            appElement.setAttribute('data-theme', theme);
        }
        
        if (window.parent.document.body) {
            window.parent.document.body.setAttribute('data-theme', theme);
        }
        document.body.setAttribute('data-theme', theme);
        
        if (window.parent.document.documentElement) {
            window.parent.document.documentElement.setAttribute('data-theme', theme);
        }
        document.documentElement.setAttribute('data-theme', theme);
    }
    
    var savedTheme = getCookie('edupath_theme');
    if (savedTheme) {
        applyTheme(savedTheme);
    } else {
        applyTheme('light');
    }
    
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'theme_change') {
            var theme = event.data.theme;
            applyTheme(theme);
            setCookie('edupath_theme', theme, 365);
        }
    });
    
    window.setEduPathTheme = function(theme) {
        applyTheme(theme);
        setCookie('edupath_theme', theme, 365);
    };
})();
</script>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "logged_in": False,
        "page": "Login",
        "auth_tab": "login",
        "user_profile": {},
        "prediction_results": None,
        "selected_college": None,
        "chat_history": [],
        "theme": "light",
        "form_step": 0,
        "form_data": {},
        "rf_model": None,
        "rf_encoders": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_data():
    return load_college_data()

df = get_data()
uniq = get_unique_values(df)
stats = aggregate_stats(df)


# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def goto(page: str):
    st.session_state.page = page
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  LOGIN / REGISTER  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def page_login():
    # Full-width hero bg
    st.markdown("""
    <div style="position:fixed;inset:0;z-index:-1;
      background:linear-gradient(135deg,#0b0720 0%,#1a1040 40%,#0f3460 100%);">
    </div>
    <div style="position:fixed;top:0;left:0;right:0;bottom:0;z-index:-1;overflow:hidden;">
      <div style="position:absolute;width:600px;height:600px;
        background:radial-gradient(circle,rgba(108,99,255,.25) 0%,transparent 70%);
        top:-100px;left:-100px;animation:meshMove 15s ease-in-out infinite alternate;"></div>
      <div style="position:absolute;width:400px;height:400px;
        background:radial-gradient(circle,rgba(157,80,187,.2) 0%,transparent 70%);
        bottom:-50px;right:-50px;animation:meshMove 20s ease-in-out infinite alternate-reverse;"></div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0 2rem;">
          <div style="font-size:4rem;">🎓</div>
          <h1 style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:900;
            background:linear-gradient(135deg,#6C63FF,#4ECDC4);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            margin:.3rem 0 .5rem;">EduPath</h1>
          <p style="color:rgba(255,255,255,.6);font-size:1.05rem;margin-bottom:0;">
            AI-Powered College Admission Prediction
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="auth-card">
          <div class="auth-logo">
            <p style="color:var(--text-muted);font-size:1rem;">
              Your journey to the right college starts here
            </p>
          </div>
        """, unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            email = st.text_input("📧 Email Address", placeholder="you@email.com", key="li_email")
            pwd   = st.text_input("🔒 Password", type="password",
                                   placeholder="Enter your password", key="li_pwd")

            col1, col2 = st.columns(2)
            with col1:
                remember = st.checkbox("Remember me", value=True)
            with col2:
                st.markdown('<div style="text-align:right;font-size:.85rem;color:#6C63FF;cursor:pointer;">Forgot password?</div>',
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In →", use_container_width=True, key="btn_login"):
                if email and pwd:
                    with st.spinner("Authenticating…"):
                        time.sleep(0.6)
                    st.session_state.logged_in = True
                    st.session_state.page = "Home"
                    name = email.split("@")[0].title()
                    st.session_state.user_profile = {"name": name, "email": email}
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Please enter your email and password.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align:center;color:var(--text-muted);font-size:.85rem;">
              — or sign in with —
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            g1, g2 = st.columns(2)
            with g1:
                if st.button("🔵  Google", use_container_width=True, key="google_btn"):
                    st.session_state.logged_in = True
                    st.session_state.page = "Home"
                    st.session_state.user_profile = {"name": "Google User", "email": "user@gmail.com"}
                    st.rerun()
            with g2:
                if st.button("⚫  GitHub", use_container_width=True, key="github_btn"):
                    st.session_state.logged_in = True
                    st.session_state.page = "Home"
                    st.session_state.user_profile = {"name": "GitHub User", "email": "user@github.com"}
                    st.rerun()

        with tab_reg:
            st.markdown("<br>", unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                reg_first = st.text_input("First Name", placeholder="Rahul", key="reg_fn")
            with r2:
                reg_last = st.text_input("Last Name", placeholder="Sharma", key="reg_ln")

            reg_email = st.text_input("📧 Email", placeholder="rahul@email.com", key="reg_em")
            reg_phone = st.text_input("📱 Phone (optional)", placeholder="+91 XXXXX XXXXX", key="reg_ph")
            reg_pwd   = st.text_input("🔒 Password", type="password",
                                       placeholder="Min 8 characters", key="reg_pw")
            reg_cpwd  = st.text_input("🔒 Confirm Password", type="password",
                                       placeholder="Re-enter password", key="reg_cpw")
            agree     = st.checkbox("I agree to the Terms of Service & Privacy Policy")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", use_container_width=True, key="btn_register"):
                if not (reg_first and reg_email and reg_pwd):
                    st.error("Please fill all required fields.")
                elif reg_pwd != reg_cpwd:
                    st.error("Passwords do not match.")
                elif len(reg_pwd) < 8:
                    st.error("Password must be at least 8 characters.")
                elif not agree:
                    st.error("Please accept the terms to continue.")
                else:
                    with st.spinner("Creating your account…"):
                        time.sleep(0.8)
                    full_name = f"{reg_first} {reg_last}".strip()
                    st.session_state.logged_in = True
                    st.session_state.page = "Home"
                    st.session_state.user_profile = {"name": full_name, "email": reg_email}
                    st.success(f"Welcome, {full_name}! 🎉")
                    st.balloons()
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)  # close auth-card

    # Stats row below card (OUTSIDE col_m block)
    st.markdown("<br>", unsafe_allow_html=True)
    mc = st.columns(4)
    for col, (icon, val, lbl) in zip(mc, [
        ("🏛️", f"{stats['total_colleges']:,}+", "Colleges"),
        ("🗺️", f"{stats['total_states']}", "States"),
        ("🎓", "10+", "Courses"),
        ("👤", "50K+", "Students Helped"),
    ]):
        with col:
            st.markdown(f"""
            <div class="login-stat-card">
              <div class="login-stat-icon">{icon}</div>
              <div class="login-stat-value">{val}</div>
              <div class="login-stat-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align:center;padding:1.8rem 1rem 1.2rem;">
          <div style="font-size:3rem;margin-bottom:.3rem;">🎓</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:900;
            background:linear-gradient(135deg,#6C63FF,#4ECDC4);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            EduPath
          </div>
          <div style="font-size:.75rem;color:rgba(255,255,255,.5);letter-spacing:1px;">
            AI COLLEGE PREDICTION
          </div>
        </div>
        """, unsafe_allow_html=True)

        # User pill
        name = st.session_state.user_profile.get("name", "Student")
        email = st.session_state.user_profile.get("email", "")
        st.markdown(f"""
        <div style="background:rgba(108,99,255,.25);border:1px solid rgba(108,99,255,.4);
          border-radius:12px;padding:.9rem 1rem;margin-bottom:1.2rem;">
          <div style="font-weight:700;font-size:.95rem;">{name}</div>
          <div style="font-size:.75rem;opacity:.6;margin-top:.15rem;">{email}</div>
        </div>""", unsafe_allow_html=True)

        current = st.session_state.page
        nav_items = [
            ("🏠", "Home",       "Home"),
            ("🎯", "Predict",    "Predict"),
            ("📊", "Results",    "Results"),
            ("🏛️", "College Info","CollegeDetail"),
            ("📈", "Analytics",  "Analytics"),
            ("🤖", "AI Chatbot", "Chatbot"),
            ("⚙️", "Settings",   "Settings"),
            ("ℹ️", "About",      "About"),
        ]

        st.markdown('<div style="margin-bottom:.5rem;font-size:.7rem;letter-spacing:1.5px;opacity:.45;text-transform:uppercase;padding-left:.5rem;">Navigation</div>', unsafe_allow_html=True)

        for icon, label, page_key in nav_items:
            is_active = current == page_key
            dot = "●" if is_active else ""
            if st.button(f"{icon}  {label}  {dot}", key=f"nav_{page_key}",
                         use_container_width=True):
                goto(page_key)

        st.markdown("---")

        # Quick stats
        st.markdown('<div style="font-size:.7rem;letter-spacing:1.5px;opacity:.45;text-transform:uppercase;padding-left:.5rem;margin-bottom:.6rem;">Dataset</div>', unsafe_allow_html=True)
        for lbl, val in [
            ("🏛️ Colleges", f"{stats['total_colleges']:,}"),
            ("🗺️ States",   str(stats['total_states'])),
            ("🏙️ Cities",   str(stats['total_cities'])),
            ("📝 Records",  f"{stats['total_records']:,}"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
              padding:.35rem .5rem;font-size:.84rem;">
              <span style="opacity:.7;">{lbl}</span>
              <span style="font-weight:700;">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Theme toggle
        theme = st.session_state.get("theme", "light")
        toggle_label = "🌙 Dark Mode" if theme == "light" else "☀️ Light Mode"
        
        if st.button(toggle_label, use_container_width=True, key="theme_toggle_btn"):
            new_theme = "dark" if theme == "light" else "light"
            st.session_state.theme = new_theme
            st.markdown(f"""
            <script>
                window.setEduPathTheme('{new_theme}');
            </script>
            """, unsafe_allow_html=True)
            st.rerun()

        # Apply current theme
        current_theme = st.session_state.get("theme", "light")
        st.markdown(f"""
        <script>
            window.setEduPathTheme('{current_theme}');
        </script>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Logout
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init_state()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  HOME  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    name = st.session_state.user_profile.get("name", "Student")
    render_hero(
        f"Welcome back, {name}! 👋",
        "Your AI-powered college admission dashboard is ready. Start predicting, explore analytics, and get personalized guidance.",
        chips=["🤖 AI Prediction", "📊 Real-time Analytics", "🤖 Chatbot Counselor",
               "📄 Downloadable Reports",],
    )

    render_kpi_row([
        {"icon": "🏛️", "label": "Total Colleges",    "value": f"{stats['total_colleges']:,}"},
        {"icon": "✅", "label": "Safe Colleges",     "value": f"{stats['safe_count']:,}"},
        {"icon": "⚠️", "label": "Moderate Colleges", "value": f"{stats['moderate_count']:,}"},
        {"icon": "🚀", "label": "Dream Colleges",    "value": f"{stats['dream_count']:,}"},
        {"icon": "🗺️", "label": "States Covered",   "value": str(stats['total_states'])},
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="glass-card fade-in">
          <h3 style="font-family:'Syne',sans-serif;margin-bottom:1rem;">🚀 Quick Start</h3>
          <p style="color:var(--text-muted);margin-bottom:1.5rem;line-height:1.7;">
            Fill in your academic profile, entrance exam scores, and course preferences
            to get personalised college predictions with admission probabilities.
          </p>
        </div>""", unsafe_allow_html=True)
        if st.button("🎯 Start Prediction →", use_container_width=True, key="home_predict"):
            goto("Predict")

    with c2:
        st.markdown("""
        <div class="glass-card fade-in-2">
          <h3 style="font-family:'Syne',sans-serif;margin-bottom:1rem;">🤖 AI Counselor</h3>
          <p style="color:var(--text-muted);margin-bottom:1.5rem;line-height:1.7;">
            Ask our AI counselor anything — best colleges, admission cutoffs, 
            scholarship information, and personalised guidance.
          </p>
        </div>""", unsafe_allow_html=True)
        if st.button("💬 Open Chatbot →", use_container_width=True, key="home_chat"):
            goto("Chatbot")

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("📊", "Quick Analytics Snapshot")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.plotly_chart(chart_classification_donut(df), use_container_width=True)
    with cc2:
        st.plotly_chart(chart_exam_cutoff(df), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  PREDICT  ████████
# ═══════════════════════════════════════════════════════════════════════════════

STEPS = ["Personal", "Academic", "Exam", "Preferences", "Predict"]

def page_predict():
    render_hero(
        "🎯 College Prediction",
        "Complete your profile to receive AI-powered college recommendations.",
        chips=["Step-by-step form", "Smart filtering", "Instant results"],
    )

    step = st.session_state.form_step
    render_stepper(STEPS, step)

    fd = st.session_state.form_data

    # ── Step 0: Personal ────────────────────────────────────────────────────────
    if step == 0:
        section_header("👤", "Personal Details")
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            fd["student_name"] = st.text_input("Full Name *",
                value=fd.get("student_name", st.session_state.user_profile.get("name", "")),
                placeholder="Rahul Sharma")
        with c2:
            fd["gender"] = st.selectbox("Gender *",
                ["Male", "Female", "Prefer not to say"],
                index=["Male","Female","Prefer not to say"].index(fd.get("gender","Male")))
        with c3:
            fd["dob"] = st.text_input("Date of Birth",
                value=fd.get("dob", ""), placeholder="DD/MM/YYYY")

        c4, c5 = st.columns(2)
        with c4:
            states_all = ["Select State"] + uniq["states"]
            idx_s = states_all.index(fd["state"]) if fd.get("state") in states_all else 0
            fd["state"] = st.selectbox("Home State *", states_all, index=idx_s)
        with c5:
            cats = ["General", "OBC", "SC", "ST", "EWS"]
            fd["category"] = st.selectbox("Category *", cats,
                index=cats.index(fd.get("category","General")))
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 1: Academic ────────────────────────────────────────────────────────
    elif step == 1:
        section_header("📚", "Academic Details")
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            years = list(range(2026, 1999, -1))
            fd["passout_year"] = st.selectbox("Pass Out Year *", years,
                index=years.index(fd.get("passout_year", 2024)) if fd.get("passout_year") in years else 2)
        with c2:
            fd["board"] = st.selectbox("Board", ["CBSE","ICSE","State Board","IB","Other"], index=0)

        c3, c4, c5 = st.columns(3)
        with c3:
            fd["tenth_pct"] = st.number_input("10th Percentage *", 0.0, 100.0,
                float(fd.get("tenth_pct", 75.0)), 0.1, format="%.1f", help="Enter marks out of 100")
        with c4:
            fd["twelfth_pct"] = st.number_input("12th Percentage *", 0.0, 100.0,
                float(fd.get("twelfth_pct", 75.0)), 0.1, format="%.1f")
        with c5:
            fd["pcm_pcb"] = st.number_input("PCM / PCB Marks (out of 300)", 0.0, 300.0,
                float(fd.get("pcm_pcb", 240.0)), 1.0)

        fd["overall_percentage"] = st.number_input("Overall Percentage *", 0.0, 100.0,
            float(fd.get("overall_percentage", 75.0)), 0.1, format="%.1f")

        if fd["overall_percentage"] >= 85:
            alert("success", "🌟 Excellent academic profile! You qualify for top colleges.")
        elif fd["overall_percentage"] >= 70:
            alert("info", "Good profile! Many colleges across categories await you.")
        else:
            alert("warn", "Consider appearing for improvement exams for better options.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 2: Exam ────────────────────────────────────────────────────────────
    elif step == 2:
        section_header("📝", "Entrance Exam Details")
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)

        exams_ui = ["All (Any Exam)", "JEE Main", "JEE Advanced", "NEET",
                     "MHT-CET", "KCET", "COMEDK", "WBJEE", "CUET", "Other"]
        exam_map = {"JEE Main": "JEE Main", "MHT-CET": "MHT-CET",
                    "KCET": "KCET", "COMEDK": "COMEDK", "NEET": "NEET",
                    "WBJEE": "WBJEE", "All (Any Exam)": "All", "Other": "All"}

        cur_exam = fd.get("exam_type_ui", "All (Any Exam)")
        if cur_exam not in exams_ui:
            cur_exam = "All (Any Exam)"
        fd["exam_type_ui"] = st.selectbox("Exam Type *", exams_ui, index=exams_ui.index(cur_exam))
        fd["exam_type"] = exam_map.get(fd["exam_type_ui"], "All")

        c1, c2, c3 = st.columns(3)
        with c1:
            fd["exam_score"] = st.number_input("Exam Score *", 0, 800, int(fd.get("exam_score", 200)))
        with c2:
            fd["rank"] = st.number_input("Rank (if applicable)", 0, 1_000_000, int(fd.get("rank", 50000)))
        with c3:
            fd["percentile"] = st.number_input("Percentile", 0.0, 100.0, float(fd.get("percentile", 85.0)), 0.1)

        fd["exam_year"] = st.selectbox("Exam Year", [2026,2025,2024,2023])
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 3: Preferences ─────────────────────────────────────────────────────
    elif step == 3:
        section_header("🏛️", "College Preferences")
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)

        courses_ui = ["All"] + uniq["courses"] + [
            "Engineering","Medical","Pharmacy","Management","Law","Design",
            "Agriculture","BCA","MCA","MBA"
        ]
        courses_ui = sorted(list(set(courses_ui)))
        fd["course_preference"] = st.selectbox("Course Preference *", courses_ui,
            index=courses_ui.index(fd.get("course_preference","All")))

        c1, c2 = st.columns(2)
        with c1:
            ps_opts = ["All"] + uniq["states"]
            fd["preferred_state"] = st.selectbox("Preferred State", ps_opts,
                index=ps_opts.index(fd.get("preferred_state","All")) if fd.get("preferred_state") in ps_opts else 0)
        with c2:
            cities = ["All"] + get_cities_for_state(df, fd.get("preferred_state","All"))
            fd["preferred_city"] = st.selectbox("Preferred City", cities,
                index=cities.index(fd.get("preferred_city","All")) if fd.get("preferred_city") in cities else 0)

        c3, c4, c5 = st.columns(3)
        with c3:
            fd["college_type"] = st.selectbox("College Type", ["Any","Government","Private","Deemed"])
        with c4:
            fd["hostel"] = st.selectbox("Hostel Required", ["No","Yes"])
        with c5:
            fd["budget"] = st.selectbox("Budget Range",
                ["Any","< 1 Lakh","1–3 Lakhs","3–5 Lakhs","> 5 Lakhs"])

        fd["nirf_priority"] = st.checkbox("Prioritise NIRF-ranked colleges", value=True)
        fd["placement_priority"] = st.checkbox("Prioritise high placement colleges", value=False)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 4: Predict ─────────────────────────────────────────────────────────
    elif step == 4:
        section_header("🔮", "Ready to Predict!")
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)

        st.markdown("#### 📋 Your Profile Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            | Field | Value |
            |---|---|
            | **Name** | {fd.get('student_name','—')} |
            | **Category** | {fd.get('category','—')} |
            | **10th %** | {fd.get('tenth_pct','—')} |
            | **12th %** | {fd.get('twelfth_pct','—')} |
            | **Overall %** | {fd.get('overall_percentage','—')} |
            """)
        with c2:
            st.markdown(f"""
            | Field | Value |
            |---|---|
            | **Exam** | {fd.get('exam_type_ui','—')} |
            | **Score** | {fd.get('exam_score','—')} |
            | **Course** | {fd.get('course_preference','—')} |
            | **Preferred State** | {fd.get('preferred_state','Any')} |
            | **Budget** | {fd.get('budget','Any')} |
            """)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Generate Predictions", use_container_width=True, key="btn_predict_final"):
            with st.spinner("🤖 AI is analysing your profile against 5,000+ college records…"):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.012)
                    progress.progress(i + 1)

            results = score_colleges(df, fd)
            results = apply_budget_filter(results, fd.get("budget", "Any"))
            st.session_state.prediction_results = results
            st.session_state.form_data = fd

            progress.empty()
            st.success(f"✅ Found {len(results)} matching colleges!")
            st.balloons()
            time.sleep(1)
            goto("Results")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Navigation buttons ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    nav_c1, nav_c2, nav_c3 = st.columns([1, 4, 1])
    with nav_c1:
        if step > 0:
            if st.button("← Back", use_container_width=True, key="step_back"):
                st.session_state.form_step -= 1
                st.session_state.form_data = fd
                st.rerun()
    with nav_c3:
        if step < len(STEPS) - 1:
            if st.button("Next →", use_container_width=True, key="step_next"):
                st.session_state.form_step += 1
                st.session_state.form_data = fd
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  RESULTS  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def page_results():
    results: pd.DataFrame = st.session_state.prediction_results
    if results is None or len(results) == 0:
        alert("warn", "No predictions yet. Please complete the Prediction form first.")
        if st.button("Go to Prediction →"):
            goto("Predict")
        return

    fd = st.session_state.form_data
    name = fd.get("student_name", "Student")

    render_hero(
        f"🏆 {name}'s College Predictions",
        f"Found {len(results)} colleges matching your profile. Explore, filter, and download your report.",
        chips=["AI Scored", "Real Data", "Filterable"],
    )

    safe_c  = int((results["College_Classification"] == "Safe").sum())
    mod_c   = int((results["College_Classification"] == "Moderate").sum())
    drm_c   = int((results["College_Classification"] == "Dream").sum())
    avg_prob = results["Probability_of_Admission"].mean() * 100

    render_kpi_row([
        {"icon": "🔢", "label": "Total Matches",    "value": str(len(results))},
        {"icon": "✅", "label": "Safe Colleges",    "value": str(safe_c)},
        {"icon": "⚠️", "label": "Moderate",        "value": str(mod_c)},
        {"icon": "🚀", "label": "Dream Colleges",   "value": str(drm_c)},
        {"icon": "📊", "label": "Avg Probability",  "value": f"{avg_prob:.0f}%"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    section_header("🔍", "Filter & Sort Results")
    with st.expander("Show Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            f_cls = st.multiselect("Classification",
                ["Safe","Moderate","Dream"], default=["Safe","Moderate","Dream"], key="r_cls")
        with fc2:
            f_state = st.multiselect("State",
                sorted(results["State"].unique().tolist()),
                default=sorted(results["State"].unique().tolist()), key="r_state")
        with fc3:
            f_min_prob = st.slider("Min Probability (%)", 0, 100, 0, 5, key="r_prob")
        with fc4:
            f_sort = st.selectbox("Sort By",
                ["Best Match Score","Probability (High→Low)","Cutoff (Low→High)"], key="r_sort")

    filtered = results[
        results["College_Classification"].isin(f_cls) &
        results["State"].isin(f_state) &
        (results["Probability_of_Admission"] * 100 >= f_min_prob)
    ].copy()

    if f_sort == "Probability (High→Low)":
        filtered = filtered.sort_values("Probability_of_Admission", ascending=False)
    elif f_sort == "Cutoff (Low→High)":
        filtered = filtered.sort_values("Historical_College_Cutoff_Marks")

    st.markdown(f"<div class='alert-info'>Showing <strong>{len(filtered)}</strong> colleges after filters.</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    view_mode = st.radio("View", ["🃏 Cards","📋 Table","🔬 Detailed"],
                          horizontal=True, label_visibility="collapsed")

    if view_mode == "🃏 Cards":
        for rank, (_, row) in enumerate(filtered.head(30).iterrows(), 1):
            clicked = render_college_card(dict(row), rank)
            if clicked:
                st.session_state.selected_college = dict(row)
                goto("CollegeDetail")

    elif view_mode == "📋 Table":
        show_cols = ["College_Name","State","City","Branch",
                     "College_Classification","Admission_Chance_Pct",
                     "Historical_College_Cutoff_Marks","Previous_Year_Admission_Rank",
                     "College_Type","NIRF_Ranking"]
        avail = [c for c in show_cols if c in filtered.columns]
        rename_map = {
            "College_Name":"College","College_Classification":"Type",
            "Admission_Chance_Pct":"Prob (%)","Historical_College_Cutoff_Marks":"Cutoff",
            "Previous_Year_Admission_Rank":"Prev Rank","College_Type":"Gov/Pvt",
            "NIRF_Ranking":"NIRF",
        }
        table_df = filtered[avail].rename(columns=rename_map).reset_index(drop=True)
        table_df.index = table_df.index + 1
        st.dataframe(table_df, use_container_width=True, height=520)

        sel_idx = st.number_input("Enter # to view college details", 1, len(filtered), 1)
        if st.button("View Selected College →", key="tbl_view"):
            st.session_state.selected_college = dict(filtered.iloc[sel_idx - 1])
            goto("CollegeDetail")

    else:
        for rank, (_, row) in enumerate(filtered.head(20).iterrows(), 1):
            with st.expander(
                f"#{rank} — {row['College_Name']} "
                f"({'✅' if row['College_Classification']=='Safe' else '⚠️' if row['College_Classification']=='Moderate' else '🚀'})"
                f" — {row['Probability_of_Admission']*100:.0f}% chance"
            ):
                d1, d2 = st.columns([2, 1])
                with d1:
                    st.markdown(f"""
                    **📍 Location:** {row.get('City','')}, {row.get('State','')}  
                    **🎓 Branch:** {row.get('Branch','')}  
                    **📝 Exam:** {row.get('Entrance_Exam','')}  
                    **✂️ Cutoff:** {float(row.get('Historical_College_Cutoff_Marks',0)):.2f}  
                    **🏆 Prev Rank:** {int(row.get('Previous_Year_Admission_Rank',0)):,}  
                    **🏛️ Type:** {row.get('College_Type','')}  
                    **🏅 NIRF:** #{int(row.get('NIRF_Ranking',0))}  
                    **💰 Fee:** ₹{float(row.get('Annual_Fee_Lakhs',0)):.2f} L/yr  
                    """)
                with d2:
                    st.plotly_chart(
                        chart_probability_gauge(float(row["Probability_of_Admission"])),
                        use_container_width=True
                    )
                if st.button("Full Details →", key=f"det_{rank}"):
                    st.session_state.selected_college = dict(row)
                    goto("CollegeDetail")

    st.markdown("<br>", unsafe_allow_html=True)
    render_report_banner()
    dl1, dl2 = st.columns(2)
    with dl1:
        html_report = generate_html_report(fd, filtered)
        st.markdown(get_html_download_link(html_report), unsafe_allow_html=True)
    with dl2:
        csv_bytes = get_csv_download(filtered)
        st.download_button("📥 Download CSV",
            data=csv_bytes,
            file_name=f"EduPath_Colleges_{name}.csv",
            mime="text/csv",
            use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  COLLEGE DETAIL  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def page_college_detail():
    college = st.session_state.selected_college
    if not college:
        alert("warn", "No college selected. Go to Results and click 'View Details'.")
        if st.button("← Back to Results"):
            goto("Results")
        return

    render_detail_hero(college)

    render_stat_grid([
        {"icon":"📊","label":"Historical Cutoff",
         "value":f"{float(college.get('Historical_College_Cutoff_Marks',0)):.2f}"},
        {"icon":"🏆","label":"Prev Closing Rank",
         "value":f"{int(college.get('Previous_Year_Admission_Rank',0)):,}"},
        {"icon":"💰","label":"Annual Fee",
         "value":f"₹{float(college.get('Annual_Fee_Lakhs',0)):.1f}L"},
        {"icon":"📈","label":"Avg Package",
         "value":f"₹{float(college.get('Avg_Package_LPA',0)):.1f}LPA"},
        {"icon":"🎯","label":"Placement Rate",
         "value":f"{int(college.get('Placement_Rate_Pct',0))}%"},
        {"icon":"🏅","label":"NIRF Rank",
         "value":f"#{int(college.get('NIRF_Ranking',0))}"},
    ])

    tab_ac, tab_fee, tab_place, tab_fac, tab_trend, tab_contact = st.tabs([
        "📚 Academic", "💰 Fees", "📈 Placement", "🏨 Facilities", "📉 Trend", "📞 Contact"
    ])

    with tab_ac:
        st.markdown(f"""
        <div class="glass-card fade-in">
          <h3 style="font-family:'Syne',sans-serif;margin-bottom:1rem;">Academic Information</h3>
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:.5rem;color:var(--text-muted);width:40%;">Branch</td>
                <td style="font-weight:600;">{college.get('Branch','')}</td></tr>
            <tr style="background:rgba(108,99,255,.04);">
                <td style="padding:.5rem;color:var(--text-muted);">Course Type</td>
                <td style="font-weight:600;">{college.get('Course_Preference','')}</td></tr>
            <tr><td style="padding:.5rem;color:var(--text-muted);">Entrance Exam</td>
                <td style="font-weight:600;">{college.get('Entrance_Exam','')}</td></tr>
            <tr style="background:rgba(108,99,255,.04);">
                <td style="padding:.5rem;color:var(--text-muted);">Category</td>
                <td style="font-weight:600;">{college.get('Category','')}</td></tr>
            <tr><td style="padding:.5rem;color:var(--text-muted);">College Type</td>
                <td style="font-weight:600;">{college.get('College_Type','')}</td></tr>
            <tr style="background:rgba(108,99,255,.04);">
                <td style="padding:.5rem;color:var(--text-muted);">Total Seats</td>
                <td style="font-weight:600;">{int(college.get('Total_Seats',0))}</td></tr>
            <tr><td style="padding:.5rem;color:var(--text-muted);">Rating</td>
                <td style="font-weight:600;">⭐ {float(college.get('Rating',0)):.1f} / 5.0</td></tr>
          </table>
          <br/>
          <h4>Admission Process</h4>
          <ol style="margin-top:.5rem;line-height:2;color:var(--text-b);">
            <li>Apply through {college.get('Entrance_Exam','')} official counselling portal</li>
            <li>Complete document verification (10th/12th marksheets, ID proof)</li>
            <li>Participate in seat allotment rounds based on rank + category</li>
            <li>Pay fees and complete enrollment within deadline</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(chart_probability_gauge(
                float(college.get("Probability_of_Admission", 0))),
                use_container_width=True)

    with tab_fee:
        render_stat_grid([
            {"icon":"🎓","label":"Annual Tuition Fee",
             "value":f"₹{float(college.get('Annual_Fee_Lakhs',0)):.2f}L"},
            {"icon":"🏨","label":"Hostel Fee (Annual)",
             "value":f"₹{float(college.get('Hostel_Fee_Lakhs',0)):.2f}L"},
            {"icon":"📚","label":"Total 4-year Cost (Est.)",
             "value":f"₹{float(college.get('Annual_Fee_Lakhs',0))*4:.1f}L"},
            {"icon":"🎁","label":"Scholarship Available","value":"Yes"},
        ])
        st.markdown("""
        <div class="alert-info">
          💡 Scholarships available for merit holders (>85%) and reserved category students.
        </div>""", unsafe_allow_html=True)

    with tab_place:
        render_stat_grid([
            {"icon":"💰","label":"Average Package",
             "value":f"₹{float(college.get('Avg_Package_LPA',0)):.1f} LPA"},
            {"icon":"🏆","label":"Highest Package",
             "value":f"₹{float(college.get('Highest_Package_LPA',0)):.1f} LPA"},
            {"icon":"🎯","label":"Placement Rate",
             "value":f"{int(college.get('Placement_Rate_Pct',0))}%"},
            {"icon":"🏢","label":"Top Recruiters","value":"100+"},
        ])
        st.plotly_chart(chart_placement_by_course(df), use_container_width=True)

    with tab_fac:
        fa1, fa2 = st.columns(2)
        with fa1:
            st.markdown("""
            <div class="glass-card fade-in">
              <h4>🏨 Hostel & Accommodation</h4>
              <ul style="margin:.8rem 0 0 1.2rem;line-height:2.2;color:var(--text-b);">
                <li>Separate boys' & girls' hostels</li>
                <li>24/7 security with CCTV</li>
                <li>High-speed Wi-Fi campus-wide</li>
                <li>Hygienic mess with multiple cuisines</li>
                <li>Laundry & recreation rooms</li>
              </ul>
            </div>""", unsafe_allow_html=True)
        with fa2:
            st.markdown("""
            <div class="glass-card fade-in">
              <h4>💻 Labs & Infrastructure</h4>
              <ul style="margin:.8rem 0 0 1.2rem;line-height:2.2;color:var(--text-b);">
                <li>Smart classrooms with projectors</li>
                <li>State-of-the-art computer labs</li>
                <li>Engineering workshops</li>
                <li>Innovation & startup incubator</li>
              </ul>
            </div>""", unsafe_allow_html=True)

    with tab_trend:
        st.plotly_chart(
            chart_cutoff_trend(df, college.get("College_Name","")),
            use_container_width=True
        )

    with tab_contact:
        city = college.get("City","")
        state = college.get("State","")
        st.markdown(f"""
        <div class="glass-card fade-in">
          <h3 style="font-family:'Syne',sans-serif;margin-bottom:1.2rem;">Contact Details</h3>
          <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:.6rem;color:var(--text-muted);width:30%;">📍 Address</td>
                <td style="font-weight:500;">{college.get('College_Name','')},<br>{city}, {state}, India</td></tr>
            <tr style="background:rgba(108,99,255,.04);">
                <td style="padding:.6rem;color:var(--text-muted);">📞 Admissions</td>
                <td style="font-weight:500;">+91-XXXX-XXXXXX</td></tr>
            <tr><td style="padding:.6rem;color:var(--text-muted);">📧 Email</td>
                <td style="font-weight:500;">admissions@college.edu.in</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("← Back to Results", use_container_width=True):
            goto("Results")
    with b2:
        if st.session_state.prediction_results is not None:
            html_rep = generate_html_report(
                st.session_state.form_data,
                pd.DataFrame([college]),
                top_n=1,
            )
            st.markdown(get_html_download_link(html_rep, "College_Report.html"),
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  ANALYTICS  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def page_analytics():
    render_hero("📈 Analytics Dashboard",
                "Explore trends, distributions, and insights across 5,000+ college records.",
                chips=["Interactive", "12 Chart Types", "Real Data"])

    section_header("🔍", "Global Data Filters")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        ga_exam = st.multiselect("Exam", uniq["exams"], default=uniq["exams"], key="ga_exam")
    with fc2:
        ga_cat  = st.multiselect("Category", uniq["categories"], default=uniq["categories"], key="ga_cat")
    with fc3:
        ga_cls  = st.multiselect("Classification", ["Safe","Moderate","Dream"],
                                  default=["Safe","Moderate","Dream"], key="ga_cls")
    with fc4:
        ga_state = st.multiselect("State", uniq["states"], default=uniq["states"], key="ga_state")

    fdf = df[
        df["Entrance_Exam"].isin(ga_exam) &
        df["Category"].isin(ga_cat) &
        df["College_Classification"].isin(ga_cls) &
        df["State"].isin(ga_state)
    ]

    st.markdown(f"<div class='alert-info'>Analysing <strong>{len(fdf):,}</strong> records</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_classification_donut(fdf), use_container_width=True)
    with c2:
        st.plotly_chart(chart_state_distribution(fdf), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_exam_cutoff(fdf), use_container_width=True)
    with c4:
        st.plotly_chart(chart_category_probability(fdf), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(chart_probability_histogram(fdf), use_container_width=True)
    with c6:
        st.plotly_chart(chart_marks_vs_prob(fdf), use_container_width=True)

    st.plotly_chart(chart_course_stacked(fdf), use_container_width=True)

    c7, c8 = st.columns([2, 1])
    with c7:
        st.plotly_chart(chart_correlation_heatmap(fdf), use_container_width=True)
    with c8:
        st.plotly_chart(chart_placement_by_course(fdf), use_container_width=True)

    st.plotly_chart(chart_state_city_sunburst(fdf), use_container_width=True)

    section_header("📥", "Download Analytics Data")
    csv_bytes = get_csv_download(fdf)
    st.download_button("📥 Download Filtered Dataset (CSV)", data=csv_bytes,
        file_name="EduPath_Analytics.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  CHATBOT  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def bot_response_rule_based(msg: str, user_fd: dict) -> str:
    m = msg.lower()
    name = user_fd.get("student_name", "")
    score = user_fd.get("exam_score", 0)
    pct = user_fd.get("overall_percentage", 0)

    if any(w in m for w in ["best college","which college","top college","recommend"]):
        if score and pct:
            return (f"Based on your profile (Score: {score}, {pct}% overall):\n\n"
                    "✅ Safe Bets: colleges where your score is 10-15% above cutoff.\n"
                    "⚠️ Moderate: cutoff ± 5% of your score.\n"
                    "🚀 Dream: slightly higher cutoffs — sometimes you get lucky!\n\n"
                    "Use the **Prediction** page for a personalised AI-ranked list. 🎯")
        return "Please complete the **Prediction form** first for personalised advice!"

    elif any(w in m for w in ["admission chance","chances","probability"]):
        if score:
            prob_est = min(95, max(20, int(score / 3.5)))
            return (f"With score {score}, estimated probability: ~{prob_est}% for average colleges.\n"
                    "Check **Results** page for exact probabilities!")
        return "Complete the **Prediction** form to see exact probabilities."

    elif any(w in m for w in ["hi","hello","hey","start","help"]):
        return (f"Hello{' ' + name + '!' if name else '!'} 👋 I'm **EduPath AI**.\n\n"
                "I can help with:\n"
                "🎯 College recommendations\n📊 Admission probability\n"
                "💰 Scholarships\n📝 Exam guidance (JEE/NEET/MHT-CET)\n"
                "What would you like to know? 😊")

    elif any(w in m for w in ["thank","thanks"]):
        return "You're welcome! 😊 Best of luck with your college journey! 🎓"

    else:
        return ("Try asking about:\n• Which colleges match my profile?\n"
                "• JEE / NEET / MHT-CET cutoffs\n• Scholarship programs\n"
                "Or use the **Predict** page for full AI analysis! 🎯")


def page_chatbot():
    render_hero("🤖 AI Counseling Chatbot",
                "Get instant, intelligent answers to all your college admission questions.",
                chips=["24/7 Available","Context-aware","Exam-specific","Personalised"])

    c_main, c_side = st.columns([2, 1])

    with c_main:
        section_header("💬", "Chat with EduPath AI")
        render_chat_messages(st.session_state.chat_history)

        st.markdown("<br>", unsafe_allow_html=True)
        c_inp, c_btn = st.columns([5, 1])
        with c_inp:
            user_msg = st.text_input("Type your question…",
                placeholder="e.g. Which colleges can I get with JEE Main 180?",
                label_visibility="collapsed", key="chat_input_main")
        with c_btn:
            send_clicked = st.button("Send →", use_container_width=True, key="chat_send")

        if send_clicked and user_msg.strip():
            ts = datetime.now().strftime("%I:%M %p")
            st.session_state.chat_history.append({"role": "user", "content": user_msg, "ts": ts})
            with st.spinner("🤔 Thinking…"):
                time.sleep(0.4)
                reply = bot_response_rule_based(user_msg, st.session_state.form_data)
            st.session_state.chat_history.append({"role": "bot", "content": reply, "ts": datetime.now().strftime("%I:%M %p")})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    with c_side:
        section_header("⚡", "Quick Questions")
        quick_qs = [
            "Which college is best for me?",
            "What are my admission chances?",
            "Best CSE colleges in Maharashtra",
            "JEE Main cutoff for NITs",
            "Scholarship options available",
        ]
        for q in quick_qs:
            if st.button(f"💡 {q}", key=f"qq_{q[:20]}", use_container_width=True):
                ts = datetime.now().strftime("%I:%M %p")
                st.session_state.chat_history.append({"role":"user","content":q,"ts":ts})
                reply = bot_response_rule_based(q, st.session_state.form_data)
                st.session_state.chat_history.append({"role":"bot","content":reply,"ts":datetime.now().strftime("%I:%M %p")})
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  SETTINGS  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def page_settings():
    render_hero("⚙️ Settings", "Manage your profile, preferences, and app configuration.")

    section_header("👤", "Profile Settings")
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    p = st.session_state.user_profile
    c1, c2 = st.columns(2)
    with c1:
        new_name = st.text_input("Display Name", value=p.get("name",""))
        new_email = st.text_input("Email", value=p.get("email",""))
    with c2:
        new_phone = st.text_input("Phone", value=p.get("phone",""), placeholder="+91 XXXXX XXXXX")
        new_city  = st.text_input("City", value=p.get("city",""))
    if st.button("Save Profile", key="save_profile"):
        st.session_state.user_profile.update({"name":new_name,"email":new_email,
                                               "phone":new_phone,"city":new_city})
        st.success("✅ Profile updated!")
    st.markdown('</div>', unsafe_allow_html=True)

    section_header("🎨", "Appearance")
    st.markdown('<div class="glass-card fade-in-2">', unsafe_allow_html=True)
    
    current_theme = st.session_state.get("theme", "light")
    theme_options = ["☀️ Light Mode", "🌙 Dark Mode"]
    current_index = 1 if current_theme == "dark" else 0
    
    selected_theme = st.radio("Select Theme", theme_options, index=current_index, key="theme_radio_settings")
    
    new_theme = "dark" if "Dark" in selected_theme else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.markdown(f"""<script>window.setEduPathTheme('{new_theme}');</script>""", unsafe_allow_html=True)
        st.rerun()
    
    st.markdown(f"""<script>window.setEduPathTheme('{st.session_state.theme}');</script>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    section_header("🗑️", "Data Management")
    st.markdown('<div class="glass-card fade-in-3">', unsafe_allow_html=True)
    if st.button("🗑️ Clear Prediction Results", key="clear_results"):
        st.session_state.prediction_results = None
        st.success("Results cleared.")
    if st.button("🗑️ Clear Chat History", key="clear_chat_settings"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")
    if st.button("🔄 Reset Form Data", key="reset_form"):
        st.session_state.form_data = {}
        st.session_state.form_step = 0
        st.success("Form reset.")
    st.markdown('</div>', unsafe_allow_html=True)

    section_header("🤖", "ML Model")
    st.markdown('<div class="glass-card fade-in-4">', unsafe_allow_html=True)
    st.markdown("**Random Forest Classifier** — trains on 80% of the 5,000-record dataset. Expected accuracy: ~70%.")
    if st.button("🚀 Train Random Forest Model", key="train_rf"):
        with st.spinner("Training…"):
            result = train_rf_model(df)
        if result:
            st.success(f"✅ Model trained! Accuracy: {result['accuracy']*100:.1f}%")
            st.code(result["report"])
        else:
            st.error("scikit-learn required. Run: pip install scikit-learn")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  ABOUT  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def page_about():
    render_hero("ℹ️ About EduPath",
                "AI-powered college admission prediction for 50,000+ Indian students.",
                chips=["v3.0", "Open Source", "Python + Streamlit"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card fade-in">
          <h3 style="font-family:'Syne',sans-serif;color:var(--text-h);">🎯 Mission</h3>
          <p style="line-height:1.8;color:var(--text-b);margin-top:.8rem;">
            EduPath removes the guesswork from college admissions using historical 
            cutoff data, entrance exam scores, and advanced ML.
          </p>
          <br/>
          <h3 style="font-family:'Syne',sans-serif;color:var(--text-h);">✨ Key Features</h3>
          <ul style="margin:.8rem 0 0 1.2rem;line-height:2.2;color:var(--text-b);">
            <li>AI-powered college scoring & ranking</li>
            <li>Safe / Moderate / Dream classification</li>
            <li>12+ interactive analytics charts</li>
            <li>Downloadable HTML/CSV reports</li>
            <li>Intelligent chatbot</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card fade-in-2">
          <h3 style="font-family:'Syne',sans-serif;color:var(--text-h);">🛠️ Tech Stack</h3>
          <div class="chip-row" style="margin-top:1rem;">
            <span class="chip">Python</span><span class="chip">Streamlit</span>
            <span class="chip">Pandas</span><span class="chip">Plotly</span>
          </div>
          <br/>
          <h3 style="font-family:'Syne',sans-serif;color:var(--text-h);">📊 Dataset</h3>
          <table style="width:100%;border-collapse:collapse;margin-top:.8rem;">
            <tr><td style="padding:.5rem;color:var(--text-muted);">Records</td><td style="font-weight:600;color:var(--text-b);">5,000</td></tr>
            <tr style="background:rgba(108,99,255,.04);"><td style="padding:.5rem;color:var(--text-muted);">Colleges</td><td style="font-weight:600;color:var(--text-b);">2,500+</td></tr>
            <tr><td style="padding:.5rem;color:var(--text-muted);">States</td><td style="font-weight:600;color:var(--text-b);">15</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card fade-in-3" style="margin-top:1.5rem;">
      <h3 style="font-family:'Syne',sans-serif;color:var(--text-h);">⚠️ Disclaimer</h3>
      <p style="color:var(--text-b);line-height:1.8;margin-top:.8rem;">
        Predictions are based on historical data. Always verify with official portals (JoSAA, MCC, CAP).
      </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  GLOBAL FOOTER  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def render_footer():
    """Renders copyright footer at the BOTTOM of all pages."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="copyright-footer">
      <div class="footer-title">🎓 EduPath v3.0</div>
      <div>© 2026 <strong class="footer-name">Bhargav Thombare</strong>. All rights reserved.</div>
      <div class="footer-subtitle">AI-Powered College Admission Prediction System</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ████████  MAIN ROUTER  ████████
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not st.session_state.logged_in:
        page_login()
        return

    render_sidebar()

    page = st.session_state.page

    if page == "Home":
        page_home()
    elif page == "Predict":
        page_predict()
    elif page == "Results":
        page_results()
    elif page == "CollegeDetail":
        page_college_detail()
    elif page == "Analytics":
        page_analytics()
    elif page == "Chatbot":
        page_chatbot()
    elif page == "Settings":
        page_settings()
    elif page == "About":
        page_about()
    else:
        page_home()

    # ── Copyright Footer (BOTTOM of every page after login) ─────────────
    render_footer()


if __name__ == "__main__":
    main()