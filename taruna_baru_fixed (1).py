import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind
from datetime import datetime
import io
import hashlib
import warnings
warnings.filterwarnings('ignore')

# --- Import khusus untuk LSTM (lazy import agar app tidak crash jika TF belum install) ---
_TF_AVAILABLE = False
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    _TF_AVAILABLE = True
except ImportError:
    pass

st.set_page_config(
    page_title="Dashboard Keuangan Karang Taruna",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

:root {
    --navy-dark:   #0F2044;
    --navy:        #1B3A6B;
    --navy-mid:    #2E5BBA;
    --navy-light:  #EEF3FB;
    --orange:      #F39C12;
    --orange-dark: #E67E22;
    --orange-light:#FEF5E7;
    --green:       #27AE60;
    --green-light: #E8F8EF;
    --red:         #E74C3C;
    --red-light:   #FDEDEC;
    --bg:          #F4F7FB;
    --card:        #FFFFFF;
    --border:      #DDE3EF;
    --text-dark:   #1A2D5A;
    --text-mid:    #4A5778;
    --text-light:  #8090B5;
}

/* ======= 1. ATUR FONT POPPINS (TANPA MERUSAK IKON) ======= */
*, html, body, [class*="css"] { 
    font-family: 'Poppins', sans-serif; 
}

/* ======= 2. PROTEKSI FONT IKON BAWAAN STREAMLIT ======= */
[data-testid="stIconMaterial"], 
[data-testid="stHeaderSidebarToggle"] *,
button[aria-label*="sidebar" i] *,
.stSidebarCollapsedControl * {
    font-family: 'Material Symbols Outlined', 'Material Symbols Rounded', 'Material Icons' !important;
    font-size: inherit;
}
.stApp { background: var(--bg); }
            
/* Material icon fallback text */
span.material-icons,
span.material-symbols-rounded,
span[class*="material"] {
    font-size: 0 !important;
    color: transparent !important;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ======= HEADER ======= */
.main-header {
    background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy) 60%, var(--navy-mid) 100%);
    padding: 26px 36px;
    border-radius: 18px;
    margin-bottom: 22px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(15,32,68,0.28);
    display: flex;
    align-items: center;
    gap: 20px;
}
.main-header::before {
    content: '';
    position: absolute; right: 0; top: 0;
    width: 320px; height: 100%;
    background: linear-gradient(135deg, transparent 40%, rgba(243,156,18,0.12) 100%);
    pointer-events: none;
}
.main-header::after {
    content: 'KT';
    position: absolute; right: 36px; top: 50%;
    transform: translateY(-50%);
    font-size: 5.5rem; font-weight: 900;
    opacity: 0.07; line-height: 1;
    color: var(--orange);
    letter-spacing: -4px;
    pointer-events: none;
}
.header-logo {
    width: 56px; height: 56px; border-radius: 16px;
    background: rgba(255,255,255,0.12);
    border: 2px solid rgba(255,255,255,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.65rem; flex-shrink: 0;
}
.header-badge-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--orange); margin-right: 7px; vertical-align: middle;
}
.header-text h1 { font-size: 1.55rem; font-weight: 900; margin: 0 0 5px 0; line-height: 1.2; }
.header-text p  { font-size: 0.84rem; margin: 0; opacity: 0.75; }
.header-badge {
    margin-left: auto; flex-shrink: 0; z-index: 1;
    background: rgba(243,156,18,0.2);
    border: 1.5px solid rgba(243,156,18,0.45);
    border-radius: 30px; padding: 6px 16px;
    font-size: 0.76rem; font-weight: 700; color: #FDEBD0;
    white-space: nowrap;
}

/* ======= WELCOME SCREEN ======= */
.welcome-box {
    background: var(--card); border-radius: 20px; padding: 48px 36px;
    text-align: center; box-shadow: 0 4px 24px rgba(27,58,107,0.09);
    border: 2px dashed var(--border); margin-bottom: 20px;
}
.welcome-icon {
    width: 64px; height: 64px; border-radius: 20px; margin: 0 auto 14px;
    background: linear-gradient(135deg, #1B3A6B, #2E5BBA);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; font-weight: 900; color: white; letter-spacing: 1px;
    box-shadow: 0 4px 18px rgba(46,91,186,0.35);
}
.welcome-title { font-size: 1.45rem; font-weight: 900; color: var(--navy); margin-bottom: 10px; }
.welcome-desc  { font-size: 0.90rem; color: var(--text-mid); line-height: 1.75; max-width: 500px; margin: 0 auto 28px; }
.welcome-steps { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; max-width: 640px; margin: 0 auto; text-align: left; }
.step-item { background: var(--navy-light); border-radius: 14px; padding: 16px; border: 1.5px solid #C9D6EE; }
.step-num  { background: var(--navy); color: white; border-radius: 50%; width: 26px; height: 26px; display: flex; align-items: center; justify-content: center; font-size: 0.76rem; font-weight: 800; margin-bottom: 9px; }
.step-title { font-size: 0.78rem; font-weight: 800; color: var(--navy); margin-bottom: 4px; }
.step-desc  { font-size: 0.72rem; color: var(--text-mid); line-height: 1.55; }

/* ======= SECTION TITLE ======= */
.section-title {
    font-size: 0.92rem; font-weight: 800; color: var(--navy);
    padding: 9px 16px; border-left: 5px solid var(--orange);
    background: var(--orange-light); border-radius: 0 10px 10px 0;
    margin: 18px 0 14px 0; display: block; line-height: 1.4;
}

/* ======= KPI CARDS ======= */
.kpi-card {
    background: var(--card); border-radius: 16px; padding: 20px 16px;
    text-align: center; box-shadow: 0 2px 14px rgba(27,58,107,0.08);
    border-top: 4px solid; height: 100%; box-sizing: border-box;
    overflow: hidden; transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 6px 22px rgba(27,58,107,0.15); }
.kpi-label {
    font-size: 0.64rem; font-weight: 700; color: var(--text-light);
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; word-break: break-word;
}
.kpi-value { font-size: 1.18rem; font-weight: 900; color: var(--text-dark); line-height: 1.2; margin-bottom: 5px; }
.kpi-sub   { font-size: 0.66rem; color: var(--text-light); line-height: 1.4; }
.kpi-delta { font-size: 0.63rem; margin-top: 5px; font-weight: 700; }

/* ======= ANOMALI ======= */
.anomali-box {
    background: #FFF8E1; border: 2px solid var(--orange);
    border-radius: 12px; padding: 14px 16px; margin: 8px 0;
    display: flex; gap: 12px; align-items: flex-start;
}
.anomali-badge { font-size: 0.72rem; font-weight: 800; color: #7D3C00; background: #EDBB99; border-radius: 6px; padding: 3px 8px; flex-shrink: 0; line-height: 1.6; }
.anomali-title { font-size: 0.82rem; font-weight: 800; color: #7D3C00; margin-bottom: 3px; }
.anomali-text  { font-size: 0.73rem; color: #7D4E00; line-height: 1.55; }

/* ======= PREDIKSI BOX ======= */
.pred-box {
    background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy) 60%, var(--navy-mid) 100%);
    border-radius: 18px; padding: 26px 28px; color: white;
    box-shadow: 0 8px 28px rgba(15,32,68,0.28);
    position: relative; overflow: hidden; margin-bottom: 18px;
}
.pred-box-title { font-size: 1.12rem; font-weight: 800; margin-bottom: 5px; }
.pred-box-sub   {
    font-size: 0.80rem; opacity: 0.75; margin-bottom: 20px;
    background: rgba(255,255,255,0.07); border-radius: 8px; padding: 6px 12px; display: inline-block;
}
.pred-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
.pred-item {
    background: rgba(255,255,255,0.10); border: 1.5px solid rgba(255,255,255,0.18);
    border-radius: 13px; padding: 14px 10px; text-align: center;
}
.pred-item-label { font-size: 0.68rem; opacity: 0.80; font-weight: 600; margin-bottom: 8px; }
.pred-item-value { font-size: 1.02rem; font-weight: 900; line-height: 1.2; }
.pred-item-delta { font-size: 0.64rem; opacity: 0.70; margin-top: 4px; }

/* ======= FORECAST CARDS ======= */
.fc-card { background: var(--card); border-radius: 14px; padding: 18px 14px; text-align: center; box-shadow: 0 2px 12px rgba(27,58,107,0.08); border-top: 4px solid var(--navy); flex: 1; }
.fc-month { font-size: 0.70rem; font-weight: 700; color: var(--text-light); text-transform: uppercase; margin-bottom: 8px; }
.fc-value { font-size: 1.05rem; font-weight: 900; color: var(--navy); margin-bottom: 3px; }
.fc-ci    { font-size: 0.65rem; color: var(--text-light); }

/* ======= HEALTH BARS ======= */
.health-row { margin-bottom: 20px; }
.health-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.health-label { font-size: 0.84rem; font-weight: 700; color: var(--text-dark); }
.health-score { font-size: 0.84rem; font-weight: 900; }
.health-bar-bg   { background: var(--border); border-radius: 10px; height: 12px; overflow: hidden; }
.health-bar-fill { height: 100%; border-radius: 10px; transition: width 0.6s ease; }
.health-sub { font-size: 0.70rem; color: var(--text-light); margin-top: 5px; }

/* ======= SCORE CIRCLE ======= */
.score-circle {
    border-radius: 18px; padding: 24px 14px; text-align: center; height: 100%;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    min-height: 190px;
}
.score-num   { font-size: 2.8rem; font-weight: 900; line-height: 1; }
.score-label { font-size: 0.88rem; font-weight: 800; margin-top: 8px; }
.score-sub   { font-size: 0.68rem; color: var(--text-mid); margin-top: 8px; line-height: 1.4; text-align: center; }

/* ======= TIP CARDS ======= */
.tip-card {
    background: var(--card); border-radius: 14px; padding: 16px;
    box-shadow: 0 2px 10px rgba(27,58,107,0.07); border-left: 5px solid;
    display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px;
}
.tip-badge { width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 900; color: white; }
.tip-title { font-size: 0.82rem; font-weight: 800; color: var(--text-dark); margin-bottom: 4px; }
.tip-text  { font-size: 0.73rem; color: var(--text-mid); line-height: 1.60; }

/* ======= A/B CARDS ======= */
.ab-card { border-radius: 14px; padding: 20px 16px; text-align: center; }
.ab-card.awal  { background: var(--green-light); border: 2px solid #A9DFBF; }
.ab-card.akhir { background: var(--navy-light);  border: 2px solid #C9D6EE; }
.ab-period { font-size: 0.78rem; font-weight: 700; color: var(--text-dark); margin-bottom: 8px; }
.ab-value  { font-size: 1.38rem; font-weight: 900; margin-bottom: 5px; }
.ab-card.awal  .ab-value { color: var(--green); }
.ab-card.akhir .ab-value { color: var(--navy); }
.ab-desc { font-size: 0.68rem; color: var(--text-mid); }
.ab-card.warning { background: var(--orange-light); border: 2px solid var(--orange); }
.ab-card.warning .ab-value { color: var(--orange-dark); }

/* ======= DIVIDER ======= */
.custom-divider {
    height: 2px; background: linear-gradient(90deg, var(--orange) 0%, transparent 100%);
    border: none; border-radius: 2px; margin: 18px 0;
}

/* ======= FOOTER ======= */
.footer-box {
    background: linear-gradient(135deg, var(--navy-dark), var(--navy));
    color: white; text-align: center; padding: 20px; border-radius: 16px;
    margin-top: 30px; font-size: 0.80rem; border-top: 3px solid var(--orange);
}
.footer-box span { opacity: 0.60; font-size: 0.70rem; display: block; margin-top: 5px; }

/* ======= DOWNLOAD BUTTON ======= */
.stDownloadButton button {
    background: var(--navy) !important; color: white !important;
    border-radius: 10px !important; font-weight: 700 !important;
    border: none !important; width: 100% !important; transition: background 0.2s !important;
}
.stDownloadButton button:hover { background: var(--navy-dark) !important; }

/* ======= TABS ======= */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: var(--card); border-radius: 14px;
    padding: 6px 8px; box-shadow: 0 2px 12px rgba(27,58,107,0.08);
    margin-bottom: 18px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important; padding: 8px 16px !important;
    font-weight: 700 !important; font-size: 0.82rem !important; color: var(--text-mid) !important;
}
.stTabs [aria-selected="true"] { background: var(--navy) !important; color: white !important; }

/* ======= SIDEBAR ======= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1A3A 0%, #0F2044 60%, #132652 100%) !important;
    border-right: 1px solid rgba(243,156,18,0.18) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0.5rem !important; padding-bottom: 2rem !important; }

[data-testid="stSidebarCollapseButton"] {
    display: flex !important; visibility: visible !important;
    background: rgba(243,156,18,0.15) !important;
    border: 1px solid rgba(243,156,18,0.35) !important;
    border-radius: 8px !important;
    font-size: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"] * {
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    fill: var(--orange) !important;
    width: 18px !important; height: 18px !important;
    flex-shrink: 0 !important;
    font-size: initial !important;
}

/* Kotak navigasi saat sidebar ditutup — tambah ikon panah */
[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important;
    align-items: center !important; justify-content: center !important;
    background: var(--navy) !important;
    border: 1px solid rgba(243,156,18,0.45) !important;
    border-radius: 0 10px 10px 0 !important;
    width: 28px !important; min-width: 28px !important;
    font-size: 0 !important; color: transparent !important;
    overflow: hidden !important;
    box-shadow: 3px 0 12px rgba(0,0,0,0.18) !important;
}
[data-testid="collapsedControl"] * {
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="collapsedControl"] svg {
    fill: var(--orange) !important;
    width: 18px !important; height: 18px !important;
    flex-shrink: 0 !important;
    font-size: initial !important;
}
/* Ikon panah kanan via pseudo-element sebagai fallback */
[data-testid="collapsedControl"]::after {
    content: '›';
    color: var(--orange) !important;
    font-size: 1.5rem !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    display: block !important;
}

/* Label selectbox — Pilih Tahun & Pilih Bulan */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
    letter-spacing: 0.2px !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important; font-weight: 800 !important;
    font-size: 0.92rem !important; margin-bottom: 10px !important; letter-spacing: 0.3px !important;
}

/* SELECT BOX */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #1B3A6B !important;
    border: 1.5px solid rgba(255,255,255,0.60) !important;
    border-radius: 10px !important; transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    background-color: #2E5BBA !important; border-color: var(--orange) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"],
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div[class*="ValueContainer"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] [class*="Input"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: rgba(255,255,255,0.75) !important; }
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #FFFFFF  !important;
    border: 1px solid #DDE3EF !important; border-radius: 10px !important;
}
[data-baseweb="popover"] [role="option"] {
    background-color: transparent !important; color: #1B3A6B  !important; font-size: 0.85rem !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: rgba(243,156,18,0.20) !important; color: #1B3A6B !important;
}

/* FILE UPLOADER */
[data-testid="stSidebar"] [data-testid="stFileUploader"] > label { display: none !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.06) !important;
    border: 2px dashed rgba(243,156,18,0.50) !important;
    border-radius: 12px !important; padding: 20px 14px !important;
    text-align: center !important; transition: all 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(243,156,18,0.08) !important; border-color: var(--orange) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: rgba(255,255,255,0.80) !important; font-size: 0.80rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: var(--orange-dark) !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 8px 20px !important; margin-top: 8px !important;
    font-size: 0.82rem !important; font-weight: 700 !important;
    font-family: 'Poppins', sans-serif !important; cursor: pointer !important;
    width: 100% !important; transition: background 0.2s !important;
    overflow: hidden !important;
}
/* Sembunyikan span duplikat penyebab teks "uploadupload" */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span:last-child {
    display: inline !important;
    color: white !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: rgba(39,174,96,0.12) !important; border-radius: 8px !important;
    padding: 10px 12px !important; border: 1px solid rgba(39,174,96,0.35) !important; margin-top: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
    color: #A9DFBF !important; opacity: 1 !important; font-size: 0.80rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg { fill: #27AE60 !important; }

/* Sidebar Components */
.sidebar-info {
    background: rgba(255,255,255,0.07); border-radius: 12px; padding: 16px;
    font-size: 0.80rem; color: rgba(255,255,255,0.90); line-height: 1.85;
    border: 1px solid rgba(255,255,255,0.12);
}
.sidebar-info strong { color: #FDEBD0; }
.sidebar-badge {
    display: inline-block; border-radius: 20px; padding: 4px 14px;
    font-size: 0.72rem; font-weight: 700; margin-top: 8px; letter-spacing: 0.3px;
}
.sidebar-badge.surplus { background: rgba(39,174,96,0.25); color: #A9DFBF; border: 1px solid rgba(39,174,96,0.4); }
.sidebar-badge.defisit { background: rgba(231,76,60,0.25); color: #F1948A; border: 1px solid rgba(231,76,60,0.4); }
.sidebar-divider { border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 16px 0; }
.sidebar-logo { text-align: center; padding: 20px 0 18px; border-bottom: 1px solid rgba(243,156,18,0.25); margin-bottom: 18px; }
.sidebar-logo .logo-wrap {
    display: inline-flex; align-items: center; justify-content: center;
    width: 60px; height: 60px; border-radius: 18px;
    background: linear-gradient(135deg, #1B3A6B, #2E5BBA);
    font-size: 1.3rem; font-weight: 900; color: white; margin-bottom: 10px;
    box-shadow: 0 4px 18px rgba(46,91,186,0.40); letter-spacing: 1px;
}
.sidebar-logo .logo-title { font-size: 1.05rem; font-weight: 900; color: white; display: block; letter-spacing: 0.3px; }
.sidebar-logo .logo-sub   { font-size: 0.74rem; color: rgba(243,156,18,0.85); font-weight: 500; display: block; margin-top: 3px; }
.sidebar-nav {
    background: rgba(255,255,255,0.06); border-radius: 12px; padding: 14px 16px;
    margin-top: 12px; font-size: 0.78rem; color: rgba(255,255,255,0.80);
    line-height: 2.4; border: 1px solid rgba(255,255,255,0.10);
}
.sidebar-nav strong { color: white; }
.sidebar-footer { margin-top: 22px; text-align: center; font-size: 0.70rem; color: rgba(243,156,18,0.55); padding-bottom: 8px; }

/* YoY box */
.yoy-box { background: var(--card); border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(27,58,107,0.07); border-left: 4px solid var(--navy-mid); margin-bottom: 12px; }
.yoy-title { font-size: 0.82rem; font-weight: 800; color: var(--text-dark); margin-bottom: 8px; }
.yoy-val   { font-size: 1.10rem; font-weight: 900; }
.yoy-delta { font-size: 0.70rem; font-weight: 700; }

/* Metric badge row */
.metric-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--navy-light); border-radius: 8px;
    padding: 5px 12px; font-size: 0.74rem; font-weight: 700;
    color: var(--navy); margin: 4px 4px 0 0; border: 1px solid #C9D6EE;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONSTANTA & HELPERS
# ============================================================

MONTH_ID = {
    1:"Januari", 2:"Februari", 3:"Maret",    4:"April",
    5:"Mei",     6:"Juni",     7:"Juli",      8:"Agustus",
    9:"September",10:"Oktober",11:"November",12:"Desember"
}
MONTH_SHORT = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
CLR_GREEN  = "#27AE60"; CLR_RED  = "#E74C3C"; CLR_BLUE  = "#2E5BBA"
CLR_GOLD   = "#F39C12"; CLR_DARK = "#1B3A6B"; CLR_NAVY  = "#1B3A6B"
LAYOUT = dict(font=dict(family="Poppins, sans-serif", size=11, color="#1A2D5A"),
              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=12, r=12, t=64, b=16))

def rupiah(n, singkat=False):
    try:
        n = float(n)
        if np.isnan(n): n = 0.0
    except Exception: n = 0.0
    n = int(n); negatif = n < 0; absn = abs(n)
    if singkat:
        if absn >= 1_000_000_000: teks = f"Rp {absn/1_000_000_000:.1f} M"
        elif absn >= 1_000_000:   teks = f"Rp {absn/1_000_000:.1f} jt"
        elif absn >= 1_000:       teks = f"Rp {absn/1_000:.0f} rb"
        else:                      teks = "Rp " + f"{absn:,}".replace(",",".")
    else:
        teks = "Rp " + f"{absn:,}".replace(",",".")
    return ("-" + teks) if negatif else teks

def clr_health(v): return CLR_GREEN if v >= 70 else ("#F39C12" if v >= 40 else CLR_RED)
def dot_health(v):
    c = clr_health(v)
    return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};vertical-align:middle;margin-right:5px;"></span>'

# ============================================================
# BERSIHKAN DATA
# ============================================================

def bersihkan_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna(how='all').reset_index(drop=True)
    df.columns = df.columns.str.strip()
    col_map = {}
    for col in df.columns:
        c = col.lower().strip()
        if   any(k in c for k in ['tanggal','date','tgl']):            col_map.setdefault('Tanggal',     col)
        elif any(k in c for k in ['pemasukan','income','masuk','debit_masuk','kredit_masuk']): col_map.setdefault('Pemasukan',  col)
        elif any(k in c for k in ['pengeluaran','expense','keluar','kredit','bayar','debit_keluar']): col_map.setdefault('Pengeluaran',col)
        elif any(k in c for k in ['saldo','balance','kas','neraca']):   col_map.setdefault('Saldo',      col)
        elif any(k in c for k in ['keterangan','ket','description','catatan','uraian','remark']): col_map.setdefault('Keterangan', col)
        elif any(k in c for k in ['kategori','category','jenis','type']): col_map.setdefault('Kategori',   col)
    df = df.rename(columns={v: k for k, v in col_map.items()})
    missing = [c for c in ['Tanggal','Pemasukan','Pengeluaran'] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: **{', '.join(missing)}**\n\nKolom tersedia: {', '.join(df.columns.tolist())}")
    for col in ['Pemasukan','Pengeluaran','Saldo']:
        if col not in df.columns: continue
        df[col] = (df[col].astype(str)
            .str.replace(r'\(([0-9.,]+)\)', r'-\1', regex=True)
            .str.replace(r'[Rp\s$]|IDR', '', regex=True)
            .str.replace(r'\.(?=\d{3}(?:[.,]|$))', '', regex=True)
            .str.replace(',', '.', regex=False).str.strip())
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Deteksi format tanggal otomatis - JANGAN hardcode dayfirst=True karena
    # akan salah parse format ISO YYYY-MM-DD (misal: 2024-01-02 jadi 1 Feb bukan 2 Jan)
    sample_tgl = df['Tanggal'].dropna().astype(str).iloc[0] if not df['Tanggal'].dropna().empty else ''
    if sample_tgl and '-' in sample_tgl and len(sample_tgl.split('-')[0]) == 4:
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=False, errors='coerce')
    else:
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Tanggal']).sort_values('Tanggal').reset_index(drop=True)
    if df.empty: raise ValueError("Tidak ada baris valid. Periksa format kolom Tanggal.")
    df['Tahun'] = df['Tanggal'].dt.year; df['Bulan'] = df['Tanggal'].dt.month
    df['Hari']  = df['Tanggal'].dt.day;  df['BulanNama']   = df['Bulan'].map(MONTH_ID)
    df['BulanPendek'] = df['Bulan'].map(MONTH_SHORT)
    if 'Saldo' not in df.columns or df['Saldo'].abs().sum() == 0:
        df['Saldo'] = (df['Pemasukan'] - df['Pengeluaran']).cumsum()
    if 'Keterangan' not in df.columns: df['Keterangan'] = ''
    if 'Kategori'   not in df.columns: df['Kategori']   = 'Umum'
    return df

# ============================================================
# AGREGASI BULANAN
# ============================================================

def agregasi_bulanan(df: pd.DataFrame, tahun: int) -> pd.DataFrame:
    d = df[df['Tahun'] == tahun]
    if d.empty: return pd.DataFrame()
    m = (d.groupby(['Bulan','BulanNama','BulanPendek'])
         .agg(Pemasukan=('Pemasukan','sum'), Pengeluaran=('Pengeluaran','sum'),
              SaldoAkhir=('Saldo','last'), JmlTransaksi=('Pemasukan','count'))
         .reset_index().sort_values('Bulan'))
    m['Surplus']    = m['Pemasukan'] - m['Pengeluaran']
    m['RasioBeban'] = np.where(m['Pemasukan'] > 0, m['Pengeluaran'] / m['Pemasukan'] * 100, 100.0)
    m['Status']     = m['Surplus'].apply(lambda x: 'Surplus' if x >= 0 else 'Defisit')
    return m

# ============================================================
# DETEKSI ANOMALI
# ============================================================

def deteksi_anomali(monthly: pd.DataFrame) -> list:
    anomali = []
    if len(monthly) < 3: return anomali
    for col, label in [('Pengeluaran','Pengeluaran'),('Pemasukan','Pemasukan')]:
        mean_v = monthly[col].mean(); std_v = monthly[col].std()
        if std_v < 1: continue
        for _, row in monthly.iterrows():
            z = (row[col] - mean_v) / std_v
            if abs(z) > 2.0:
                arah = "sangat TINGGI" if z > 0 else "sangat RENDAH"
                anomali.append({'bulan': row['BulanNama'], 'kolom': label, 'nilai': row[col], 'arah': arah, 'z': z})
    return anomali

# ============================================================
# PREDIKSI LSTM (INTEGRASI BARU)
# ============================================================

@st.cache_data(show_spinner="🧠 Model LSTM sedang melatih dan mempelajari pola harian data Anda...",
               hash_funcs={pd.DataFrame: lambda df: hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()})
def prediksi_lstm(df_asli: pd.DataFrame, last_bln: int, tahun: int, last_sal: float) -> dict:
    """Fungsi Prediksi menggunakan Deep Learning LSTM berbasis Data Harian"""
    
    # Lazy import TensorFlow agar app tidak crash jika belum terinstall
    if not _TF_AVAILABLE:
        # Fallback ke rata-rata historis jika TensorFlow tidak tersedia
        df_harian_fb = df_asli.groupby('Tanggal')[['Pemasukan', 'Pengeluaran']].sum().reset_index()
        avg_pem_fb = float(df_harian_fb['Pemasukan'].mean() * 30)
        avg_pen_fb = float(df_harian_fb['Pengeluaran'].mean() * 30)
        
        def next_month_name_fb(base_bln, offset):
            bln = ((base_bln - 1 + offset) % 12) + 1
            thn = tahun + (base_bln - 1 + offset) // 12
            return MONTH_ID[bln], bln

        fc3_fb = []
        saldo_accum_fb = last_sal
        for i in range(3):
            sur = avg_pem_fb - avg_pen_fb
            saldo_accum_fb += sur
            bn_nama, bn_num = next_month_name_fb(last_bln, i+1)
            fc3_fb.append({
                'bulan': bn_nama, 'bulan_num': bn_num,
                'pemasukan': round(avg_pem_fb), 'pengeluaran': round(avg_pen_fb),
                'surplus': round(sur), 'saldo_akhir': round(saldo_accum_fb),
                'ci_pem_80': round(avg_pem_fb * 0.15), 'ci_pen_80': round(avg_pen_fb * 0.15)
            })
        return {
            'pemasukan': fc3_fb[0]['pemasukan'], 'pengeluaran': fc3_fb[0]['pengeluaran'],
            'surplus': fc3_fb[0]['surplus'], 'saldo': fc3_fb[0]['saldo_akhir'],
            'bulan_nama': fc3_fb[0]['bulan'],
            'tahun': tahun if fc3_fb[0]['bulan_num'] > last_bln else tahun + 1,
            'confidence': "Rendah (TensorFlow tidak tersedia)", 'model': "Rata-rata Historis (TF tidak install)",
            'n_bulan': max(1, len(df_asli) // 30), 'avg_pem': avg_pem_fb, 'avg_pen': avg_pen_fb,
            'forecast_3bln': fc3_fb
        }

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # 1. Konversi ke data harian agar jumlah baris cukup untuk training LSTM
    df_harian = df_asli.groupby('Tanggal')[['Pemasukan', 'Pengeluaran']].sum().reset_index()
    df_harian.set_index('Tanggal', inplace=True)
    
    # Isi tanggal yang bolong dengan nilai 0 agar berurutan (Time-Series requirement)
    idx = pd.date_range(df_harian.index.min(), df_harian.index.max())
    df_harian = df_harian.reindex(idx, fill_value=0)
    
    data = df_harian[['Pemasukan', 'Pengeluaran']].values
    n_data = len(data)
    
    TIME_STEPS = 30
    
    # HELPER untuk nama bulan
    def next_month_name(base_bln, offset):
        bln = ((base_bln - 1 + offset) % 12) + 1
        thn = tahun + (base_bln - 1 + offset) // 12
        return MONTH_ID[bln], bln

    fc3 = []
    avg_pem_hist = float(df_harian['Pemasukan'].mean() * 30)
    avg_pen_hist = float(df_harian['Pengeluaran'].mean() * 30)

    # FALLBACK: Jika data terlalu sedikit (< 45 hari), LSTM tidak akan bekerja optimal
    if n_data <= 45:
        saldo_accum = last_sal
        for i in range(3):
            sur = avg_pem_hist - avg_pen_hist
            saldo_accum += sur
            bn_nama, bn_num = next_month_name(last_bln, i+1)
            fc3.append({
                'bulan': bn_nama, 'bulan_num': bn_num,
                'pemasukan': round(avg_pem_hist), 'pengeluaran': round(avg_pen_hist),
                'surplus': round(sur), 'saldo_akhir': round(saldo_accum),
                'ci_pem_80': round(avg_pem_hist * 0.15), 'ci_pen_80': round(avg_pen_hist * 0.15)
            })
        return {
            'pemasukan': fc3[0]['pemasukan'], 'pengeluaran': fc3[0]['pengeluaran'],
            'surplus': fc3[0]['surplus'], 'saldo': fc3[0]['saldo_akhir'],
            'bulan_nama': fc3[0]['bulan'], 'tahun': tahun if fc3[0]['bulan_num'] > last_bln else tahun + 1,
            'confidence': "Rendah (Data Terbatas)", 'model': "Rata-rata Historis (Data LSTM kurang)",
            'n_bulan': max(1, n_data // 30), 'avg_pem': avg_pem_hist, 'avg_pen': avg_pen_hist,
            'forecast_3bln': fc3
        }

    # 2. Scaling Data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 3. Membuat Sequences (X) dan Target (y)
    X, y = [], []
    for i in range(len(data_scaled) - TIME_STEPS):
        X.append(data_scaled[i:i+TIME_STEPS])
        y.append(data_scaled[i+TIME_STEPS])
    X, y = np.array(X), np.array(y)
    
    # 4. Bangun Arsitektur LSTM (Ringan & Cepat untuk Dashboard)
    model = Sequential([
        LSTM(32, input_shape=(TIME_STEPS, 2)),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # 5. Training
    model.fit(X, y, epochs=30, batch_size=16, verbose=0)
    
    # 6. Prediksi Iteratif untuk 90 Hari ke Depan (3 Bulan)
    future_preds = []
    curr_batch = data_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, 2)
    
    for _ in range(90):
        p = model.predict(curr_batch, verbose=0)
        future_preds.append(p[0])
        # Update batch: geser data, masukkan hasil prediksi terbaru
        curr_batch = np.append(curr_batch[:, 1:, :], [p], axis=1)
        
    # 7. Kembalikan Skala Nilai (Inverse Transform)
    future_rp = scaler.inverse_transform(future_preds)
    
    # Hindari nilai negatif dari output NN
    future_pem = np.maximum(0, future_rp[:, 0])
    future_pen = np.maximum(0, future_rp[:, 1])
    
    # 8. Agregasikan Prediksi Harian ke format Bulanan (3 Bulan)
    saldo_accum = last_sal
    for i in range(3):
        # Ambil potongan 30 hari untuk tiap bulan
        p_val = float(future_pem[i*30:(i+1)*30].sum())
        e_val = float(future_pen[i*30:(i+1)*30].sum())
        sur = p_val - e_val
        saldo_accum += sur
        bn_nama, bn_num = next_month_name(last_bln, i+1)
        
        fc3.append({
            'bulan': bn_nama, 'bulan_num': bn_num,
            'pemasukan': round(p_val), 'pengeluaran': round(e_val),
            'surplus': round(sur), 'saldo_akhir': round(saldo_accum),
            'ci_pem_80': round(p_val * 0.12), # Rentang simpangan
            'ci_pen_80': round(e_val * 0.12)
        })

    return {
        'pemasukan': fc3[0]['pemasukan'],
        'pengeluaran': fc3[0]['pengeluaran'],
        'surplus': fc3[0]['surplus'],
        'saldo': fc3[0]['saldo_akhir'],
        'bulan_nama': fc3[0]['bulan'],
        'tahun': tahun if fc3[0]['bulan_num'] > last_bln else tahun + 1,
        'confidence': "Tinggi" if n_data >= 180 else "Sedang",
        'model': "LSTM (Long Short-Term Memory)",
        'n_bulan': max(1, n_data // 30),
        'avg_pem': avg_pem_hist,
        'avg_pen': avg_pen_hist,
        'forecast_3bln': fc3
    }


# ============================================================
# HEALTH SCORE
# ============================================================

def hitung_health(monthly: pd.DataFrame) -> dict:
    ti, to = float(monthly['Pemasukan'].sum()), float(monthly['Pengeluaran'].sum()); n = len(monthly)
    tabungan   = max(0.0, min(100.0, (ti - to) / ti * 100)) if ti > 0 else 0.0
    stabilitas = float((monthly['Surplus'] >= 0).sum()) / n * 100 if n > 0 else 0.0
    if n >= 2:
        f = float(monthly['SaldoAkhir'].iloc[0]); l = float(monthly['SaldoAkhir'].iloc[-1])
        if f <= 0: ref = max(1.0, float(monthly['Pengeluaran'].mean())); pertumbuhan = max(0.0, min(100.0, (l/ref)*50))
        else: pct_grow = (l - f) / f * 100; pertumbuhan = max(0.0, min(100.0, pct_grow * 2))
    else: pertumbuhan = 50.0
    avg_pen = float(monthly['Pengeluaran'].mean()); last_sal = float(monthly['SaldoAkhir'].iloc[-1]) if n > 0 else 0.0
    if avg_pen > 0 and last_sal > 0: bln_aman = last_sal / avg_pen; cadangan = min(100.0, bln_aman / 3 * 100)
    elif last_sal <= 0: bln_aman, cadangan = 0.0, 0.0
    else: bln_aman, cadangan = 0.0, 50.0
    return {'tabungan': round(tabungan,1), 'stabilitas': round(stabilitas,1),
            'pertumbuhan': round(pertumbuhan,1), 'cadangan': round(cadangan,1), 'bln_aman': round(bln_aman,1)}

# ============================================================
# SIDEBAR
# ============================================================

@st.cache_data(show_spinner=False)
def _baca_cepat(file_bytes: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_excel(io.BytesIO(file_bytes), header=0)
        if not any(k in ' '.join(raw.columns.str.lower()) for k in ['tanggal','date','tgl']):
            raw = pd.read_excel(io.BytesIO(file_bytes), header=1)
        return bersihkan_data(raw)
    except Exception: return pd.DataFrame()

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-wrap">
                <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="3" y1="22" x2="21" y2="22"/>
                    <line x1="6" y1="18" x2="6" y2="11"/>
                    <line x1="10" y1="18" x2="10" y2="11"/>
                    <line x1="14" y1="18" x2="14" y2="11"/>
                    <line x1="18" y1="18" x2="18" y2="11"/>
                    <polygon points="12 2 20 7 4 7"/>
                </svg>
            </div>
            <span class="logo-title">Karang Taruna</span>
            <span class="logo-sub">Portal Keuangan Desa · v4.0</span>
        </div>
        
        <style>
        /* Membuat semua teks bawaan yang bertumpukan jadi transparan */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] section div {
            color: transparent !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            height: 0px !important;
            padding: 0px !important;
            margin: 0px !important;
        }
        /* Mengisi kotak dropzone dengan teks baru yang rapi */
        [data-testid="stFileUploaderDropzone"] section::before {
            content: "✨ Klik di sini untuk upload file";
            color: #E67E22;
            font-weight: 600;
            font-size: 0.95rem;
            display: block;
            text-align: center;
            padding: 20px 0;
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("### Unggah Data Keuangan")
        uploaded = st.file_uploader("", type=['xlsx','xls'], label_visibility="collapsed",
            help="Upload file rekap Excel (.xlsx/.xls). Kolom wajib: Tanggal, Pemasukan, Pengeluaran. Opsional: Saldo, Keterangan, Kategori")
        
        tahun_sel = None; bulan_sel = "Semua Bulan"

        if uploaded:
            df_preview = _baca_cepat(uploaded.getvalue())
            if not df_preview.empty:
                # Reset cache jika file berubah (cek berdasarkan ukuran file sebagai proxy)
                file_size = len(uploaded.getvalue())
                if st.session_state.get('_last_file_size') != file_size:
                    st.session_state.pop('df_clean', None)
                    st.session_state['_last_file_size'] = file_size
                st.session_state['df_clean'] = df_preview
                st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
                st.markdown("### Filter Tampilan")
                tahun_list = sorted(df_preview['Tahun'].unique(), reverse=True)
                tahun_sel  = st.selectbox("Pilih Tahun", options=tahun_list, format_func=lambda x: f"Tahun {x}")
                bulan_tersedia = sorted(df_preview[df_preview['Tahun'] == tahun_sel]['Bulan'].unique())
                bulan_opts = ['Semua Bulan'] + [MONTH_ID[b] for b in bulan_tersedia]
                bulan_sel  = st.selectbox("Pilih Bulan", options=bulan_opts)

                df_fil = df_preview[df_preview['Tahun'] == tahun_sel]
                if bulan_sel != 'Semua Bulan':
                    bln_num = {v: k for k, v in MONTH_ID.items()}.get(bulan_sel)
                    if bln_num: df_fil = df_fil[df_fil['Bulan'] == bln_num]
                total_pem = df_fil['Pemasukan'].sum(); total_pen = df_fil['Pengeluaran'].sum()
                n_trx = len(df_fil); n_bln = df_fil['Bulan'].nunique()
                surplus = total_pem - total_pen
                label_periode = bulan_sel if bulan_sel != 'Semua Bulan' else f"Tahun {tahun_sel}"
                badge_cls = "surplus" if surplus >= 0 else "defisit"
                badge_txt = "Surplus" if surplus >= 0 else "Defisit"

                st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="sidebar-info">
                    <strong>{label_periode}</strong><br>
                    {n_trx} transaksi &nbsp;·&nbsp; {n_bln} bulan<br>
                    Masuk: <strong>{rupiah(total_pem, True)}</strong><br>
                    Keluar: <strong>{rupiah(total_pen, True)}</strong><br>
                    Selisih: <strong>{rupiah(surplus, True)}</strong><br><br>
                    <strong>Diperbarui</strong><br>
                    {datetime.now().strftime('%d %b %Y · %H:%M WIB')}
                    <br><span class="sidebar-badge {badge_cls}">{badge_txt}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <hr class="sidebar-divider">
                <div class="sidebar-nav">
                    <strong>Menu Navigasi</strong><br>
                    <strong>Ringkasan</strong> — KPI &amp; Kesehatan<br>
                    <strong>Grafik</strong> — Visualisasi Data<br>
                    <strong>Prediksi</strong> — Perkiraan Bulan Depan<br>
                    <strong>Detail</strong> — Tabel Transaksi<br>
                    <strong>Tips</strong> — Saran Keuangan
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("File tidak valid. Pastikan ada kolom Tanggal, Pemasukan, Pengeluaran.")

        st.markdown("""
        <div class="sidebar-footer">
            Karang Taruna &nbsp;·&nbsp; Transparansi &amp; Kemajuan Bersama
        </div>""", unsafe_allow_html=True)
    return uploaded, tahun_sel, bulan_sel

# ============================================================
# WELCOME SCREEN
# ============================================================

def render_welcome():
    st.markdown("""
    <div class="welcome-box">
        <div class="welcome-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="3" y1="22" x2="21" y2="22"/>
                <line x1="6" y1="18" x2="6" y2="11"/>
                <line x1="10" y1="18" x2="10" y2="11"/>
                <line x1="14" y1="18" x2="14" y2="11"/>
                <line x1="18" y1="18" x2="18" y2="11"/>
                <polygon points="12 2 20 7 4 7"/>
            </svg>
        </div>
        <div class="welcome-title">Selamat Datang di Dashboard Keuangan</div>
        <div class="welcome-desc">
            Dashboard ini membantu warga desa dan anggota Karang Taruna memantau keuangan
            organisasi secara mudah, jelas, dan transparan — tanpa perlu memahami ilmu data.
        </div>
        <div class="welcome-steps">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-title">Unggah Data Excel</div>
                <div class="step-desc">Klik tombol upload di sidebar kiri, lalu pilih file Excel keuangan Anda.</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-title">Pilih Tahun / Bulan</div>
                <div class="step-desc">Pilih tahun atau bulan yang ingin Anda lihat lewat filter di sidebar.</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-title">Lihat Hasilnya</div>
                <div class="step-desc">Grafik, prediksi 3 bulan dengan LSTM, dan saran keuangan tampil otomatis.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Format File Excel yang Dibutuhkan</div>', unsafe_allow_html=True)
    contoh = pd.DataFrame({'Tanggal':['01/01/2024','07/01/2024','14/01/2024','21/01/2024'],
        'Pemasukan':['500.000','750.000','1.200.000','300.000'], 'Pengeluaran':['200.000','400.000','600.000','150.000'],
        'Saldo':['300.000','650.000','1.250.000','1.400.000'],
        'Keterangan':['Iuran Warga','Donasi','Dana Desa','Arisan'], 'Kategori':['Iuran','Donasi','Dana Pemerintah','Sosial']})
    st.dataframe(contoh, use_container_width=True, hide_index=True)
    st.info("Kolom **Tanggal, Pemasukan, Pengeluaran** wajib ada. Kolom **Saldo, Keterangan, Kategori** opsional. Angka boleh pakai titik ribuan (1.000.000) atau tanpa titik.")

# ============================================================
# KPI CARDS
# ============================================================

def _delta_label(monthly, col):
    if len(monthly) < 2: return ""
    last = monthly[col].iloc[-1]; prev = monthly[col].iloc[-2]
    if prev == 0: return ""
    d = (last - prev) / prev * 100; arrow = "▲" if d >= 0 else "▼"
    color = CLR_GREEN if d >= 0 else CLR_RED
    return f'<div class="kpi-delta" style="color:{color};">{arrow} {abs(d):.1f}% vs bulan lalu</div>'

def render_kpi(monthly, pred):
    ti = monthly['Pemasukan'].sum(); to = monthly['Pengeluaran'].sum()
    sur = ti - to; sal = monthly['SaldoAkhir'].iloc[-1]; pct = (sur / ti * 100) if ti > 0 else 0.0
    _sur_clr = CLR_GREEN if sur >= 0 else CLR_RED
    # ── SVG helpers ──────────────
    def _svg(stroke, paths):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" '
                f'viewBox="0 0 24 24" fill="none" stroke="{stroke}" '
                f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                f'{paths}</svg>')

    ico_bank   = _svg(CLR_NAVY,
                      '<line x1="3" y1="22" x2="21" y2="22"/>'
                      '<line x1="6" y1="18" x2="6" y2="11"/>'
                      '<line x1="10" y1="18" x2="10" y2="11"/>'
                      '<line x1="14" y1="18" x2="14" y2="11"/>'
                      '<line x1="18" y1="18" x2="18" y2="11"/>'
                      '<polygon points="12 2 20 7 4 7"/>')

    ico_up     = _svg(CLR_NAVY,
                      '<line x1="12" y1="19" x2="12" y2="5"/>'
                      '<polyline points="5 12 12 5 19 12"/>')

    ico_down   = _svg(CLR_NAVY,
                      '<line x1="12" y1="5" x2="12" y2="19"/>'
                      '<polyline points="19 12 12 19 5 12"/>')

    ico_sur    = (f'<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" '
                  f'viewBox="0 0 24 24" fill="{CLR_NAVY}" stroke="{CLR_NAVY}" '
                  f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                  + ('<polygon points="12 4 22 20 2 20"/>' if sur >= 0
                     else '<polygon points="12 20 2 4 22 4"/>')
                  + '</svg>')

    ico_pct    = _svg(CLR_NAVY,
                      '<line x1="19" y1="5" x2="5" y2="19"/>'
                      '<circle cx="6.5" cy="6.5" r="2.5"/>'
                      '<circle cx="17.5" cy="17.5" r="2.5"/>')

    kartu = [
        {'w': CLR_NAVY,   'icon': ico_bank, 'label':'Saldo Kas',
         'nilai': rupiah(sal, True),  'sub':'Uang tunai tersedia saat ini',   'delta':''},
        {'w': CLR_GREEN,  'icon': ico_up,   'label':'Total Pemasukan',
         'nilai': rupiah(ti, True),   'sub':'Semua uang yang masuk',          'delta': _delta_label(monthly,'Pemasukan')},
        {'w': CLR_RED,    'icon': ico_down, 'label':'Total Pengeluaran',
         'nilai': rupiah(to, True),   'sub':'Semua uang yang keluar',         'delta': _delta_label(monthly,'Pengeluaran')},
        {'w': _sur_clr,   'icon': ico_sur,  'label':'Surplus / Defisit',
         'nilai': rupiah(sur, True),
         'sub':'Keuangan sehat' if sur >= 0 else 'Pengeluaran melebihi pemasukan', 'delta':''},
        {'w': CLR_GOLD,   'icon': ico_pct,  'label':'Persentase Ditabung',
         'nilai': f"{max(0, pct):.1f}%",
         'sub': f"dari pemasukan  ·  {'Target >20% tercapai' if pct >= 20 else 'Target >20%'}", 'delta':''},
    ]
    cols = st.columns(5)
    for col, k in zip(cols, kartu):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{k['w']};">
                <div style="height:30px;margin-bottom:6px;display:flex;align-items:center;justify-content:center;color:{k['w']};font-size:1.4rem;font-weight:800;line-height:1;">{k['icon']}</div>
                <div class="kpi-label">{k['label']}</div>
                <div class="kpi-value">{k['nilai']}</div>
                <div class="kpi-sub">{k['sub']}</div>
                {k['delta']}
            </div>""", unsafe_allow_html=True)

# ============================================================
# PREDIKSI BOX + FORECAST 3 BULAN
# ============================================================

def render_prediksi(pred, monthly):
    sur = pred['surplus']; t = "+" if sur >= 0 else ""
    dp = ((pred['pemasukan'] - pred['avg_pem']) / pred['avg_pem'] * 100) if pred['avg_pem'] > 0 else 0
    de = ((pred['pengeluaran'] - pred['avg_pen']) / pred['avg_pen'] * 100) if pred['avg_pen'] > 0 else 0
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-box-title">Prediksi Keuangan — {pred['bulan_nama']} {pred['tahun']}</div>
        <div class="pred-box-sub">
            Model: <strong>{pred['model']}</strong> &nbsp;·&nbsp;
            Data Historis: {pred['n_bulan']} bulan &nbsp;·&nbsp;
            Kepercayaan LSTM: <strong>{pred['confidence']}</strong>
        </div>
        <div class="pred-grid">
            <div class="pred-item">
                <div class="pred-item-label">Perkiraan Pemasukan</div>
                <div class="pred-item-value">{rupiah(pred['pemasukan'],True)}</div>
                <div class="pred-item-delta">{"▲" if dp>=0 else "▼"} {abs(dp):.1f}% vs rata-rata</div>
            </div>
            <div class="pred-item">
                <div class="pred-item-label">Perkiraan Pengeluaran</div>
                <div class="pred-item-value">{rupiah(pred['pengeluaran'],True)}</div>
                <div class="pred-item-delta">{"▲" if de>=0 else "▼"} {abs(de):.1f}% vs rata-rata</div>
            </div>
            <div class="pred-item">
                <div class="pred-item-label">Perkiraan {"Surplus" if sur>=0 else "Defisit"}</div>
                <div class="pred-item-value">{t}{rupiah(sur,True)}</div>
                <div class="pred-item-delta">{"Keuangan aman" if sur>=0 else "Siapkan cadangan"}</div>
            </div>
            <div class="pred-item">
                <div class="pred-item-label">Prediksi Saldo Akhir</div>
                <div class="pred-item-value">{rupiah(pred['saldo'],True)}</div>
                <div class="pred-item-delta">Setelah {pred['bulan_nama']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if pred['n_bulan'] < 3:
        st.warning(f"Data hanya {pred['n_bulan']} bulan — model LSTM bekerja lebih baik dengan data yang panjang. Jika kurang, maka digunakan nilai rata-rata biasa.")

    st.markdown('<div class="section-title">Perkiraan 3 Bulan ke Depan</div>', unsafe_allow_html=True)
    fc3 = pred.get('forecast_3bln', [])
    if fc3:
        cols = st.columns(3)
        for i, fc in enumerate(fc3):
            with cols[i]:
                sur_fc = fc['surplus']; clr = CLR_NAVY if sur_fc >= 0 else CLR_RED; ci_p = fc.get('ci_pem_80', 0)
                st.markdown(f"""
                <div class="kpi-card" style="border-top-color:{clr}; padding-top:18px;">
                    <div class="kpi-label">Bulan {i+1} — {fc['bulan']}</div>
                    <div class="kpi-value" style="color:{clr};">{'+'if sur_fc>=0 else ''}{rupiah(sur_fc, True)}</div>
                    <div class="kpi-sub">
                        Masuk: {rupiah(fc['pemasukan'],True)}<br>
                        Keluar: {rupiah(fc['pengeluaran'],True)}
                    </div>
                    <div class="kpi-delta" style="color:#8090B5; font-size:0.60rem; margin-top:5px;">
                        ± {rupiah(ci_p, True)} (rentang error)
                    </div>
                </div>""", unsafe_allow_html=True)

    if len(monthly) >= 3:
        st.plotly_chart(chart_forecast(monthly, pred), use_container_width=True)

    sur_clr = CLR_GREEN if sur >= 0 else CLR_RED
    pesan_sur = "Lebih banyak uang masuk dari keluar — keuangan aman." if sur >= 0 else "Pengeluaran lebih besar dari pemasukan — siapkan tabungan cadangan."
    st.markdown(f"""
    <div style="background:#EEF3FB;border-radius:14px;padding:20px 22px;border-left:5px solid var(--navy);margin-top:6px;font-size:0.85rem;color:#1A2D5A;line-height:1.80;">
        <div style="font-weight:800;font-size:0.95rem;color:#1B3A6B;margin-bottom:12px;">Ringkasan Prediksi — {pred['bulan_nama']} {pred['tahun']}</div>
        Perkiraan <strong>Pemasukan</strong>: <strong>{rupiah(pred['pemasukan'])}</strong><br>
        Perkiraan <strong>Pengeluaran</strong>: <strong>{rupiah(pred['pengeluaran'])}</strong><br>
        <span style="color:{sur_clr};font-weight:700;">{pesan_sur}</span><br>
        Prediksi <strong>Saldo Akhir</strong>: <strong>{rupiah(pred['saldo'])}</strong><br>
        Kepercayaan Model: <strong>{pred['confidence']}</strong>
        <div style="margin-top:14px;padding-top:10px;border-top:1px dashed #C9D6EE;font-size:0.76rem;color:#6B7A99;">
            Prediksi ini diekstraksi dari model LSTM berbasis data harian Anda.
        </div>
    </div>
    """, unsafe_allow_html=True)

    anomali = deteksi_anomali(monthly)
    if anomali:
        st.markdown('<div class="section-title">Anomali Keuangan Terdeteksi</div>', unsafe_allow_html=True)
        for a in anomali:
            badge = "Naik" if "TINGGI" in a['arah'] else "Turun"
            st.markdown(f"""
            <div class="anomali-box">
                <div class="anomali-badge">{badge}</div>
                <div>
                    <div class="anomali-title">Bulan {a['bulan']} — {a['kolom']} {a['arah']}</div>
                    <div class="anomali-text">Nilai tercatat: <strong>{rupiah(a['nilai'], True)}</strong> — sangat jauh dari rata-rata (z-score = {a['z']:.1f}). Periksa kembali transaksi bulan ini.</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ============================================================
# GRAFIK
# ============================================================

def chart_saldo(monthly):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['BulanNama'], y=monthly['SaldoAkhir'], mode='lines+markers', name='Saldo Aktual',
        line=dict(color=CLR_NAVY, width=3), fill='tozeroy', fillcolor='rgba(27,58,107,0.10)',
        marker=dict(size=9, color=CLR_NAVY), hovertemplate='<b>%{x}</b><br>Saldo: Rp %{y:,.0f}<extra></extra>'))
    if len(monthly) >= 2:
        xs = np.arange(len(monthly), dtype=float).reshape(-1,1)
        trend = LinearRegression().fit(xs, monthly['SaldoAkhir'].values).predict(xs)
        fig.add_trace(go.Scatter(x=monthly['BulanNama'], y=trend, mode='lines', name='Garis Tren',
            line=dict(color=CLR_GOLD, width=2.5, dash='dot'), hovertemplate='<b>%{x}</b><br>Tren: Rp %{y:,.0f}<extra></extra>'))
    fig.update_layout(title=dict(text="Pergerakan Saldo Kas + Garis Tren", font=dict(size=13, color=CLR_DARK)),
        yaxis=dict(tickformat=",", tickprefix="Rp ", gridcolor="#DDE3EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"), legend=dict(orientation="h", y=1.10, x=0), **LAYOUT)
    return fig

def chart_bulanan(monthly):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly['BulanPendek'], y=monthly['Pemasukan'], name='Pemasukan',
        marker_color=CLR_NAVY, marker_line_width=0, hovertemplate='%{x}<br>Pemasukan: Rp %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(x=monthly['BulanPendek'], y=monthly['Pengeluaran'], name='Pengeluaran',
        marker_color=CLR_GOLD, marker_line_width=0, hovertemplate='%{x}<br>Pengeluaran: Rp %{y:,.0f}<extra></extra>'))
    fig.update_layout(title=dict(text="Pemasukan vs Pengeluaran per Bulan", font=dict(size=13, color=CLR_DARK)),
        barmode='group', yaxis=dict(tickformat=",", gridcolor="#DDE3EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"), legend=dict(orientation="h", y=1.10, x=0), **LAYOUT)
    return fig

def chart_surplus(monthly):
    colors = [CLR_NAVY if s >= 0 else CLR_RED for s in monthly['Surplus']]
    fig = go.Figure(go.Bar(x=monthly['BulanPendek'], y=monthly['Surplus'],
        marker_color=colors, marker_line_width=0, hovertemplate='%{x}<br>Rp %{y:,.0f}<extra></extra>'))
    fig.add_hline(y=0, line_dash="dash", line_color="#1B3A6B", line_width=1.5,
                  annotation_text="Titik Impas", annotation_position="top right", annotation_font_size=10)
    fig.update_layout(title=dict(text="Surplus / Defisit Tiap Bulan", font=dict(size=13, color=CLR_DARK)),
        yaxis=dict(tickformat=",", gridcolor="#DDE3EF"), xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False, **LAYOUT)
    return fig

def chart_donut(ti, to):
    if ti <= 0 and to <= 0: return go.Figure()
    if ti >= to:
        sur = ti - to; labels = ['Pengeluaran', 'Surplus / Ditabung']; values = [to, sur]; colors = [CLR_GOLD, CLR_NAVY]
        pct = sur / ti * 100 if ti > 0 else 0
        center = f"<b>{pct:.1f}%</b><br><span style='font-size:10px'>ditabung</span>"
    else:
        kelebihan = to - ti; labels = ['Ditanggung Pemasukan', 'Kelebihan Pengeluaran']; values = [ti, kelebihan]; colors = [CLR_NAVY, CLR_GOLD]
        pct = kelebihan / to * 100 if to > 0 else 0
        center = f"<b style='color:#E74C3C;'>defisit</b><br><span style='font-size:10px'>{pct:.1f}%</span>"
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors, line=dict(color='white', width=3)),
        hovertemplate='<b>%{label}</b><br>Rp %{value:,.0f} (%{percent})<extra></extra>'))
    fig.add_annotation(text=center, x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=CLR_DARK))
    fig.update_layout(title=dict(text="Komposisi Penggunaan Dana", font=dict(size=13, color=CLR_DARK)),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins, sans-serif", size=11), margin=dict(l=12,r=12,t=50,b=16))
    return fig

def chart_forecast(monthly, pred):
    fc3 = pred.get('forecast_3bln', [])
    if not fc3: return go.Figure()
    hist_x = monthly['BulanNama'].tolist(); hist_p = monthly['Pemasukan'].tolist(); hist_e = monthly['Pengeluaran'].tolist()
    fc_x = [f['bulan'] for f in fc3]; fc_p = [f['pemasukan'] for f in fc3]; fc_e = [f['pengeluaran'] for f in fc3]
    ci_p_hi = [f['pemasukan']+f.get('ci_pem_80',0)*((i+1)**0.5) for i,f in enumerate(fc3)]
    ci_p_lo = [max(0,f['pemasukan']-f.get('ci_pem_80',0)*((i+1)**0.5)) for i,f in enumerate(fc3)]
    ci_e_hi = [f['pengeluaran']+f.get('ci_pen_80',0)*((i+1)**0.5) for i,f in enumerate(fc3)]
    ci_e_lo = [max(0,f['pengeluaran']-f.get('ci_pen_80',0)*((i+1)**0.5)) for i,f in enumerate(fc3)]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Forecast Pemasukan","Forecast Pengeluaran"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=hist_x, y=hist_p, name='Aktual Pemasukan', line=dict(color=CLR_NAVY, width=2.5), marker=dict(size=7), mode='lines+markers', hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_x, y=ci_p_hi, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_x, y=ci_p_lo, name='Rentang Prediksi', mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(27,58,107,0.18)', showlegend=True, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_x, y=fc_p, name='Forecast Pemasukan', mode='lines+markers', line=dict(color=CLR_NAVY, width=2, dash='dot'), marker=dict(size=9, symbol='diamond', color=CLR_NAVY), hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_x, y=hist_e, name='Aktual Pengeluaran', line=dict(color=CLR_GOLD, width=2.5), marker=dict(size=7), mode='lines+markers', hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'), row=1, col=2)
    fig.add_trace(go.Scatter(x=fc_x, y=ci_e_hi, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter(x=fc_x, y=ci_e_lo, name='Rentang Prediksi Pengeluaran', mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(243,156,18,0.18)', showlegend=True, hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter(x=fc_x, y=fc_e, name='Forecast Pengeluaran', mode='lines+markers', line=dict(color=CLR_GOLD, width=2, dash='dot'), marker=dict(size=9, symbol='diamond', color=CLR_GOLD), hovertemplate='%{x}: Rp %{y:,.0f}<extra></extra>'), row=1, col=2)
    fig.update_layout(title=dict(text=f"Visualisasi Perkiraan 3 Bulan ke Depan ({pred['model']})", font=dict(size=13, color=CLR_DARK)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins, sans-serif", size=11, color="#1A2D5A"),
        margin=dict(l=12,r=12,t=80,b=80), legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", bgcolor="rgba(255,255,255,0.90)", bordercolor="#DDE3EF", borderwidth=1, font=dict(size=10)))
    for ann in fig.layout.annotations: ann.update(y=ann.y-0.04, font=dict(size=11, color="#1A2D5A"))
    fig.update_yaxes(tickformat=",", tickprefix="Rp ", gridcolor="#DDE3EF"); fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig

def chart_akurasi(monthly):
    n = len(monthly)
    if n < 4: return None
    xs = np.arange(1, n+1, dtype=float).reshape(-1,1); sp = max(2, int(n*0.75))
    if sp >= n: return None
    mp = LinearRegression().fit(xs[:sp], monthly['Pemasukan'].values[:sp])
    me = LinearRegression().fit(xs[:sp], monthly['Pengeluaran'].values[:sp])
    pp = mp.predict(xs[sp:]); pe = me.predict(xs[sp:]); xl = monthly['BulanNama'].values[sp:]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Akurasi Historis Pemasukan","Akurasi Historis Pengeluaran"))
    for col_idx, (actuals, preds, clr, name) in enumerate([(monthly['Pemasukan'].values[sp:], pp, CLR_NAVY, 'Pemasukan'),(monthly['Pengeluaran'].values[sp:], pe, CLR_GOLD, 'Pengeluaran')], 1):
        fig.add_trace(go.Scatter(x=xl, y=actuals, name=f'Aktual {name}', line=dict(color=clr, width=2.5), marker=dict(size=8), mode='lines+markers', showlegend=(col_idx==1)), row=1, col=col_idx)
        fig.add_trace(go.Scatter(x=xl, y=preds, name=f'Prediksi Linear {name}', line=dict(color=clr, width=2.5, dash='dot'), marker=dict(size=8, symbol='diamond'), showlegend=(col_idx==1)), row=1, col=col_idx)
    fig.update_layout(title=dict(text="Akurasi Historis: Data Nyata vs Perkiraan Tren", font=dict(size=13, color=CLR_DARK)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins, sans-serif", size=11, color="#1A2D5A"),
        margin=dict(l=12,r=12,t=80,b=80), legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", bgcolor="rgba(255,255,255,0.90)", bordercolor="#DDE3EF", borderwidth=1, font=dict(size=10)))
    for ann in fig.layout.annotations: ann.update(y=ann.y-0.04, font=dict(size=11, color="#1A2D5A"))
    fig.update_yaxes(tickformat=",", tickprefix="Rp ", gridcolor="#DDE3EF"); fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig

def chart_kategori(df, tahun):
    d = df[df['Tahun'] == tahun]
    if 'Kategori' not in d.columns: return None
    unique_kat = d['Kategori'].dropna().unique()
    if len(unique_kat) <= 1 and (len(unique_kat) == 0 or unique_kat[0] == 'Umum'): return None
    kat = (d.groupby('Kategori').agg(Pengeluaran=('Pengeluaran','sum'), Pemasukan=('Pemasukan','sum')).reset_index().sort_values('Pengeluaran', ascending=True))
    fig = go.Figure()
    fig.add_trace(go.Bar(y=kat['Kategori'], x=kat['Pengeluaran'], orientation='h', name='Pengeluaran', marker_color=CLR_GOLD, marker_line_width=0, hovertemplate='%{y}<br>Pengeluaran: Rp %{x:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(y=kat['Kategori'], x=kat['Pemasukan'], orientation='h', name='Pemasukan', marker_color=CLR_NAVY, marker_line_width=0, hovertemplate='%{y}<br>Pemasukan: Rp %{x:,.0f}<extra></extra>'))
    fig.update_layout(title=dict(text="Pemasukan & Pengeluaran per Kategori", font=dict(size=13, color=CLR_DARK)),
        barmode='group', xaxis=dict(tickformat=",", gridcolor="#DDE3EF"), yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=1.10, x=0), **LAYOUT)
    return fig

def chart_yoy(df, tahun_list):
    records = []
    for th in tahun_list:
        m = agregasi_bulanan(df, th)
        if m.empty: continue
        for _, row in m.iterrows():
            records.append({'Tahun': str(th), 'Bulan': row['Bulan'], 'BulanPendek': row['BulanPendek'], 'Pemasukan': row['Pemasukan'], 'Pengeluaran': row['Pengeluaran']})
    if not records: return None
    df_yoy = pd.DataFrame(records); colors_list = [CLR_NAVY, CLR_GREEN, CLR_GOLD, "#E67E22"]
    fig = go.Figure()
    for i, th in enumerate(tahun_list):
        sub = df_yoy[df_yoy['Tahun']==str(th)].sort_values('Bulan'); clr = colors_list[i % len(colors_list)]
        fig.add_trace(go.Scatter(x=sub['BulanPendek'], y=sub['Pemasukan'], mode='lines+markers', name=f'Pemasukan {th}',
            line=dict(color=clr, width=2), marker=dict(size=7), hovertemplate=f'%{{x}} {th}: Rp %{{y:,.0f}}<extra></extra>'))
    fig.update_layout(title=dict(text="Perbandingan Pemasukan Tahun ke Tahun", font=dict(size=13, color=CLR_DARK)),
        yaxis=dict(tickformat=",", tickprefix="Rp ", gridcolor="#DDE3EF"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"), legend=dict(orientation="h", y=1.10, x=0), **LAYOUT)
    return fig

# ============================================================
# HEALTH BARS
# ============================================================

def render_health(monthly):
    h = hitung_health(monthly)
    bars = [
        {'label':'Tingkat Tabungan',            'val': h['tabungan'],    'sub': f"{h['tabungan']:.1f}% dari pemasukan berhasil ditabung  ·  Target sehat: di atas 20%"},
        {'label':'Bulan dengan Keuangan Positif','val': h['stabilitas'],  'sub': f"{h['stabilitas']:.0f}% bulan berjalan surplus  ·  Semakin tinggi semakin stabil"},
        {'label':'Pertumbuhan Saldo',             'val': h['pertumbuhan'],'sub': f"Skor pertumbuhan: {h['pertumbuhan']:.1f}%  ·  Menandakan kas makin bertambah"},
        {'label':'Cadangan Kas Darurat',          'val': h['cadangan'],   'sub': f"Cukup untuk ≈ {h['bln_aman']} bulan ke depan  ·  Idealnya 2–3 bulan"},
    ]
    col_bars, col_score = st.columns([3, 1])
    with col_bars:
        for b in bars:
            c = clr_health(b['val'])
            st.markdown(f"""
            <div class="health-row">
                <div class="health-meta">
                    <span class="health-label">{b['label']}</span>
                    <span class="health-score" style="color:{c};">{dot_health(b['val'])} {b['val']:.0f}%</span>
                </div>
                <div class="health-bar-bg">
                    <div class="health-bar-fill" style="width:{b['val']}%; background:{c};"></div>
                </div>
                <div class="health-sub">{b['sub']}</div>
            </div>""", unsafe_allow_html=True)
    with col_score:
        avg = float(np.mean([h['tabungan'],h['stabilitas'],h['pertumbuhan'],h['cadangan']]))
        c = clr_health(avg)
        sts = "Sangat Sehat" if avg >= 70 else ("Perlu Perhatian" if avg >= 40 else "Kritis")
        st.markdown(f"""
        <div class="score-circle" style="background:{c}18; border:3px solid {c};">
            <div class="score-num" style="color:{c};">{avg:.0f}%</div>
            <div class="score-label" style="color:{c};">{sts}</div>
            <div class="score-sub">Skor Kesehatan<br>Keuangan Keseluruhan</div>
        </div>""", unsafe_allow_html=True)

# ============================================================
# A/B TEST
# ============================================================

def render_ab(df, tahun):
    dy = df[df['Tahun'] == tahun]
    a  = dy[dy['Hari'] <= 15]['Pemasukan'].dropna(); k = dy[dy['Hari'] > 15]['Pemasukan'].dropna()
    aa = float(a.mean()) if len(a) > 0 else 0.0; ak = float(k.mean()) if len(k) > 0 else 0.0
    aa = 0.0 if np.isnan(aa) else aa; ak = 0.0 if np.isnan(ak) else ak
    denom = max(1.0, (aa + ak) / 2); pct = abs(aa - ak) / denom * 100
    sig, pv = False, 1.0
    if len(a) >= 2 and len(k) >= 2: _, pv = ttest_ind(a, k, equal_var=False); sig = bool(pv < 0.05)
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.markdown(f"""
        <div class="ab-card awal">
            <div class="ab-period">Awal Bulan (Tgl 1–15)</div>
            <div class="ab-value">{rupiah(aa, True)}</div>
            <div class="ab-desc">Rata-rata pemasukan per transaksi</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="ab-card akhir">
            <div class="ab-period">Akhir Bulan (Tgl 16–31)</div>
            <div class="ab-value">{rupiah(ak, True)}</div>
            <div class="ab-desc">Rata-rata pemasukan per transaksi</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        if len(a) < 2 or len(k) < 2:
            st.markdown(f"""
            <div style="background-color: #F0F4F8; color: #4A5778; padding: 16px; border-radius: 12px; border: 2px solid #DDE3EF; font-size: 0.85rem; line-height: 1.5;">
                Data belum cukup untuk uji statistik (butuh minimal 2 transaksi per periode). Selisih saat ini: <strong>{pct:.1f}%</strong>.
            </div>
            """, unsafe_allow_html=True)
        elif sig:
            lb = "awal bulan" if aa > ak else "akhir bulan"
            st.markdown(f"""
            <div class="ab-card" style="background: var(--orange-light); border: 2px solid var(--orange); height: 100%; box-sizing: border-box;">
                <div class="ab-period">Hasil Analisis Pola</div>
                <div class="ab-value" style="color: var(--orange-dark); font-size: 1.38rem;">{pct:.1f}%</div>
                <div class="ab-desc">Signifikan di {lb}. Rencanakan pengeluaran saat kas penuh.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ab-card" style="background: #E8F8EF; border: 2px solid #A9DFBF; height: 100%; box-sizing: border-box;">
                <div class="ab-period">Hasil Analisis Pola</div>
                <div class="ab-value" style="color: var(--green); font-size: 1.38rem;">{pct:.1f}%</div>
                <div class="ab-desc">Pemasukan merata sepanjang bulan. Arus kas Anda cenderung stabil.</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TIPS & SARAN
# ============================================================

def render_tips(monthly, pred):
    ti = monthly['Pemasukan'].sum(); to = monthly['Pengeluaran'].sum()
    r = to / ti if ti > 0 else 1.0; sur = pred['surplus']; tips = []
    if r > 0.85:
        tips.append({'badge':'!','w':CLR_RED,'judul':'Pengeluaran Terlalu Besar',
            'isi':f'Lebih dari 85% pemasukan habis untuk pengeluaran ({r*100:.0f}%). Coba kurangi belanja yang tidak mendesak, atau cari sumber pemasukan tambahan.'})
    elif r > 0.70:
        tips.append({'badge':'i','w':CLR_GOLD,'judul':'Pengeluaran Perlu Diperhatikan',
            'isi':f'Pengeluaran sudah {r*100:.0f}% dari pemasukan. Masih aman, tapi mulailah menabung lebih banyak agar ada cadangan dana mendadak.'})
    else:
        tips.append({'badge':'OK','w':CLR_GREEN,'judul':'Keuangan Sangat Sehat',
            'isi':f'Pengeluaran hanya {r*100:.0f}% dari pemasukan. Pertahankan pola ini dan pertimbangkan menyisihkan sebagian untuk kegiatan produktif desa.'})
    if sur < 0:
        tips.append({'badge':'!','w':CLR_RED,'judul':f'Waspadai Bulan {pred["bulan_nama"]}',
            'isi':f'Prediksi LSTM bulan depan: pengeluaran diproyeksikan lebih besar dari pemasukan. Siapkan dana cadangan dari sekarang.'})
    else:
        tips.append({'badge':'OK','w':CLR_GREEN,'judul':f'Proyeksi Bulan {pred["bulan_nama"]} Positif',
            'isi':f'Berdasarkan LSTM, perkiraan surplus ≈ {rupiah(sur,True)} bulan depan. Manfaatkan untuk menambah tabungan atau kegiatan sosial.'})
    best = monthly.loc[monthly['Pemasukan'].idxmax()]
    tips.append({'badge':'#1','w':CLR_NAVY,'judul':f'Bulan Terkuat: {best["BulanNama"]}',
        'isi':f'Pemasukan tertinggi di bulan {best["BulanNama"]} ({rupiah(best["Pemasukan"],True)}). Jadwalkan kegiatan atau pembelian penting saat kas sedang penuh.'})
    tips.append({'badge':'N','w':CLR_GREEN,'judul':'Selalu Catat Setiap Transaksi',
        'isi':'Model Artificial Intelligence (LSTM) yang bekerja di latar belakang sangat bergantung pada data harian. Catat semua transaksi Anda agar AI dapat memprediksi dengan akurat.'})
    cols = st.columns(2)
    for i, t in enumerate(tips):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="tip-card" style="border-color:{t['w']};">
                <div class="tip-badge" style="background:{t['w']};">{t['badge']}</div>
                <div>
                    <div class="tip-title">{t['judul']}</div>
                    <div class="tip-text">{t['isi']}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ============================================================
# TABEL REKAP
# ============================================================

def render_tabel(monthly, tahun):
    tampil = monthly[['BulanNama','Pemasukan','Pengeluaran','Surplus','RasioBeban','SaldoAkhir','JmlTransaksi','Status']].copy()
    tampil.columns = ['Bulan','Pemasukan','Pengeluaran','Surplus/Defisit','Rasio Beban','Saldo Akhir','Jml. Transaksi','Status']
    for col in ['Pemasukan','Pengeluaran','Surplus/Defisit','Saldo Akhir']: tampil[col] = tampil[col].apply(rupiah)
    tampil['Rasio Beban'] = tampil['Rasio Beban'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(tampil, use_container_width=True, hide_index=True)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        monthly.to_excel(writer, index=False, sheet_name='Rekap Keuangan')
    buf.seek(0)
    st.download_button(label="Download Rekap Excel", data=buf.getvalue(),
        file_name=f"Rekap_Keuangan_Karang_Taruna_{tahun}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ============================================================
# AKURASI MODEL
# ============================================================

def render_akurasi(monthly, pred):
    if len(monthly) < 4:
        st.info("Butuh minimal 4 bulan data untuk menampilkan grafik validasi prediksi.")
        return
    fig_ak = chart_akurasi(monthly)
    if fig_ak: st.plotly_chart(fig_ak, use_container_width=True)
    conf = pred.get('confidence', 'Rendah'); model = pred.get('model', '—'); n_bulan = pred.get('n_bulan', 0)
    if 'Tinggi' in conf: clr_conf = "#27AE60"; pesan_conf = "Model LSTM telah mempelajari banyak data harian Anda."
    elif 'Sedang' in conf: clr_conf = "#F39C12"; pesan_conf = "Model LSTM cukup bisa diandalkan, namun bisa sedikit meleset karena data masih kurang dari 6 bulan."
    else: clr_conf = "#E74C3C"; pesan_conf = "Prediksi masih kasar. Kumpulkan lebih banyak data transaksi harian agar LSTM bisa bekerja optimal."
    st.markdown(f"""
    <div style="background:white;border-radius:16px;padding:22px 24px;box-shadow:0 3px 14px rgba(27,58,107,0.09);border-left:5px solid #1B3A6B;margin-top:10px;">
        <div style="font-size:0.94rem;font-weight:800;color:#1B3A6B;margin-bottom:16px;">Seberapa Akurat Prediksi LSTM Ini?</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">
            <div style="background:#EEF3FB;border-radius:12px;padding:16px;text-align:center;">
                <div style="width:14px;height:14px;border-radius:50%;background:{clr_conf};margin:0 auto 10px;"></div>
                <div style="font-size:0.72rem;font-weight:700;color:#8090B5;margin-bottom:4px;">Tingkat Kepercayaan</div>
                <div style="font-size:0.92rem;font-weight:900;color:#1B3A6B;">{conf}</div>
            </div>
            <div style="background:#EEF3FB;border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:1.3rem;font-weight:800;color:#1B3A6B;margin-bottom:6px;">{n_bulan * 30}</div>
                <div style="font-size:0.72rem;font-weight:700;color:#8090B5;margin-bottom:4px;">Row Transaksi Dipelajari</div>
                <div style="font-size:0.92rem;font-weight:900;color:#1B3A6B;">Data Harian</div>
            </div>
            <div style="background:#EEF3FB;border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:0.65rem;font-weight:800;color:#1B3A6B;margin-bottom:6px;word-break:break-all;">{model}</div>
                <div style="font-size:0.72rem;font-weight:700;color:#8090B5;margin-bottom:4px;">Metode Prediksi</div>
            </div>
        </div>
        <div style="margin-top:16px;background:#EEF3FB;border-radius:12px;padding:14px 18px;font-size:0.82rem;color:#1A2D5A;line-height:1.75;">
            <strong>Cara membaca grafik di atas:</strong> Garis <em>penuh</em> adalah data nyata, garis <em>putus-putus</em> adalah tren.<br><br>
            <span style="color:{clr_conf};font-weight:700;">{pesan_conf}</span> 
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================

def main():
    # Sembunyikan tombol collapse sidebar (tulisan "double_arrow_right")
    st.markdown("""
    <script>
    (function removeSidebarToggle() {
        function hide() {
            var selectors = [
                '[data-testid="stSidebarCollapsedControl"]',
                '[data-testid="collapsedControl"]'
            ];
            selectors.forEach(function(sel) {
                var els = document.querySelectorAll(sel);
                els.forEach(function(el) { el.style.display = 'none'; });
            });
        }
        hide();
        var obs = new MutationObserver(hide);
        obs.observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """, unsafe_allow_html=True)

    uploaded, tahun_sel, bulan_sel = render_sidebar()
    tgl = datetime.now().strftime('%d %B %Y')
    st.markdown(f"""
    <div class="main-header">
        <div class="header-logo">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <line x1="3" y1="22" x2="21" y2="22"/>
                <line x1="6" y1="18" x2="6" y2="11"/>
                <line x1="10" y1="18" x2="10" y2="11"/>
                <line x1="14" y1="18" x2="14" y2="11"/>
                <line x1="18" y1="18" x2="18" y2="11"/>
                <polygon points="12 2 20 7 4 7"/>
            </svg>
        </div>
        <div class="header-text">
            <h1>Dashboard Keuangan Karang Taruna</h1>
            <p>Pantau keuangan desa dengan mudah, jelas, dan transparan &nbsp;·&nbsp; {tgl}</p>
        </div>
        <div class="header-badge">
            <span class="header-badge-dot"></span>Portal Manajemen Desa
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not uploaded:
        render_welcome()
        st.markdown("""
        <div class="footer-box">
            Dashboard Keuangan Karang Taruna &nbsp;·&nbsp; Transparansi &amp; Kemajuan Bersama
            <span>Upload file Excel di sidebar untuk mulai memantau keuangan Anda</span>
        </div>""", unsafe_allow_html=True)
        return

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']
    else:
        try:
            file_bytes = uploaded.getvalue()
            raw = pd.read_excel(io.BytesIO(file_bytes), header=0)
            if not any(k in ' '.join(raw.columns.str.lower()) for k in ['tanggal','date','tgl']):
                raw = pd.read_excel(io.BytesIO(file_bytes), header=1)
            df = bersihkan_data(raw); st.session_state['df_clean'] = df
        except ValueError as ve:
            st.error(str(ve)); return
        except Exception as e:
            st.error(f"Gagal membaca file: {e}\n\nPastikan ada kolom: **Tanggal | Pemasukan | Pengeluaran**"); return

    if df.empty:
        st.warning("Data kosong atau tidak bisa dibaca. Periksa kembali format file Excel Anda."); return

    tahun_list  = sorted(df['Tahun'].unique(), reverse=True)
    tahun_aktif = tahun_sel if (tahun_sel and tahun_sel in tahun_list) else tahun_list[0]
    monthly     = agregasi_bulanan(df, tahun_aktif)
    if monthly.empty:
        st.warning(f"Tidak ada data untuk tahun {tahun_aktif}."); return

    if bulan_sel != 'Semua Bulan':
        bln_num = {v: k for k, v in MONTH_ID.items()}.get(bulan_sel)
        if bln_num:
            monthly_filtered = monthly[monthly['Bulan'] == bln_num]
            if monthly_filtered.empty:
                st.warning(f"Tidak ada data untuk bulan {bulan_sel}."); monthly_filtered = monthly
        else: monthly_filtered = monthly
    else: monthly_filtered = monthly

    # --- EKSEKUSI LSTM ---
    last_bln = int(monthly['Bulan'].iloc[-1])
    last_sal = float(monthly['SaldoAkhir'].iloc[-1])
    pred = prediksi_lstm(df, last_bln, tahun_aktif, last_sal)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ringkasan", "Grafik", "Prediksi", "Detail", "Tips & Saran"])

    with tab1:
        st.markdown(f'<div class="section-title">Ringkasan Keuangan — Tahun {tahun_aktif}' + (f' &nbsp;·&nbsp; {bulan_sel}' if bulan_sel != 'Semua Bulan' else '') + '</div>', unsafe_allow_html=True)
        render_kpi(monthly_filtered, pred)
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Kondisi Kesehatan Keuangan</div>', unsafe_allow_html=True)
        st.caption("Skor 0–100%  ·  Sangat Sehat (70–100%)  ·  Perlu Perhatian (40–70%)  ·  Kritis (0–40%)")
        render_health(monthly_filtered)

    with tab2:
        st.markdown(f'<div class="section-title">Grafik Keuangan — Tahun {tahun_aktif}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(chart_saldo(monthly_filtered), use_container_width=True)
        with c2: st.plotly_chart(chart_donut(monthly_filtered['Pemasukan'].sum(), monthly_filtered['Pengeluaran'].sum()), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(chart_bulanan(monthly_filtered), use_container_width=True)
        with c4: st.plotly_chart(chart_surplus(monthly_filtered), use_container_width=True)
        fig_kat = chart_kategori(df, tahun_aktif)
        if fig_kat: st.plotly_chart(fig_kat, use_container_width=True)
        if len(tahun_list) > 1:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Perbandingan Tahun ke Tahun</div>', unsafe_allow_html=True)
            fig_yoy = chart_yoy(df, tahun_list[:4])
            if fig_yoy: st.plotly_chart(fig_yoy, use_container_width=True)

    with tab3:
        st.markdown(f'<div class="section-title">Prediksi Keuangan — {pred["bulan_nama"]} {pred["tahun"]} dan Seterusnya</div>', unsafe_allow_html=True)
        render_prediksi(pred, monthly)
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Validasi Akurasi Prediksi</div>', unsafe_allow_html=True)
        st.caption("Seberapa dekat prediksi model dengan data nyata masa lalu.")
        render_akurasi(monthly, pred)

    with tab4:
        st.markdown(f'<div class="section-title">Rekap Keuangan Bulanan — Tahun {tahun_aktif}</div>', unsafe_allow_html=True)
        st.caption("Rincian lengkap pemasukan, pengeluaran, rasio beban, dan saldo setiap bulan")
        render_tabel(monthly, tahun_aktif)
        if bulan_sel != 'Semua Bulan':
            bln_num = {v: k for k, v in MONTH_ID.items()}.get(bulan_sel)
            if bln_num:
                dd = df[(df['Tahun'] == tahun_aktif) & (df['Bulan'] == bln_num)]
                if not dd.empty:
                    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                    st.markdown(f'<div class="section-title">Detail Transaksi: {bulan_sel} {tahun_aktif}</div>', unsafe_allow_html=True)
                    dc = ['Tanggal','Pemasukan','Pengeluaran','Saldo']
                    if 'Keterangan' in dd.columns: dc.append('Keterangan')
                    if 'Kategori'   in dd.columns: dc.append('Kategori')
                    st.dataframe(dd[dc], use_container_width=True, hide_index=True)
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Pola Pemasukan: Awal vs Akhir Bulan</div>', unsafe_allow_html=True)
        st.caption("Apakah pemasukan lebih banyak di awal bulan (tgl 1–15) atau akhir bulan (tgl 16–31)?")
        render_ab(df, tahun_aktif)

    with tab5:
        st.markdown('<div class="section-title">Saran &amp; Tips Keuangan</div>', unsafe_allow_html=True)
        render_tips(monthly, pred)

    st.markdown(f"""
    <div class="footer-box">
        Dashboard Keuangan Karang Taruna &nbsp;·&nbsp; Transparansi &amp; Kemajuan Bersama
        <span>Data Tahun {tahun_aktif} &nbsp;·&nbsp; Diperbarui {datetime.now().strftime('%d %B %Y pukul %H:%M WIB')} &nbsp;·&nbsp; v4.0</span>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()