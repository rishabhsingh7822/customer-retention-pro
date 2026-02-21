import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
import numpy as np
import os
import json
from groq import Groq 
from dotenv import load_dotenv
import streamlit_authenticator as stauth 
load_dotenv()

# ============================================
# 1. PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="RETAINION PLATFORM // Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 1B. THEME STATE
# ============================================
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"

# ============================================
# 2. DESIGN SYSTEM 
# ============================================
if st.session_state.theme == "light":
    css_colors = """
    --bg-primary:    #ffffff;
    --bg-secondary:  #f8f9fa;
    --bg-card:       #ffffff;
    --bg-elevated:   #f1f3f5;
    --accent-green:  #00aa00;
    --accent-amber:  #e67e00;
    --accent-red:    #d63031;
    --accent-blue:   #0066cc;
    --text-primary:  #000000;
    --text-secondary:#2d3436;
    --text-muted:    #636e72;
    --border:        #2d3436;
    --border-bright: #000000;
    """
else:  # Dark Mode
    css_colors = """
    --bg-primary:    #07080d;
    --bg-secondary:  #0d0f18;
    --bg-card:       #0f1119;
    --bg-elevated:   #141720;
    --accent-green:  #00e5a0;
    --accent-amber:  #f5a623;
    --accent-red:    #ff3d57;
    --accent-blue:   #4d9cff;
    --text-primary:  #e2ddd6;
    --text-secondary:#8a8fa8;
    --text-muted:    #4a4f63;
    --border:        #1c1f2e;
    --border-bright: #2a2f45;
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Barlow+Condensed:wght@300;400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

:root {{
    {css_colors}
    --font-display:  'Barlow Condensed', sans-serif;
    --font-mono:     'IBM Plex Mono', monospace;
    --font-body:     'Barlow', sans-serif;
}}

/* Target modern Streamlit classes and all text elements to fix invisible text */
html, body, [class*="css"], [class*="st-"], p, span, h1, h2, h3, h4, h5, h6, li, label {{
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}}

/* Force background colors on all Streamlit containers */
.main, .block-container, [data-testid="stAppViewContainer"], 
[data-testid="stHeader"], .stApp {{
    background-color: var(--bg-primary) !important;
}}

/* Column backgrounds */
[data-testid="column"] {{
    background-color: var(--bg-primary) !important;
}}

/* Ensure all input containers have proper background */
[data-testid="stVerticalBlock"] > div {{
    background-color: transparent !important;
}}

/* Form elements background */
[data-testid="stForm"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    padding: 1rem !important;
    border-radius: 4px !important;
}}

#MainMenu, footer {{ visibility: hidden; }}
header {{ background: transparent !important; }}
.block-container {{ padding: 1.5rem 2rem !important; max-width: 100% !important; }}

[data-testid="stSidebar"] {{ background: var(--bg-secondary) !important; border-right: 2px solid var(--border) !important; }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 1.5rem; }}

/* Header area where toggle button lives */
header[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* Ensure buttons are always visible and above other elements */
[data-testid="stSidebarNav"] {{
    z-index: 999998 !important;
}}

.sidebar-brand {{
    font-family: var(--font-display);
    font-size: 1.4rem; font-weight: 800; letter-spacing: 0.15em;
    color: var(--accent-green); text-transform: uppercase;
    padding: 0 1rem 1rem; border-bottom: 2px solid var(--border); margin-bottom: 1.5rem;
}}
.sidebar-brand span {{ color: var(--text-secondary); font-weight: 300; }}

.visibility-selector {{
    background: var(--bg-card);
    border: 2px solid var(--border-bright);
    padding: 0.8rem;
    margin: 0 1rem 1.5rem;
    border-radius: 6px;
}}
.visibility-label {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    display: block;
    margin-bottom: 0.5rem;
}}

[data-testid="stRadio"] label {{
    font-family: var(--font-mono) !important; font-size: 0.75rem !important;
    letter-spacing: 0.08em !important; color: var(--text-primary) !important; 
    font-weight: 500 !important; text-transform: uppercase !important;
}}
[data-testid="stRadio"] div[data-baseweb="radio"] div {{ background: transparent !important; border: none !important; }}

.stat-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; padding: 1rem; border-top: 2px solid var(--border); margin-top: 1rem; }}
.stat-pill {{ background: var(--bg-card); border: 2px solid var(--border); padding: 8px 10px; border-radius: 4px; }}
.stat-pill .label {{ font-family: var(--font-mono); font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; display: block; margin-bottom: 2px; }}
.stat-pill .value {{ font-family: var(--font-mono); font-size: 0.85rem; font-weight: 600; color: var(--accent-green); }}

.page-header {{ display: flex; align-items: baseline; gap: 1rem; margin-bottom: 0.25rem; }}
.page-title {{ font-family: var(--font-display); font-size: 2.8rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-primary); line-height: 1; }}
.page-tag {{ font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent-green); letter-spacing: 0.15em; text-transform: uppercase; border: 2px solid var(--accent-green); padding: 3px 8px; border-radius: 2px; font-weight: 600; }}
.page-subtitle {{ font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted); letter-spacing: 0.08em; margin-bottom: 1.5rem; text-transform: uppercase; }}

.kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; background: var(--border-bright); border: 2px solid var(--border-bright); margin-bottom: 1.5rem; }}
.kpi-card {{ background: var(--bg-card); padding: 1.2rem 1.4rem; position: relative; overflow: hidden; }}
.kpi-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--accent-color, var(--accent-green)); }}
.kpi-label {{ font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; display: block; }}
.kpi-value {{ font-family: var(--font-mono); font-size: 1.9rem; font-weight: 700; color: var(--text-primary); line-height: 1; display: block; }}
.kpi-sub {{ font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted); margin-top: 4px; display: block; }}

.section-label {{ font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.2em; padding: 0.8rem 0 0.4rem; border-top: 2px solid var(--border); margin: 1rem 0 0.8rem; display: flex; align-items: center; gap: 8px; font-weight: 600; }}
.section-label::after {{ content: ''; flex: 1; height: 2px; background: var(--border); }}

[data-testid="stDataFrame"] {{ border: 2px solid var(--border-bright) !important; }}
[data-testid="stDataFrame"] th {{ background: var(--bg-secondary) !important; font-family: var(--font-mono) !important; font-size: 0.65rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; color: var(--text-primary) !important; font-weight: 600 !important; }}
[data-testid="stDataFrame"] td {{ font-family: var(--font-mono) !important; font-size: 0.78rem !important; color: var(--text-primary) !important; }}

[data-testid="stButton"] > button {{ font-family: var(--font-mono) !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: transparent !important; color: var(--accent-green) !important; border: 2px solid var(--accent-green) !important; border-radius: 2px !important; padding: 0.5rem 1.2rem !important; transition: all 0.15s ease !important; }}
[data-testid="stButton"] > button:hover {{ background: var(--accent-green) !important; color: var(--bg-primary) !important; }}
[data-testid="stButton"] > button[kind="primary"] {{ background: var(--accent-green) !important; color: var(--bg-primary) !important; font-weight: 700 !important; }}
[data-testid="stButton"] > button[kind="primary"]:hover {{ opacity: 0.85 !important; }}

[data-testid="stSlider"] label {{ font-family: var(--font-mono) !important; font-size: 0.7rem !important; color: var(--text-primary) !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; font-weight: 600 !important; }}
.stSlider > div > div > div > div {{ background: var(--accent-green) !important; }}

[data-testid="stNumberInput"] label, [data-testid="stSelectbox"] label {{ font-family: var(--font-mono) !important; font-size: 0.7rem !important; color: var(--text-primary) !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; font-weight: 600 !important; }}

/* Input boxes background */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-primary) !important;
}}

/* Select box background */
[data-baseweb="select"] > div {{
    background-color: var(--bg-card) !important;
    border-color: var(--border-bright) !important;
    color: var(--text-primary) !important;
}}

.ai-stream-box {{ background: var(--bg-secondary); border: 2px solid var(--border-bright); border-left: 4px solid var(--accent-green); padding: 1.2rem 1.4rem; font-family: var(--font-body); font-size: 0.88rem; line-height: 1.65; color: var(--text-primary); border-radius: 0 4px 4px 0; min-height: 80px; }}

[data-testid="stChatMessage"] {{ background: var(--bg-card) !important; border: 2px solid var(--border) !important; border-radius: 4px !important; font-family: var(--font-body) !important; }}
/* Fix Chat Input Background and Blinking Cursor */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="base-input"] {{
    background-color: var(--bg-card) !important;
}}
[data-testid="stChatInput"] textarea {{ 
    font-family: var(--font-mono) !important; 
    font-size: 0.82rem !important; 
    background-color: var(--bg-card) !important; 
    color: var(--text-primary) !important; 
    border: 2px solid var(--border-bright) !important; 
    caret-color: var(--accent-green) !important; /* Forces cursor to be visible and green */
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: var(--text-muted) !important; /* Ensures placeholder text is legible */
}}

/* Fix AI Markdown Inline Code (Numbers and Highlights) */
p code, li code, .ai-stream-box code, [data-testid="stChatMessage"] code {{
    background-color: var(--bg-elevated) !important;
    color: var(--accent-green) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.9em !important;
}}

.badge {{ display: inline-block; font-family: var(--font-mono); font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; padding: 4px 9px; border-radius: 3px; }}
.badge-high    {{ background: rgba(214,48,49,0.2);  color: var(--accent-red);   border: 2px solid var(--accent-red);  }}
.badge-medium {{ background: rgba(230,126,0,0.2); color: var(--accent-amber); border: 2px solid var(--accent-amber); }}
.badge-low    {{ background: rgba(0,170,0,0.15);   color: var(--accent-green); border: 2px solid var(--accent-green); }}

.shap-row {{ display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 2px solid var(--border); font-family: var(--font-mono); font-size: 0.72rem; }}
.shap-feature {{ color: var(--text-primary); flex: 1; font-weight: 500; }}
.shap-value   {{ color: var(--text-primary); font-weight: 700; width: 60px; text-align: right; }}
.shap-bar-wrap {{ width: 80px; background: var(--bg-elevated); height: 6px; border-radius: 3px; border: 1px solid var(--border); }}
.shap-bar-pos  {{ height: 100%; border-radius: 3px; background: var(--accent-red); }}
.shap-bar-neg  {{ height: 100%; border-radius: 3px; background: var(--accent-green); }}

[data-testid="stExpander"] {{ background: var(--bg-card) !important; border: 2px solid var(--border) !important; border-radius: 4px !important; }}
[data-testid="stExpander"] summary {{ font-family: var(--font-mono) !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; color: var(--text-primary) !important; font-weight: 600 !important; }}

[data-baseweb="select"] {{ background: var(--bg-card) !important; border: 2px solid var(--border-bright) !important; font-family: var(--font-mono) !important; font-size: 0.78rem !important; color: var(--text-primary) !important; }}

[data-testid="stInfo"] {{ background: rgba(77,156,255,0.1) !important; border: 2px solid var(--accent-blue) !important; border-left: 4px solid var(--accent-blue) !important; font-family: var(--font-body) !important; font-size: 0.84rem !important; border-radius: 0 4px 4px 0 !important; }}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-secondary); }}
::-webkit-scrollbar-thumb {{ background: var(--border-bright); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}

/* ============================================
   MOBILE RESPONSIVE STYLES
   ============================================ */
@media screen and (max-width: 768px) {{
    /* Container adjustments */
    .block-container {{ 
        padding: 1rem 0.75rem !important; 
    }}
    
    /* Header scaling */
    .page-title {{ 
        font-size: 1.8rem !important; 
        line-height: 1.1 !important;
    }}
    .page-tag {{ 
        font-size: 0.6rem !important; 
        padding: 2px 6px !important;
    }}
    .page-subtitle {{ 
        font-size: 0.65rem !important; 
        margin-bottom: 1rem !important;
    }}
    
    /* KPI Grid - 2 columns on mobile */
    .kpi-grid {{ 
        grid-template-columns: repeat(2, 1fr) !important; 
        gap: 1px !important;
    }}
    .kpi-card {{ 
        padding: 1rem 1rem !important; 
    }}
    .kpi-label {{ 
        font-size: 0.58rem !important; 
        margin-bottom: 0.4rem !important;
    }}
    .kpi-value {{ 
        font-size: 1.5rem !important; 
    }}
    .kpi-sub {{ 
        font-size: 0.58rem !important; 
    }}
    
    /* Sidebar brand */
    .sidebar-brand {{ 
        font-size: 1.1rem !important; 
        padding: 0 0.75rem 0.75rem !important;
    }}
    
    /* Visibility selector */
    .visibility-selector {{ 
        padding: 0.6rem !important; 
        margin: 0 0.75rem 1rem !important;
    }}
    .visibility-label {{ 
        font-size: 0.6rem !important; 
    }}
    
    /* Radio buttons - larger touch targets */
    [data-testid="stRadio"] label {{ 
        font-size: 0.7rem !important; 
        padding: 0.5rem 0 !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
    }}
    
    /* Stat pills in sidebar - 1 column */
    .stat-row {{ 
        grid-template-columns: 1fr !important; 
        gap: 4px !important;
        padding: 0.75rem !important;
    }}
    .stat-pill {{ 
        padding: 10px 12px !important; 
    }}
    .stat-pill .label {{ 
        font-size: 0.58rem !important; 
    }}
    .stat-pill .value {{ 
        font-size: 0.95rem !important; 
    }}
    
    /* Section labels */
    .section-label {{ 
        font-size: 0.6rem !important; 
        padding: 0.6rem 0 0.3rem !important;
        margin: 0.75rem 0 0.6rem !important;
    }}
    
    /* Buttons - larger touch targets */
    [data-testid="stButton"] > button {{ 
        font-size: 0.68rem !important; 
        padding: 0.65rem 1rem !important;
        min-height: 44px !important;
    }}
    
    /* Forms and inputs - larger touch targets */
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] select,
    [data-testid="stSlider"] {{ 
        min-height: 44px !important;
        font-size: 0.85rem !important;
    }}
    
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label {{ 
        font-size: 0.68rem !important; 
    }}
    
    /* Tables */
    [data-testid="stDataFrame"] th {{ 
        font-size: 0.6rem !important; 
        padding: 8px 4px !important;
    }}
    [data-testid="stDataFrame"] td {{ 
        font-size: 0.72rem !important; 
        padding: 8px 4px !important;
    }}
    
    /* AI stream box */
    .ai-stream-box {{ 
        padding: 1rem 1rem !important; 
        font-size: 0.82rem !important;
        line-height: 1.6 !important;
    }}
    
    /* Chat interface */
    [data-testid="stChatMessage"] {{ 
        padding: 0.75rem !important; 
        font-size: 0.82rem !important;
    }}
    [data-testid="stChatInput"] textarea {{ 
        font-size: 0.85rem !important; 
        min-height: 44px !important;
    }}
    
    /* Badges */
    .badge {{ 
        font-size: 0.58rem !important; 
        padding: 4px 8px !important;
    }}
    
    /* SHAP visualization */
    .shap-row {{ 
        font-size: 0.68rem !important; 
        padding: 8px 0 !important;
        flex-wrap: wrap !important;
    }}
    .shap-feature {{ 
        flex: 0 0 100% !important;
        margin-bottom: 4px !important;
    }}
    .shap-value {{ 
        width: 50px !important; 
    }}
    .shap-bar-wrap {{ 
        width: 60px !important; 
        height: 8px !important;
    }}
    
    /* Expanders */
    [data-testid="stExpander"] summary {{ 
        font-size: 0.68rem !important; 
        padding: 0.75rem !important;
        min-height: 44px !important;
    }}
}}

/* Tablet adjustments (768px - 1024px) */
@media screen and (min-width: 769px) and (max-width: 1024px) {{
    .page-title {{ 
        font-size: 2.2rem !important; 
    }}
    
    /* KPI Grid - 2 columns on tablet */
    .kpi-grid {{ 
        grid-template-columns: repeat(2, 1fr) !important; 
    }}
    
    .kpi-card {{ 
        padding: 1.1rem 1.2rem !important; 
    }}
    
    .kpi-value {{ 
        font-size: 1.7rem !important; 
    }}
}}
/* =========================================
       PROTECT UI ICONS & DATAFRAME ARROWS
       ========================================= */
    /* 1. Restore the original font for all Streamlit UI icons */
    .material-symbols-rounded, .material-icons, [class*="icon"], [class*="stIcon"] {{
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }}
    
    /* 2. Ensure DataFrame sorting arrows and UI SVGs don't disappear into the background */
    svg {{
        fill: var(--text-primary) !important;
    }}
    
    /* 3. Make DataFrame headers specifically pop in all modes */
    [data-testid="stDataFrame"] th {{
        border-bottom: 2px solid var(--accent-green) !important;
    }}
    
    /* 4. Ensure Dropdown/Selectbox arrows are visible */
    [data-baseweb="select"] svg {{
        fill: var(--text-primary) !important;
    }}
    /* =========================================
       FIX: SIDEBAR TOGGLE & VISIBILITY (V7 MAX-SPECIFICITY)
       ========================================= */
    header {{ 
        visibility: visible !important; 
        background: transparent !important;
        z-index: 99999 !important; 
    }}
    #MainMenu, footer {{ 
        visibility: hidden !important; 
    }}
    
    /* 1. Lock the hitbox size and physically throw native text off-screen */
    html body div[data-testid="stSidebarCollapseButton"],
    html body div[data-testid="collapsedControl"] {{
        position: relative !important;
        width: 3rem !important;
        height: 3rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: transparent !important;
        border: none !important;
        color: transparent !important;
        text-indent: -9999px !important; /* Throw text off screen */
        overflow: hidden !important;
    }}
    
    /* 2. MAX SPECIFICITY: Target ONLY the visual spans/icons and nuke them. 
          We avoid '*' so we don't accidentally delete Streamlit's click listener */
    html body div[data-testid="stSidebarCollapseButton"] span,
    html body div[data-testid="stSidebarCollapseButton"] svg,
    html body div[data-testid="stSidebarCollapseButton"] i,
    html body div[data-testid="collapsedControl"] span,
    html body div[data-testid="collapsedControl"] svg,
    html body div[data-testid="collapsedControl"] i {{
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }}
    
    /* 3. Inject the UNHIDE ICON (Sleek right arrow) */
    html body div[data-testid="collapsedControl"]::after {{
        content: "❯" !important; 
        font-size: 1.6rem !important;
        color: var(--text-primary) !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        pointer-events: none !important; 
        text-indent: 0 !important; /* Pull the icon back on-screen */
        visibility: visible !important;
    }}
    
    /* 4. Inject X to HIDE */
    html body div[data-testid="stSidebarCollapseButton"]::after {{
        content: "✖" !important; 
        font-size: 1.2rem !important; 
        color: var(--text-muted) !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        pointer-events: none !important; 
        text-indent: 0 !important; /* Pull the icon back on-screen */
        visibility: visible !important;
    }}
    
    html body div[data-testid="stSidebarCollapseButton"]:hover::after {{
        color: var(--accent-red) !important;
    }}
    /* =========================================
       ULTIMATE DATAFRAME MENU & POPOVER OVERLAP FIX
       ========================================= */
    /* Force the popover container background */
    div[data-baseweb="popover"] {{
        background-color: var(--bg-card) !important;
    }}

    ul[data-baseweb="menu"] {{
        background-color: var(--bg-card) !important;
    }}

    /* Force menu items to expand their height automatically and align content */
    ul[data-baseweb="menu"] li {{
        height: auto !important;
        min-height: 38px !important; 
        padding: 8px 16px !important;
        display: flex !important;
        align-items: center !important;
        flex-direction: row !important;
    }}

    /* Strictly forbid text from wrapping or stacking */
    ul[data-baseweb="menu"] li span,
    ul[data-baseweb="menu"] li div,
    ul[data-baseweb="menu"] li p {{
        white-space: nowrap !important;
        line-height: normal !important;
        height: auto !important;
        margin: 0 !important;
        display: inline-block !important;
    }}
    
    /* Ensure the sorting/filtering icons have proper space and don't shrink */
    ul[data-baseweb="menu"] li svg {{
        flex-shrink: 0 !important;
        margin-right: 12px !important;
        display: inline-block !important;
    }}
    /* =========================================
       FILE UPLOADER LIGHT/DARK MODE (FINAL)
       ========================================= */
    /* 1. Target the actual HTML section of the uploader */
    [data-testid="stFileUploader"] section {{
        background-color: var(--bg-card) !important;
        border: 2px dashed var(--border-bright) !important; 
        border-radius: 6px !important;
    }}
    
    /* 2. Force inner wrappers to be transparent so the card background shows through */
    [data-testid="stFileUploader"] section div {{
        background-color: transparent !important;
    }}
    
    /* 3. Make the text and small instructions match your theme */
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {{
        color: var(--text-primary) !important;
    }}
    
    /* 4. Fix the cloud icon so it has no black background */
    [data-testid="stFileUploader"] section svg {{
        fill: var(--text-primary) !important;
        background-color: transparent !important;
    }}
    
    /* 5. Style the Browse Files button */
    [data-testid="stFileUploader"] button {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-bright) !important;
        color: var(--text-primary) !important;
    }}
    
    [data-testid="stFileUploader"] button:hover {{
        border-color: var(--accent-green) !important;
        color: var(--accent-green) !important;
    }}
</style>
""", unsafe_allow_html=True)


# 2B. SECURE AUTHENTICATION
# ============================================
import streamlit_authenticator as stauth
import time

# 1. Initialize the tracker and timer in memory
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0
if 'lockout_time' not in st.session_state:
    st.session_state.lockout_time = 0

DEV_PASSWORD = "Rishabh_DevKey_2026"

# 2. Set up the Authenticator configuration
hashed_pass = stauth.Hasher.hash('Retainion123')

credentials = {
    "usernames": {
        "Rishabh_admin": {
            "email": "rishabh@retainion.com",
            "name": "Rishabh singh",
            "password": hashed_pass
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "retainion_dashboard",
    "auth_cookie_key",
    0 
)

# ============================================
# THE VISUAL LOGIC SWITCH
# ============================================
if st.session_state.get("authentication_status"):
    # IF LOGGED IN: Reset attempts and skip drawing the login UI entirely!
    st.session_state.login_attempts = 0
    
else:
    # IF NOT LOGGED IN: Draw the branded login screen
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, text_col, _ = st.columns([1, 2, 1])

    with text_col:
        st.markdown("""
        <div style="text-align:center;margin-bottom:2rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                        letter-spacing:0.28em;color:#00e5a0;text-transform:uppercase;
                        margin-bottom:0.9rem;">
                ◈ &nbsp; Intelligence Platform &nbsp; ◈
            </div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:4rem;
                        font-weight:800;letter-spacing:0.08em;color:#e2ddd6;
                        text-transform:uppercase;line-height:0.95;">
                RE<span style="color:#00e5a0;">TAINION</span>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                        color:#4a4f63;letter-spacing:0.18em;text-transform:uppercase;
                        margin-top:0.6rem;">
                Authorised Access Only — Credentials Required
            </div>
            <div style="width:48px;height:3px;background:#00e5a0;margin:1.2rem auto 0;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Handle the Lockout Timer UI
        if st.session_state.login_attempts >= 3:
            elapsed_time = time.time() - st.session_state.lockout_time
            remaining_time = int(120 - elapsed_time)
            
            if remaining_time > 0:
                import streamlit.components.v1 as components
                
                _, lock_col, _ = st.columns([1, 1.5, 1])
                with lock_col:
                    st.error("🔒 SYSTEM LOCKED: Too many failed attempts.")
                    
                    # 1. Automatic Live Countdown (Powered by JavaScript to prevent typing interruptions)
                    timer_html = f"""
                    <style>
                        body {{ margin: 0; font-family: sans-serif; background-color: transparent; }}
                        .warning-box {{
                            border-left: 4px solid #f5a623;
                            background-color: rgba(245, 166, 35, 0.1);
                            color: #8a8fa8; 
                            padding: 0.8rem 1rem;
                            border-radius: 4px;
                            font-size: 0.9rem;
                        }}
                        .highlight {{ color: #f5a623; font-family: monospace; font-size: 1.05rem; font-weight: bold; }}
                    </style>
                    <div class="warning-box">
                        Please wait <span class="highlight" id="time">{remaining_time}</span><span class="highlight"> seconds</span> before trying again.
                    </div>
                    <script>
                        let timeLeft = {remaining_time};
                        const timerEl = document.getElementById('time');
                        const countdown = setInterval(() => {{
                            timeLeft--;
                            if(timerEl) timerEl.innerText = timeLeft;
                            if(timeLeft <= 0) {{
                                clearInterval(countdown);
                                window.parent.location.reload(); // Instantly reloads the page to unlock
                            }}
                        }}, 1000);
                    </script>
                    """
                    components.html(timer_html, height=70)
                    
                    # 2. Developer Key Input 
                    dev_input = st.text_input("Developer Key", type="password")
                    
                    # 3. Removed the "Refresh Timer" column entirely!
                    if st.button("Unlock System", use_container_width=True):
                        if dev_input == DEV_PASSWORD:
                            st.session_state.login_attempts = 0
                            st.session_state.lockout_time = 0
                            st.session_state['authentication_status'] = None
                            st.success("System unlocked! Refreshing...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Access Denied: Invalid Developer Key.")
                st.stop()
            else:
                st.session_state.login_attempts = 0
                st.session_state.lockout_time = 0
                st.session_state['authentication_status'] = None

        # ── LOGIN FORM ────────────────────────────────────────────────────────
        _, form_col, _ = st.columns([1, 1.5, 1])
        with form_col:
            st.markdown("""
            <style>
            /* ── Card wrapper ── */
            div[data-testid="stVerticalBlock"]:has(> div > [data-testid="stForm"]) {
                background: #0f1119;
                border: 1px solid #1c1f2e;
                border-top: 3px solid #00e5a0;
                padding: 2rem 2rem 1.5rem;
                border-radius: 2px;
            }

            /* ── Input labels ── */
            [data-testid="stForm"] [data-testid="stTextInput"] label {
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 0.62rem !important;
                color: #4a4f63 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.2em !important;
                font-weight: 600 !important;
            }

            /* ── Input boxes ── */
            [data-testid="stForm"] [data-testid="stTextInput"] input {
                background: #07080d !important;
                border: 1px solid #2a2f45 !important;
                border-bottom: 2px solid #00e5a0 !important;
                border-radius: 0 !important;
                color: #e2ddd6 !important;
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 0.88rem !important;
                padding: 0.65rem 0.9rem !important;
            }
            [data-testid="stForm"] [data-testid="stTextInput"] input:focus {
                border-color: #00e5a0 !important;
                box-shadow: 0 0 0 1px rgba(0,229,160,0.2) !important;
            }

            /* ── Submit button ── */
            [data-testid="stFormSubmitButton"] { margin-top: 0.8rem; }
            [data-testid="stFormSubmitButton"] button {
                width: 100% !important;
                background: #00e5a0 !important;
                color: #07080d !important;
                border: none !important;
                border-radius: 0 !important;
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 0.75rem !important;
                font-weight: 700 !important;
                letter-spacing: 0.2em !important;
                text-transform: uppercase !important;
                padding: 0.8rem !important;
            }
            [data-testid="stFormSubmitButton"] button:hover {
                opacity: 0.85 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # Attempt indicator
            used = st.session_state.login_attempts
            left = 3 - used
            dots = "".join([
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;background:{"#ff3d57" if i < used else "#1c1f2e"};border:1px solid {"#ff3d57" if i < used else "#2a2f45"};"></span>'
                for i in range(3)
            ])
            st.markdown(f"""
            <div style="margin-bottom:1.2rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                            color:#4a4f63;letter-spacing:0.18em;text-transform:uppercase;
                            margin-bottom:8px;">Security Clearance</div>
                <div>{dots}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                            color:#4a4f63;margin-top:5px;letter-spacing:0.1em;">
                    {left} of 3 attempts remaining
                </div>
            </div>
            """, unsafe_allow_html=True)

            authenticator.login("main")

            # ── CRITICAL BUG FIX: rerun immediately on success so the
            #    dashboard renders cleanly without the login form above it ──
            auth_status = st.session_state.get("authentication_status")

            if auth_status == True:
                st.session_state.login_attempts = 0
                st.rerun()                          # ← this was missing; caused double-click & bleed-through

            elif auth_status == False:
                st.session_state.login_attempts += 1
                if st.session_state.login_attempts >= 3:
                    st.session_state.lockout_time = time.time()
                    st.rerun()
                else:
                    attempts_left = 3 - st.session_state.login_attempts
                    st.markdown(f"""
                    <div style="background:rgba(255,61,87,0.08);border:1px solid #ff3d57;
                                border-left:3px solid #ff3d57;padding:0.75rem 1rem;
                                font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                                color:#ff3d57;letter-spacing:0.05em;margin-top:0.5rem;">
                        ✖ &nbsp; Invalid credentials — {attempts_left} attempt{'s' if attempts_left != 1 else ''} remaining
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()

            else:  # None — waiting for input
                st.stop()


# 3. LOAD ARTIFACTS
@st.cache_resource
def load_artifacts():
    if os.path.exists('models'):
        model_dir = 'models'
    elif os.path.exists('../models'):
        model_dir = '../models'
    else:
        st.error("Critical Error: 'models' folder not found.")
        st.stop()

    model       = joblib.load(os.path.join(model_dir, 'xgboost_churn.pkl'))
    customer_db = pd.read_csv(os.path.join(model_dir, 'customer_database.csv'))

    with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)

    try:
        explainer = joblib.load(os.path.join(model_dir, 'shap_explainer.pkl'))
    except:
        explainer = None

    feature_names = metadata['feature_names']
    X_all = customer_db[feature_names].fillna(0)
    all_probs = model.predict_proba(X_all)[:, 1]
    
    customer_db['churn_probability'] = all_probs
    customer_db['risk_level'] = pd.cut(
        all_probs,
        bins=[0, 0.4, 0.7, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    customer_db_sorted = customer_db.sort_values('churn_probability', ascending=False)

    return model, customer_db, customer_db_sorted, metadata, explainer

# Instantly unpack the pre-calculated data from memory on every page switch
model, customer_db, customer_db_sorted, metadata, explainer = load_artifacts()
feature_names = metadata['feature_names']


# ============================================
# 4. AI CONFIGURATION (GROQ)
# ============================================
def configure_ai():
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        return Groq(api_key=api_key)
    except Exception:
        return None

# Initialize AI Client
client = configure_ai()
ai_active = client is not None

# ============================================
# 4B. CHART COLOR HELPER
# ============================================
def get_chart_colors():
    """Returns appropriate chart colors based on current theme"""
    if st.session_state.theme == "light":
        return {
            'paper_bgcolor': '#ffffff',
            'plot_bgcolor': '#f8f9fa',
            'text_color': '#111827',     
            'grid_color': '#d1d5db',     
            'secondary_text': '#4b5563'  
        }
    else:  # Dark Mode
        return {
            'paper_bgcolor': '#07080d',
            'plot_bgcolor': '#0d0f18',
            'text_color': '#b4b9c7',     
            'grid_color': '#10121a',     
            'secondary_text': '#6b7086'  
        }

def get_chart_height(default_height=400):
    """Returns responsive chart height - smaller for mobile"""
    is_mobile = st.query_params.get("mobile", "false") == "true"

    return default_height

# Add mobile detection script
st.markdown("""
""", unsafe_allow_html=True)

# ============================================
# 5. AI HELPERS
# ============================================
def stream_to_placeholder(prompt: str, placeholder, system: str = ""):
    if not ai_active:
        placeholder.info(
            "ℹ️ AI features require a Groq API key. "
            "Add GROQ_API_KEY to .env file to enable AI analysis."
        )
        return

    full_text = ""
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Used Llama 3 on Groq
        stream = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            stream=True
        )
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_text += content
                placeholder.markdown(
                    f'<div class="ai-stream-box">{full_text}▌</div>',
                    unsafe_allow_html=True
                )
        
        placeholder.markdown(
            f'<div class="ai-stream-box">{full_text}</div>',
            unsafe_allow_html=True
        )
        return full_text
    except Exception as e:
        placeholder.error(f"AI Error: {str(e)}")
        return ""


def get_shap_factors(customer_dict: dict):
    if not explainer:
        return []
    try:
        X    = pd.DataFrame([customer_dict])[feature_names].fillna(0)
        vals = explainer.shap_values(X)[0]
        return sorted([
            {"feature": f, "value": float(v), "impact": float(s),
             "effect": "increases" if s > 0 else "decreases"}
            for f, v, s in zip(feature_names, X.values[0], vals)
        ], key=lambda x: abs(x["impact"]), reverse=True)[:5]
    except:
        return []


def render_shap_factors(factors: list):
    if not factors:
        return
    max_impact = max(abs(f["impact"]) for f in factors) or 1
    html = '<div class="section-label">Top Churn Drivers</div>'
    for f in factors:
        pct    = abs(f["impact"]) / max_impact * 100
        is_pos = f["impact"] > 0
        bar    = f'<div class="shap-bar-pos" style="width:{pct:.0f}%"></div>' if is_pos else \
                 f'<div class="shap-bar-neg" style="width:{pct:.0f}%"></div>'
        color  = "var(--accent-red)" if is_pos else "var(--accent-green)"
        arrow  = "▲" if is_pos else "▼"
        html += f"""
        <div class="shap-row">
            <span class="shap-feature">{f['feature']}</span>
            <span class="shap-value">{f['value']:.2f}</span>
            <div class="shap-bar-wrap">{bar}</div>
            <span style="color:{color};font-family:var(--font-mono);font-size:0.65rem">{arrow}</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_kpis(items: list):
    """items = list of (label, value, sub, accent_color)"""
    cols_css = " ".join(["1fr"] * len(items))
    html = f'<div class="kpi-grid" style="grid-template-columns:{cols_css}">'
    for label, value, sub, color in items:
        html += f"""
        <div class="kpi-card" style="--accent-color:{color}">
            <span class="kpi-label">{label}</span>
            <span class="kpi-value">{value}</span>
            <span class="kpi-sub">{sub}</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ============================================
# 6. SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        CHURN ANALYSIS DASHBOARD <span>// AI</span>
    </div>
    """, unsafe_allow_html=True)

    # Fetch the logged-in user's name from memory
    current_user_name = st.session_state.get("name", "User")

    # Secure Logout Button
    authenticator.logout("Logout", "sidebar")
    st.markdown(f"<span style='color:var(--text-muted); font-size: 0.8rem;'>Logged in as: **{current_user_name}**</span>", unsafe_allow_html=True)

# Theme Toggle Switch
    def update_theme():
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

    st.toggle("Light Mode", value=(st.session_state.theme == "light"), on_change=update_theme)
    
    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["1. Live Predictor",
         "2. AI Chat Analyst",
         "3. Risk Overview",
         "4. At-Risk Customers",
         "5. Executive Brief",
         "6. Batch Scorer"],
        label_visibility="collapsed"
    )

    high_risk_count = int((customer_db['risk_level'] == 'HIGH').sum())
    revenue_risk    = customer_db[customer_db['risk_level'] == 'HIGH']['Monetary'].sum()

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">
            <span class="label">Customers</span>
            <span class="value">{len(customer_db):,}</span>
        </div>
        <div class="stat-pill">
            <span class="label">High Risk</span>
            <span class="value" style="color:var(--accent-red)">{high_risk_count:,}</span>
        </div>
        <div class="stat-pill">
            <span class="label">Accuracy</span>
            <span class="value">{metadata['accuracy']:.1%}</span>
        </div>
        <div class="stat-pill">
            <span class="label">ROC-AUC</span>
            <span class="value">{metadata['roc_auc']:.3f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI Status indicator
    ai_status_color = "#00e5a0" if ai_active else "#8a8fa8"
    ai_status_text = "Active" if ai_active else "Disabled"
    
    st.markdown(f"""
    <div style="padding:1rem;font-family:var(--font-mono);font-size:0.6rem;
                color:var(--text-muted);border-top:1px solid var(--border);margin-top:1rem">
        XGBoost · {len(feature_names)} features<br>
        AI: <span style="color:{ai_status_color}">{ai_status_text}</span><br>
        Customer Retention Intelligence v2.0
    </div>
    """, unsafe_allow_html=True)


# ============================================
# PAGE 1 — LIVE PREDICTOR
# ============================================
if page == "1. Live Predictor":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Live Predictor</span>
        <span class="page-tag">Real-time</span>
    </div>
    <div class="page-subtitle">Adjust profile → instant churn risk + AI retention strategy</div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

        recency           = st.slider("Recency (days)",         0,   365, 30)
        frequency         = st.slider("Total Orders",           1,    100,  5)
        monetary          = st.number_input("Total Spent ($)",  0, 100000, 500)
        avg_order_value   = st.slider("Avg Order Value ($)",    0,  10000, 100)
        customer_age      = st.slider("Customer Age (days)",    0,  1825, 180)
        product_diversity = st.slider("Unique Products",        1,   100, 10)
        orders_last_30d   = st.slider("Orders Last 30d",        0,    20,  2)
        orders_last_90d   = st.slider("Orders Last 90d",        0,    50,  5)
        is_weekend        = st.slider("Weekend Purchases (%)",  0,   100, 30) / 100
        spending_trend    = st.selectbox("Spending Trend", ["Increasing", "Decreasing"])

    # Build feature vector
    profile = {
        'Recency':                recency,
        'Frequency':              frequency,
        'Monetary':               monetary,
        'customer_age_days':      customer_age,
        'days_between_purchases': customer_age / max(frequency, 1),
        'orders_last_30d':        orders_last_30d,
        'orders_last_90d':        orders_last_90d,
        'avg_order_value':        avg_order_value,
        'max_order_value':        avg_order_value * 1.5,
        'min_order_value':        avg_order_value * 0.5,
        'std_order_value':        avg_order_value * 0.3,
        'product_diversity':      product_diversity,
        'repeat_purchase_rate':   0.5,
        'avg_basket_size':        frequency * 2,
        'total_items_bought':     frequency * 10,
        'is_weekend':             is_weekend,
        'hour':                   14,
        'spending_increasing':    1 if spending_trend == "Increasing" else 0
    }

    input_df   = pd.DataFrame([profile])[feature_names]
    churn_prob = float(model.predict_proba(input_df)[0][1])

    risk_color = (
        "var(--accent-red)"   if churn_prob > 0.7 else
        "var(--accent-amber)" if churn_prob > 0.4 else
        "var(--accent-green)"
    )
    risk_label = (
        "HIGH RISK"   if churn_prob > 0.7 else
        "MEDIUM RISK" if churn_prob > 0.4 else
        "LOW RISK"
    )
    risk_badge = (
        "badge-high"   if churn_prob > 0.7 else
        "badge-medium" if churn_prob > 0.4 else
        "badge-low"
    )

    with col_right:
        # KPIs
        render_kpis([
            ("Churn Probability", f"{churn_prob:.1%}", risk_label,         risk_color),
            ("Customer LTV",      f"${monetary:,}",   "total historical",  "var(--accent-blue)"),
            ("Total Orders",      str(frequency),       "all time",          "var(--accent-amber)"),
            ("Days Inactive",     str(recency),         "since last order",  "var(--text-secondary)"),
        ])

        # 3D Chart
        st.markdown('<div class="section-label">Vector Space</div>', unsafe_allow_html=True)
        
        # Get dynamic colors
        chart_colors = get_chart_colors()
        
        fig = go.Figure()
        
        # Trace 0 -> Renamed to "Current Customer"
        fig.add_trace(go.Scatter3d(
            name='Current Customer',  # <--- ADDED PROPER NAME
            x=[recency], y=[frequency], z=[monetary],
            mode='markers',
            hovertemplate=(
                "<b>Customer</b><br>"
                "Recency: %{x}d<br>Frequency: %{y}<br>LTV: $%{z}<extra></extra>"
            ),
            marker=dict(
                size=16, opacity=0.95,
                color=[churn_prob], colorscale='RdYlGn_r', cmin=0, cmax=1,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            )
        ))
        
        # Trace 1 ->"Benchmarks"
        fig.add_trace(go.Scatter3d(
            name='Benchmarks',  # <--- ADDED PROPER NAME
            x=[15, 310], y=[45, 1], z=[9000, 40],
            mode='text',
            text=['◆ Champions', '◆ Churned'],
            textfont=dict(color=['#00e5a0', "#e7001f"], size=10),
            hoverinfo='none'
        ))
        
        fig.update_layout(
            paper_bgcolor=chart_colors['paper_bgcolor'], 
            plot_bgcolor=chart_colors['plot_bgcolor'],
            
            # --- MOVED LEGEND TO BOTTOM RIGHT ---
            legend=dict(
                x=0.99,
                y=0.01,
                xanchor='right',
                yanchor='bottom',
                font=dict(family='IBM Plex Mono', size=10, color=chart_colors['secondary_text']),
                bgcolor='rgba(0,0,0,0)' # Transparent background
            ),
            
            hoverlabel=dict(
                bgcolor=chart_colors['plot_bgcolor'], 
                font_size=12, 
                font_family="IBM Plex Mono", 
                font_color=chart_colors['text_color'],
                bordercolor="#00e5a0"
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text='Recency', font=dict(family='IBM Plex Mono', size=14, color=chart_colors['secondary_text'])), 
                    backgroundcolor=chart_colors['plot_bgcolor'],  
                    gridcolor=chart_colors['grid_color'],         
                    showbackground=True,
                    tickfont=dict(family='IBM Plex Mono', size=13, color=chart_colors['text_color']) 
                ),
                yaxis=dict(
                    title=dict(text='Frequency', font=dict(family='IBM Plex Mono', size=14, color=chart_colors['secondary_text'])), 
                    backgroundcolor=chart_colors['plot_bgcolor'],
                    gridcolor=chart_colors['grid_color'],
                    showbackground=True,
                    tickfont=dict(family='IBM Plex Mono', size=13, color=chart_colors['text_color'])
                ),
                zaxis=dict(
                    title=dict(text='Monetary', font=dict(family='IBM Plex Mono', size=14, color=chart_colors['secondary_text'])), 
                    backgroundcolor=chart_colors['plot_bgcolor'],
                    gridcolor=chart_colors['grid_color'],
                    showbackground=True,
                    tickfont=dict(family='IBM Plex Mono', size=13, color=chart_colors['text_color'])
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=340
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # SHAP + AI
        col_shap, col_ai = st.columns(2)

        with col_shap:
            factors = get_shap_factors(profile)
            render_shap_factors(factors)

        with col_ai:
            st.markdown('<div class="section-label">AI Strategy</div>', unsafe_allow_html=True)

            if st.button("◈ Generate Strategy", type="primary", use_container_width=True):
                factors_text = "\n".join(
                    [f"- {f['feature']}: {f['value']:.2f} ({f['effect']} churn)"
                     for f in (get_shap_factors(profile) or [])]
                ) or "SHAP not available"

                prompt = f"""Customer retention analysis:

CHURN RISK: {churn_prob:.1%} ({risk_label})
Days inactive: {recency} | Orders: {frequency} | LTV: ${monetary:.0f}
Avg order: ${avg_order_value} | Products: {product_diversity} | Trend: {spending_trend}
Weekend buyer: {is_weekend*100:.0f}% | Age: {customer_age}d

SHAP DRIVERS:
{factors_text}

Write a sharp, specific retention strategy:
1. Immediate action (today)
2. Personalized email subject + 2-sentence body
3. Exact offer (% discount + why this amount)
4. One behavioral insight

Be direct. Reference their actual numbers. No generic advice."""

                placeholder = st.empty()
                stream_to_placeholder(prompt, placeholder)
            else:
                st.markdown(
                    '<div class="ai-stream-box" style="color:var(--text-muted);font-size:0.78rem">'
                    'Click to generate AI retention strategy based on this customer\'s behavior.'
                    '</div>',
                    unsafe_allow_html=True
                )


# ============================================
# PAGE 2 — AI CHAT ANALYST 
# ============================================
elif page == "2. AI Chat Analyst":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">AI Analyst</span>
        <span class="page-tag">Chat</span>
    </div>
    <div class="page-subtitle">Ask anything about your customer data in plain English</div>
    """, unsafe_allow_html=True)

    # Quick question chips
    st.markdown('<div class="section-label">Quick Questions</div>', unsafe_allow_html=True)
    qc1, qc2, qc3, qc4 = st.columns(4)
    questions = {
        "Who to call today?":     "Which top customers should I prioritize calling today and why?",
        "Why are they churning?": "What are the top 5 behavioral reasons customers are churning?",
        "Revenue impact?":        "If we retain 25% of high-risk customers, how much revenue do we save?",
        "Best offer strategy?":   "What discount strategy maximizes ROI across different risk segments?"
    }
    triggered = None
    for btn, (label, question) in zip([qc1, qc2, qc3, qc4], questions.items()):
        with btn:
            if st.button(btn_label := label, use_container_width=True):
                triggered = question

    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        if role == "user":
            with st.chat_message(role, avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message(role, avatar="🧠"):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your customers...") or triggered

    if user_input:
        if not ai_active:
            st.info(
                "ℹ️ AI Chat requires a GROQ_API_KEY. "
                "Add GROQ_API_KEY to .env file to enable chat."
            )
        else:
            top5 = customer_db_sorted.head(5)[
                ['Customer ID', 'churn_probability', 'Recency', 'Monetary', 'Frequency']
            ].to_string(index=False)

            system_instruction = f"""You are a senior customer analytics AI for a retention platform.
            Database: {len(customer_db):,} customers.
            High risk: {high_risk_count:,} | Revenue at risk: ${revenue_risk:,.0f}
            Model: XGBoost, {metadata['accuracy']:.1%} accuracy, {metadata['roc_auc']:.3f} AUC
            Top 5 at-risk:
            {top5}
            Be precise, data-driven, and concise. Max 4 sentences per answer."""

            # Display user message immediately
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            
            # Add to internal history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Call Groq
            with st.chat_message("assistant", avatar="🧠"):
                ph = st.empty()
                full_text = ""
                
                try:
                    # Prepare messages for Groq
                    messages = [{"role": "system", "content": system_instruction}]
                    # Append chat history (Groq uses 'user'/'assistant' roles just like st.session_state)
                    messages.extend(st.session_state.chat_history)

                    stream = client.chat.completions.create(
                        messages=messages,
                        model="llama-3.3-70b-versatile",
                        stream=True
                    )
                    
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_text += content
                            ph.markdown(full_text + "▌")
                    ph.markdown(full_text)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": full_text})
                    
                except Exception as e:
                    ph.error(f"Error: {e}")

    if st.session_state.chat_history:
        if st.button("◈ Clear"):
            st.session_state.chat_history = []
            st.rerun()


# ============================================
# PAGE 3 — RISK OVERVIEW
# ============================================
elif page == "3. Risk Overview":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Risk Overview</span>
        <span class="page-tag">Live</span>
    </div>
    <div class="page-subtitle">Full portfolio churn risk across all customer segments</div>
    """, unsafe_allow_html=True)

    render_kpis([
        ("Total Customers",  f"{len(customer_db):,}",     "in database",         "var(--accent-blue)"),
        ("High Risk",        f"{high_risk_count:,}",       "need action now",      "var(--accent-red)"),
        ("Medium Risk",      f"{(customer_db['risk_level']=='MEDIUM').sum():,}", "monitor closely", "var(--accent-amber)"),
        ("Revenue at Risk",  f"${revenue_risk:,.0f}",      "high risk segment",    "var(--accent-red)"),
    ])

    c1, c2 = st.columns(2)
    
    # Get dynamic colors for charts
    chart_colors = get_chart_colors()

    with c1:
        st.markdown('<div class="section-label">Risk Distribution</div>', unsafe_allow_html=True)
        rc = customer_db['risk_level'].value_counts()
        fig_d = go.Figure(go.Pie(
            labels=rc.index, values=rc.values,
            marker_colors=['#ff3d57', '#f5a623', '#00e5a0'],
            hole=0.55,
            textfont=dict(family='IBM Plex Mono', size=10, color=chart_colors['text_color']),
            hovertemplate='%{label}<br>%{value:,} customers<br>%{percent}<extra></extra>'
        ))
        fig_d.update_layout(
            paper_bgcolor=chart_colors['paper_bgcolor'],
            showlegend=True,
            legend=dict(font=dict(family='IBM Plex Mono', size=9, color=chart_colors['secondary_text'])),
            margin=dict(l=20, r=20, t=20, b=20),
            height=280  # Reduced for better mobile display
        )
        st.plotly_chart(fig_d, use_container_width=True, theme=None)

    with c2:
        st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)
        fig_h = go.Figure(go.Histogram(
            x=customer_db['churn_probability'], nbinsx=40,
            marker=dict(color='#00e5a0', opacity=0.7,
                        line=dict(color=chart_colors['plot_bgcolor'], width=0.5))
        ))
        fig_h.update_layout(
            paper_bgcolor=chart_colors['paper_bgcolor'], plot_bgcolor=chart_colors['plot_bgcolor'],
            xaxis=dict(
                title=dict(text='Churn Probability', font=dict(family='IBM Plex Mono', size=9, color=chart_colors['secondary_text'])),
                gridcolor=chart_colors['grid_color'],
                tickfont=dict(family='IBM Plex Mono', size=8, color=chart_colors['secondary_text'])
            ),
            yaxis=dict(
                title=dict(text='Customers', font=dict(family='IBM Plex Mono', size=9, color=chart_colors['secondary_text'])),
                gridcolor=chart_colors['grid_color'],
                tickfont=dict(family='IBM Plex Mono', size=8, color=chart_colors['secondary_text'])
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            height=280  # Reduced for better mobile display
        )
        st.plotly_chart(fig_h, use_container_width=True, theme=None)

    # Recency vs Monetary
    st.markdown('<div class="section-label">Recency vs Monetary — Risk Heatmap</div>', unsafe_allow_html=True)
    fig_s = go.Figure(go.Scatter(
        x=customer_db['Recency'],
        y=customer_db['Monetary'],
        mode='markers',
        marker=dict(
            color=customer_db['churn_probability'],
            colorscale='RdYlGn_r', size=4, opacity=0.55,
            colorbar=dict(
                title=dict(text='Risk', font=dict(family='IBM Plex Mono', size=9, color=chart_colors['secondary_text'])),
                tickfont=dict(family='IBM Plex Mono', size=8, color=chart_colors['secondary_text'])
            )
        ),
        hovertemplate='Recency: %{x}d<br>LTV: $%{y:.0f}<extra></extra>'
    ))
    fig_s.update_layout(
        paper_bgcolor=chart_colors['paper_bgcolor'], plot_bgcolor=chart_colors['plot_bgcolor'],
        xaxis=dict(
            title=dict(text='Days Since Last Purchase', font=dict(family='IBM Plex Mono', size=9, color=chart_colors['secondary_text'])),
            gridcolor=chart_colors['grid_color'],
            tickfont=dict(family='IBM Plex Mono', size=8, color=chart_colors['secondary_text'])
        ),
        yaxis=dict(
            title=dict(text='Total Spend ($)', font=dict(family='IBM Plex Mono', size=9, color=chart_colors['secondary_text'])),
            gridcolor=chart_colors['grid_color'],
            tickfont=dict(family='IBM Plex Mono', size=8, color=chart_colors['secondary_text'])
        ),
        margin=dict(l=20, r=60, t=10, b=40),
        height=360
    )
    st.plotly_chart(fig_s, use_container_width=True, theme=None)


# ============================================
# PAGE 4 — AT-RISK CUSTOMERS
# ============================================
elif page == "4. At-Risk Customers":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">At-Risk Customers</span>
        <span class="page-tag">Auto-detected</span>
    </div>
    <div class="page-subtitle">Sorted by highest churn probability — no manual lookup required</div>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        risk_filter = st.selectbox("Risk Segment", ["HIGH", "MEDIUM", "ALL"])
    with fc2:
        top_n = st.slider("Show Top N", 10, len(customer_db), 30)
    with fc3:
        sort_by = st.selectbox(
            "Sort By",
            ["Churn Risk", "Revenue Impact", "Days Inactive"]
        )

    filt = customer_db_sorted if risk_filter == "ALL" else \
           customer_db_sorted[customer_db_sorted['risk_level'] == risk_filter]

    sort_col = {"Churn Risk": "churn_probability",
                "Revenue Impact": "Monetary",
                "Days Inactive": "Recency"}[sort_by]

    display_df = filt.sort_values(sort_col, ascending=False).head(top_n)

    table_df = display_df[[
        'Customer ID', 'churn_probability', 'risk_level',
        'Recency', 'Frequency', 'Monetary', 'product_diversity'
    ]].copy()

    table_df.columns = ['ID', 'Churn %', 'Risk',
                        'Days Inactive', 'Orders', 'LTV ($)', 'Products']
    table_df['Churn %'] = table_df['Churn %'].apply(lambda x: f"{x:.1%}")
    table_df['LTV ($)'] = table_df['LTV ($)'].apply(lambda x: f"${x:,.0f}")

    st.dataframe(table_df, use_container_width=True, hide_index=True, height=420)

    dl_col, ai_col = st.columns(2)

    with dl_col:
        st.download_button(
            "◈ Export CSV",
            data=display_df.to_csv(index=False),
            file_name=f"at_risk_{risk_filter}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with ai_col:
        if st.button("◈ AI Strategy — Top Customer", use_container_width=True):
            if len(display_df) == 0:
                st.error("No customers visible in current view.")
            else:
                top_display = display_df.iloc[0]
                cust_id = top_display['Customer ID']
                top = customer_db[customer_db['Customer ID'] == cust_id].iloc[0]
                
                prob = float(top['churn_probability'])
                factors = get_shap_factors(top.to_dict())
            facts   = "\n".join([f"- {f['feature']}: {f['value']:.2f} ({f['effect']} churn)"
                                 for f in factors]) or "N/A"

            prompt = f"""Most at-risk customer analysis:

Risk: {prob:.1%} | Inactive: {top['Recency']:.0f}d | LTV: ${top['Monetary']:.0f}
Orders: {top['Frequency']:.0f} | Products: {top['product_diversity']:.0f}

SHAP drivers:
{facts}

Provide:
1. Single most important action (be specific)
2. Email subject line + 2-line body (use their actual data)
3. Exact discount recommendation with ROI reasoning
4. Risk if we do nothing (revenue estimate)

Sharp and direct. No fluff."""

            with st.expander("◈ AI Retention Strategy", expanded=True):
                ph = st.empty()
                stream_to_placeholder(prompt, ph)


# ============================================
# PAGE 5 — EXECUTIVE BRIEF
# ============================================
elif page == "5. Executive Brief":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Executive Brief</span>
        <span class="page-tag">Board-Level</span>
    </div>
    <div class="page-subtitle">AI-generated business intelligence for leadership</div>
    """, unsafe_allow_html=True)

    med_risk     = int((customer_db['risk_level'] == 'MEDIUM').sum())
    avg_risk     = customer_db['churn_probability'].mean()
    high_df      = customer_db[customer_db['risk_level'] == 'HIGH']

    render_kpis([
        ("Total Customers",  f"{len(customer_db):,}",   "scored",              "var(--accent-blue)"),
        ("Revenue at Risk",  f"${revenue_risk:,.0f}",   "high-risk segment",  "var(--accent-red)"),
        ("Avg Churn Risk",   f"{avg_risk:.1%}",          "portfolio wide",      "var(--accent-amber)"),
        ("Model Accuracy",   f"{metadata['accuracy']:.1%}", f"AUC {metadata['roc_auc']:.3f}", "var(--accent-green)"),
    ])

    col_brief, col_chart = st.columns([3, 2])

    with col_brief:
        st.markdown('<div class="section-label">AI Executive Report</div>', unsafe_allow_html=True)

        if st.button("◈ Generate Board Report", type="primary", use_container_width=True):
            prompt = f"""You are a Chief Revenue Officer presenting a retention brief.

DATA:
- Portfolio: {len(customer_db):,} customers
- High Risk: {high_risk_count:,} ({high_risk_count/len(customer_db):.1%})
- Medium Risk: {med_risk:,}
- Revenue at Risk: ${revenue_risk:,.0f}
- Avg Churn Probability: {avg_risk:.1%}
- Avg Days Inactive (High Risk): {high_df['Recency'].mean():.0f}
- Avg LTV (High Risk): ${high_df['Monetary'].mean():,.0f}
- Model: XGBoost, {metadata['accuracy']:.1%} accuracy

Write a 5-sentence executive brief:
Sentence 1: Current situation (use numbers)
Sentence 2: Primary risk driver
Sentence 3: Revenue impact if no action
Sentence 4: Two recommended actions with expected outcomes
Sentence 5: 30-day target metric

Executive tone. Data-driven. No bullet points."""

            ph = st.empty()
            stream_to_placeholder(prompt, ph)
        else:
            st.markdown(
                '<div class="ai-stream-box" style="color:var(--text-muted);font-size:0.78rem">'
                'Click to generate AI executive brief using live customer data.'
                '</div>',
                unsafe_allow_html=True
            )

        # Key metrics table
        st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            "Metric": [
                "High Risk Customers",
                "Avg Days Inactive (High Risk)",
                "Avg LTV (High Risk)",
                "Total Revenue at Risk",
                "ROC-AUC Score",
                "Features in Model"
            ],
            "Value": [
                f"{high_risk_count:,}",
                f"{high_df['Recency'].mean():.0f} days",
                f"${high_df['Monetary'].mean():,.0f}",
                f"${revenue_risk:,.0f}",
                f"{metadata['roc_auc']:.3f}",
                str(len(feature_names))
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col_chart:
        st.markdown('<div class="section-label">Revenue by Segment</div>', unsafe_allow_html=True)
        
        # Get dynamic colors for charts
        chart_colors = get_chart_colors()
        
        seg = customer_db.groupby('risk_level', observed=False)['Monetary'].sum().reset_index()
        
        # FIX: Map colors explicitly so HIGH is always Red
        color_map = {'HIGH': '#ff3d57', 'MEDIUM': '#f5a623', 'LOW': '#00e5a0'}
        colors = seg['risk_level'].map(color_map).tolist()
        
        fig_b = go.Figure(go.Bar(
            x=seg['risk_level'],
            y=seg['Monetary'],
            marker=dict(color=colors, line=dict(width=0)), # Apply fixed colors
            text=seg['Monetary'].apply(lambda x: f"${x:,.0f}"),
            textposition='outside',
            textfont=dict(family='IBM Plex Mono', size=9, color=chart_colors['text_color'])
        ))
        fig_b.update_layout(
            paper_bgcolor=chart_colors['paper_bgcolor'], plot_bgcolor=chart_colors['plot_bgcolor'],
            xaxis=dict(gridcolor=chart_colors['grid_color'],
                       tickfont=dict(family='IBM Plex Mono', size=9, color=chart_colors['secondary_text'])),
            yaxis=dict(gridcolor=chart_colors['grid_color'],
                       tickfont=dict(family='IBM Plex Mono', size=9, color=chart_colors['secondary_text'])),
            margin=dict(l=10, r=10, t=30, b=10),
            height=280
        )
        st.plotly_chart(fig_b, use_container_width=True, theme=None)

        # ============================================
# PAGE 6 — BATCH SCORER
# ============================================
elif page == "6. Batch Scorer":
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Batch Scorer</span>
        <span class="page-tag">Pipeline</span>
    </div>
    <div class="page-subtitle">Upload raw customer data to generate bulk retention scores</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Data Ingestion Pipeline</div>', unsafe_allow_html=True)

    # The Streamlit Drag-and-Drop widget
    uploaded_file = st.file_uploader("Upload a CSV file to begin bulk scoring", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read the uploaded file into Pandas
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"Successfully loaded {len(batch_df):,} customer records. Initializing XGBoost scoring pipeline...")
            
            # --- FEATURE MATCHING ---
            # Create a copy to hold our model-ready features
            score_df = batch_df.copy()
            
            # Check for missing columns and fill them with 0 so the model doesn't crash
            missing_cols = set(feature_names) - set(score_df.columns)
            for c in missing_cols:
                score_df[c] = 0
                
            # Keep only the exact columns the model was trained on, in the exact right order
            X_batch = score_df[feature_names].fillna(0)
            
            # --- BATCH PREDICTION ---
            with st.spinner("Scoring thousands of records..."):
                # Run the entire dataframe through the model at once!
                batch_probs = model.predict_proba(X_batch)[:, 1]
                
            # Attach the new AI scores back to the original uploaded data
            batch_df['Churn_Probability'] = batch_probs
            batch_df['Risk_Level'] = pd.cut(
                batch_probs,
                bins=[0, 0.4, 0.7, 1.0],
                labels=['LOW', 'MEDIUM', 'HIGH']
            )
            
            # Sort so the highest risk customers are at the very top
            batch_df = batch_df.sort_values('Churn_Probability', ascending=False)
            
            # --- UI REPORT ---
            st.markdown('<div class="section-label">Scoring Results</div>', unsafe_allow_html=True)
            
            # Calculate summary metrics
            high_risk_batch = (batch_df['Risk_Level'] == 'HIGH').sum()
            avg_risk_batch = batch_df['Churn_Probability'].mean()
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Scored", f"{len(batch_df):,}")
            with c2:
                st.metric("High Risk Identified", f"{high_risk_batch:,}")
            with c3:
                st.metric("Average Churn Risk", f"{avg_risk_batch:.1%}")
            
            # Display the freshly scored data
            st.dataframe(batch_df.head(len(batch_df)), use_container_width=True)
            
            # --- DOWNLOAD BUTTON ---
            st.markdown('<div class="section-label">Export Intelligence</div>', unsafe_allow_html=True)
            
            csv_export = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="◈ Download Scored Dataset (CSV)",
                data=csv_export,
                file_name="retainion_batch_scored.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Pipeline Error: {e}")