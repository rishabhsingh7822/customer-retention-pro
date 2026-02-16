import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objs as go
import numpy as np
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Retention Engine AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LOAD MODEL (Fixed Path) ---
@st.cache_resource
def load_model():
    # This points directly to your 'models' folder
    model_path = os.path.join("models", "xgboost_churn.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Error: Model not found at {model_path}. Please check the folder structure.")
        st.stop()
        
    return joblib.load(model_path)

model = load_model()

# --- 3. UTILITY FUNCTIONS ---
def get_risk_color(prob):
    if prob > 0.7: return "#FF4B4B" # Brand Red
    if prob > 0.4: return "#FFA500" # Warning Orange
    return "#00C853"                # Success Green

def generate_email(probability, monetary):
    if probability > 0.7:
        return f"🚨 HIGH RISK \n\nSubject: We miss you! (20% OFF)\n\nHi there,\nWe noticed it's been a while. Since you are a VIP (LTV: ${monetary}), here is 20% off.\nCode: SAVE20"
    elif probability > 0.4:
        return f"⚠️ MEDIUM RISK \n\nSubject: New items for you\n\nHi there,\nCheck out our latest collection. We think you'll love these new arrivals.\nCode: WELCOMEBACK"
    else:
        return f"✅ LOW RISK \n\nSubject: Thank you\n\nHi there,\nJust a quick note to say thanks for your loyalty!\n(No discount needed)"

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Live Simulation")
    

    # Input sliders (Real-time)
    recency = st.slider("Recency (Days)", 0, 365, 30)
    frequency = st.slider("Frequency (Orders)", 1, 50, 5)
    monetary = st.number_input("Monetary Value ($)", 0, 10000, 500)
    
    st.divider()
    st.caption("Model: XGBoost Classifier | v1.0")

# --- 5. PREDICTION ---
input_data = pd.DataFrame({
    'Recency': [recency],
    'Frequency': [frequency],
    'Monetary': [monetary]
})

try:
    prediction_prob = model.predict_proba(input_data)[0][1]
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# --- 6. DASHBOARD UI ---
st.title("🔮 Retention Command Center")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{prediction_prob:.1%}", delta="Risk Level", delta_color="off")
col2.metric("Customer Lifetime Value", f"${monetary}")
col3.markdown(f"### Status: <span style='color:{get_risk_color(prediction_prob)}'>{'CRITICAL' if prediction_prob > 0.7 else 'SAFE'}</span>", unsafe_allow_html=True)

st.divider()

# --- 7. 3D INTERACTIVE GRAPH ---
col_graph, col_action = st.columns([2, 1])

with col_graph:
    st.subheader("Customer Vector Space")
    
    fig = go.Figure()

    # The Customer Orb
    fig.add_trace(go.Scatter3d(
        x=[recency], y=[frequency], z=[monetary],
        mode='markers',
        # 2. Custom Tooltip (Hover)
        hovertemplate=(
            "<b>%{text}</b><br><br>" +
            "Recency: %{x} days<br>" +
            "Frequency: %{y} orders<br>" +
            "LTV: $%{z}<br>" +
            "<extra></extra>" # Hides the secondary box
        ),
        marker=dict(
            size=25,
            color=[prediction_prob],
            colorscale='RdYlGn_r', 
            cmin=0, cmax=1,
            opacity=0.9,
            line=dict(color='white', width=2)
        ),
        name='Current Customer'
    ))

    # Reference Zones
    fig.add_trace(go.Scatter3d(
        x=[20, 300], y=[40, 2], z=[8000, 100],
        mode='text',
        text=['💎 VIP', '💀 Churn'],
        textposition="top center",
        textfont=dict(color="white"),
        hoverinfo='none'
    ))

    # 3. Clean Dark Style (High Contrast Fix)
    fig.update_layout(
        template="plotly_dark", # Forces white text and dark theme
        scene=dict(
            # Dark background for the cube, Light Grey grid lines for visibility
            xaxis=dict(title='Recency', backgroundcolor="#1E1E1E", gridcolor="lightgrey", showbackground=True, zerolinecolor="white"),
            yaxis=dict(title='Frequency', backgroundcolor="#1E1E1E", gridcolor="lightgrey", showbackground=True, zerolinecolor="white"),
            zaxis=dict(title='Monetary', backgroundcolor="#1E1E1E", gridcolor="lightgrey", showbackground=True, zerolinecolor="white"),
            bgcolor="#1E1E1E"
        ),
        paper_bgcolor="#262730", # Distinct card background color
        margin=dict(l=10, r=10, b=10, t=40),
        height=400
    )

    # 4. THE CRITICAL FIX: theme=None
    # This tells Streamlit: "Do not overwrite my colors!"
    st.plotly_chart(fig, use_container_width=True, theme=None)

with col_action:
    st.subheader("⚡ Recommended Action")
    st.info("Automated Strategy:")
    st.text_area("Email Draft", generate_email(prediction_prob, monetary), height=250)