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

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    """
    Loads the pre-trained XGBoost model from the 'models' directory.
    """
    model_path = os.path.join("models", "xgboost_churn.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Error: Model not found at {model_path}. Please check the folder structure.")
        st.stop()
        
    return joblib.load(model_path)

model = load_model()

# --- 3. UTILITY FUNCTIONS ---
def get_risk_color(prob):
    """Returns a hex color code based on the churn probability."""
    if prob > 0.7:
        return "#FF4B4B"  # Brand Red
    if prob > 0.4:
        return "#FFA500"  # Warning Orange
    return "#00C853"      # Success Green

def generate_email(probability, monetary):
    """Generates a dynamic email draft based on risk level and LTV."""
    if probability > 0.7:
        return (f"HIGH RISK \n\n"
                f"Subject: We miss you! (20% OFF)\n\n"
                f"Hi there,\n"
                f"We noticed it's been a while. Since you are a VIP (LTV: ${monetary}), "
                f"here is 20% off.\n"
                f"Code: SAVE20")
    elif probability > 0.4:
        return (f"MEDIUM RISK \n\n"
                f"Subject: New items for you\n\n"
                f"Hi there,\n"
                f"Check out our latest collection. We think you'll love these arrivals.\n"
                f"Code: WELCOMEBACK")
    else:
        return (f"LOW RISK \n\n"
                f"Subject: Thank you\n\n"
                f"Hi there,\n"
                f"Just a quick note to say thanks for your loyalty!\n"
                f"(No discount needed)")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Live Simulation")
    st.write("Adjust the sliders to see real-time churn risk updates.")
    
    # Input sliders
    recency = st.slider("Recency (Days Since Last Purchase)", 0, 365, 30)
    frequency = st.slider("Frequency (Total Orders)", 1, 50, 5)
    monetary = st.number_input("Monetary Value (Total Spend $)", 0, 10000, 500)
    
    st.divider()
    st.caption("Model: XGBoost Classifier | v1.0")

# --- 5. PREDICTION LOGIC ---
# Create input DataFrame matching the model's expected features
input_data = pd.DataFrame({
    'Recency': [recency],
    'Frequency': [frequency],
    'Monetary': [monetary]
})

try:
    # Extracting probability for the positive class (Churn)
    prediction_prob = model.predict_proba(input_data)[0][1]
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# --- 6. DASHBOARD UI ---
st.title("🔮 Retention Command Center")

# Metric Row
col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{prediction_prob:.1%}", delta="Risk Level", delta_color="off")
col2.metric("Customer Lifetime Value", f"${monetary:,}")

status_text = 'CRITICAL' if prediction_prob > 0.7 else 'SAFE'
col3.markdown(
    f"### Status: <span style='color:{get_risk_color(prediction_prob)}'>{status_text}</span>", 
    unsafe_allow_html=True
)

st.divider()

# --- 7. 3D INTERACTIVE GRAPH (Visibility Fixed) ---
col_graph, col_action = st.columns([2, 1])

with col_graph:
    st.subheader("Customer Vector Space")
    
    fig = go.Figure()

    # The Customer Orb
    fig.add_trace(go.Scatter3d(
        x=[recency], y=[frequency], z=[monetary],
        mode='markers',
        marker=dict(
            size=18,
            color=[prediction_prob],
            colorscale='RdYlGn_r', # Red for high churn, Green for low
            showscale=True,
            colorbar=dict(title="Churn Risk", thickness=15, x=-0.2),
            cmin=0, cmax=1,
            opacity=1.0,
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
        textfont=dict(color="white" if prediction_prob > 0.5 else "red")
    ))

    # High-Contrast Layout
    fig.update_layout(
        template="plotly_dark",  # Forces dark background for contrast
        scene=dict(
            xaxis=dict(
                title='Recency', 
                backgroundcolor="rgb(30, 30, 30)", 
                gridcolor="gray", 
                showbackground=True
            ),
            yaxis=dict(
                title='Frequency', 
                backgroundcolor="rgb(30, 30, 30)", 
                gridcolor="gray", 
                showbackground=True
            ),
            zaxis=dict(
                title='Monetary', 
                backgroundcolor="rgb(30, 30, 30)", 
                gridcolor="gray", 
                showbackground=True
            )
        ),
        paper_bgcolor="rgba(0,0,0,0)", # Keeps outer container transparent
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

with col_action:
    st.subheader("Recommended Action")
    st.info("Automated Retention Strategy:")
    
    email_content = generate_email(prediction_prob, monetary)
    st.text_area("Email Draft", email_content, height=300)
    
    if st.button("🚀 Queue for Sending"):
        st.success("Campaign added to automation queue!")