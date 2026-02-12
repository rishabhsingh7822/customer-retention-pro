import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD THE TRAINED MODEL ---
# We use @st.cache_resource so it only loads once (faster)
@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_churn.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found! Please run the notebook to save 'xgboost_churn.pkl' in the models folder.")
    st.stop()

# --- 2. UI LAYOUT ---
st.set_page_config(page_title="Churn Predictor", page_icon="🔮")

st.title("🔮 Customer Retention Engine")
st.markdown("Adjust the sliders to see how customer behavior impacts Churn Risk.")

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    recency = st.slider("Days Since Last Purchase (Recency)", 
                        min_value=0, max_value=365, value=30)
    
    frequency = st.slider("Total Orders (Frequency)", 
                          min_value=1, max_value=50, value=5)

with col2:
    monetary = st.number_input("Total Spend $ (Monetary)", 
                               min_value=0, max_value=10000, value=500)

# --- 3. PREDICTION LOGIC ---
# Create a dataframe matching the training data structure
input_data = pd.DataFrame({
    'Recency': [recency],
    'Frequency': [frequency],
    'Monetary': [monetary]
})

if st.button("Predict Churn Risk"):
    # Get probability (0 to 1)
    prediction_prob = model.predict_proba(input_data)[0][1]
    prediction_class = model.predict(input_data)[0]
    
    st.divider()
    
    # Display Results
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric("Churn Probability", f"{prediction_prob:.1%}")
    
    with col_res2:
        if prediction_prob > 0.7:
            st.error("HIGH RISK: Customer Likely to Churn")
        elif prediction_prob > 0.3:
            st.warning("MEDIUM RISK: At Risk")
        else:
            st.success("LOW RISK: Loyal Customer")