import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt


def generate_email(probability, monetary):
    """Generates a retention email based on risk level."""
    if probability > 0.7:
        offer = "20% OFF"
        tone = "We miss you!"
    elif probability > 0.3:
        offer = "10% OFF"
        tone = "Check out our new arrivals."
    else:
        return "Customer is Loyal. No action needed."
        
    email_draft = f"""
    Subject: {tone} Here is a gift for you.
    
    Hi there,
    
    We noticed it's been a while. As a valued customer who has spent ${monetary}, 
    we want to offer you {offer} on your next purchase.
    
    Use code: COMEBACK2026
    """
    return email_draft

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

    # --- EXPLAINABILITY ---
    st.subheader("Why is this customer at risk?")

    # 1. Create the Explainer (this runs fast for XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # 2. Create the Plot
    fig, ax = plt.subplots()
    # We use a bar plot to show Positive (Red) vs Negative (Blue) impact
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)

    # 3. Render in Streamlit
    st.pyplot(fig)

    # --- ACTIONABLE INSIGHT ---
    st.divider()
    st.subheader("Recommended Action")
    email = generate_email(prediction_prob, monetary)
    st.text_area("Draft Email", email, height=200)

    # Display Results
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.metric("Churn Probability", f"{prediction_prob:.1%}")

    with col_res2:
        if prediction_prob > 0.7:
            st.error("HIGH RISK: Customer Likely to Churn")
        elif prediction_prob > 0.5:
            st.warning("MEDIUM RISK: At Risk")
        else:
            st.success("LOW RISK: Loyal Customer")