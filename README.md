# RETAINION: AI Customer Retention Platform ◈

RETAINION is an end-to-end, AI-powered customer retention dashboard. It combines traditional Machine Learning (XGBoost) with Generative AI (Llama 3 via Groq) to predict customer churn, explain the mathematical reasons behind it, and automatically generate personalized retention strategies.

This platform moves beyond basic analytics—it turns predictive data into immediate business action.

---

## 🚀 Key Features

* **Live Predictor:** Adjust customer metrics (recency, frequency, monetary value, etc.) in real-time to instantly see how their churn probability changes.
* **Explainable AI (SHAP):** No "black-box" decisions. The dashboard shows exactly *why* a customer is flagged as high risk (e.g., "Days Inactive is increasing churn risk").
* **AI Retention Strategist:** Uses Groq's API (Llama-3.3-70b) to read the mathematical SHAP drivers and generate a highly personalized email and discount offer to save the customer.
* **AI Chat Analyst:** Chat with your customer database in plain English. Ask questions like *"Who should I call today?"* or *"What is the revenue impact of our high-risk segment?"*
* **Batch Scorer:** Upload a raw CSV of thousands of customers. The pipeline cleans the data, runs it through the XGBoost model in bulk, and lets you download the scored dataset.
* **Executive Brief:** One-click, auto-generated board reports summarizing current portfolio risk and revenue impact.

---

## 🛠️ Technology Stack

* **Frontend & UI:** Streamlit (heavily customized with raw CSS for a sleek, responsive, dark/light mode interface)
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas
* **Model Explainability:** SHAP (SHapley Additive exPlanations)
* **Generative AI:** Groq Cloud API (Llama 3.3 70B)
* **Data Visualization:** Plotly (Interactive 3D vector spaces, heatmaps, and charts)
* **Security:** Streamlit-Authenticator (Brute-force protection, auto-lockout, and secure password hashing)

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/customer-retention-pro.git
   cd customer-retention-pro
   ```

2. **Install dependencies:**
   Make sure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your secret keys.
   ```env
   # Your Groq API key for the GenAI features
   GROQ_API_KEY="gsk_your_api_key_here"
   
   # Developer override key for the security lockout system
   DEV_PASSWORD="your_secure_password"
   ```

4. **Run the Application:**
   ```bash
   streamlit run app/main.py
   ```

---

## 📁 Project Structure

```
├── app/
│   ├── main.py          # Core Streamlit application and UI logic
│   └── style.css        # Custom CSS for the design system and responsiveness
├── models/
│   ├── xgboost_churn.pkl       # Pre-trained Machine Learning model
│   ├── shap_explainer.pkl      # SHAP explainer for model interpretability 
│   ├── customer_database.csv   # Demo database for live analysis
│   └── model_metadata.json     # Configuration file map for the 17 features
├── .env                 # Secret keys (not tracked in git)
└── requirements.txt     # Python dependencies
```

---

## 🧠 How the AI Works Together

1. **The Math (XGBoost):** The ML model ingests 17 behavioral features (`Recency`, `LTV`, `Orders_Last_30d`, etc.) and calculates a Churn Probability (0% to 100%).
2. **The "Why" (SHAP):** The SHAP explainer breaks down that probability, identifying which specific features pushed the score up or down.
3. **The Action (LLM):** The Groq LLM reads the SHAP output as context and writes a targeted, empathetic strategy to fix the specific behavioral problems identified by the math.

---

## 📝 License
This project is for portfolio and demonstration purposes. Feel free to fork and modify!