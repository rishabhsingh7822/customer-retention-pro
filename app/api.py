from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import json
from dotenv import load_dotenv

# 1. SECURITY : Load from .env
load_dotenv()
# Switch to checking Groq key for consistency
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment.")

app = FastAPI(title="Retention Engine API", version="2.0")

# Enable CORS for dashboard connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. PATH CONFIGURATION:
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check if 'models' is in current dir or parent dir
if os.path.exists(os.path.join(current_dir, "models")):
    model_dir = os.path.join(current_dir, "models")
elif os.path.exists(os.path.join(os.path.dirname(current_dir), "models")):
    model_dir = os.path.join(os.path.dirname(current_dir), "models")
else:
    # Fallback for some deployment structures
    model_dir = current_dir 

model_path = os.path.join(model_dir, "xgboost_churn.pkl")
metadata_path = os.path.join(model_dir, "model_metadata.json")
customer_db_path = os.path.join(model_dir, "customer_database.csv")

# 3. LOAD ARTIFACTS
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Load metadata to get correct feature names
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
else:
    # Fallback feature list if metadata is missing
    feature_names = [
        'Recency', 'Frequency', 'Monetary', 'customer_age_days',
        'days_between_purchases', 'orders_last_30d', 'orders_last_90d',
        'avg_order_value', 'max_order_value', 'min_order_value',
        'std_order_value', 'product_diversity', 'repeat_purchase_rate',
        'avg_basket_size', 'total_items_bought', 'is_weekend',
        'hour', 'spending_increasing'
    ]

if os.path.exists(customer_db_path):
    customer_db = pd.read_csv(customer_db_path)
    # Pre-calculate risk levels for /at-risk endpoint
    if model:
        X_all = customer_db[feature_names].fillna(0)
        probs = model.predict_proba(X_all)[:, 1]
        customer_db['churn_probability'] = probs
        customer_db['risk_level'] = pd.cut(probs, bins=[0, 0.4, 0.7, 1.0], labels=['LOW', 'MEDIUM', 'HIGH'])
else:
    customer_db = pd.DataFrame()

# 4. DATA MODEL
class CustomerInput(BaseModel):
    recency: int
    frequency: int
    monetary: float

# 5. FEATURE ENGINEERING 
def build_features(recency, frequency, monetary):
    avg_order_val = monetary / frequency if frequency > 0 else 0
    est_age = recency + (frequency * 30)
    
    data = {
        'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary],
        'customer_age_days': [est_age],
        'days_between_purchases': [est_age/frequency if frequency > 0 else 0],
        'orders_last_30d': [1 if recency < 30 else 0],
        'orders_last_90d': [frequency if recency < 90 else 0],
        'avg_order_value': [avg_order_val], 
        'max_order_value': [avg_order_val*1.5],
        'min_order_value': [avg_order_val*0.5], 
        'std_order_value': [avg_order_val*0.2], 'product_diversity': [frequency], 
        'repeat_purchase_rate': [0.5], 'avg_basket_size': [3.0], 
        'total_items_bought': [frequency*5], 'is_weekend': [0], 'hour': [12], 
        'spending_increasing': [1]
    }
    return pd.DataFrame(data, columns=feature_names)

# 6. ENDPOINTS
@app.get("/")
def home():
    return {"status": "online", "model_loaded": model is not None, "api_version": "2.0"}

@app.get("/health")
def health_check():
    """Health check endpoint for dashboard"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database_loaded": not customer_db.empty,
        "ai_configured": bool(GROQ_API_KEY)  # Updated check
    }

@app.post("/predict")
def predict(data: CustomerInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_df = build_features(data.recency, data.frequency, data.monetary)
        prob = float(model.predict_proba(input_df)[0][1])
        return {"churn_probability": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/at-risk")
def get_at_risk(risk_level: str = "HIGH"):
    if customer_db.empty:
        raise HTTPException(status_code=503, detail="Database not loaded")
    
    filtered = customer_db[customer_db['risk_level'] == risk_level]
    top_20 = filtered.sort_values('churn_probability', ascending=False).head(20)
    
    # Ensure column names match what the frontend expects
    return top_20[['Customer ID', 'churn_probability', 'Recency', 'Monetary']].to_dict(orient='records')