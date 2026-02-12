## How to Run this Project

1. **Clone the repo**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/customer-retention-pro.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the Model** (Important!):
   Run the notebook `notebooks/01_churn_analysis.ipynb` to train and save the `xgboost_churn.pkl` file locally.

4. **Run the App**:
   ```bash
   streamlit run app/main.py
   ```