# app/streamlit_app.py
import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Paths
# --------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"

MODEL_FILE = MODELS_DIR / "rf_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
FEATURES_FILE = MODELS_DIR / "feature_names.pkl"
DF_CLEAN_FILE = DATA_PROCESSED_DIR / "df_clean.csv"

# --------------------
# Load artifacts and data
# --------------------
@st.cache_resource
def load_artifacts():
    model = None
    scaler = None
    feature_names = None
    if MODEL_FILE.exists():
        model = joblib.load(MODEL_FILE)
    if SCALER_FILE.exists():
        scaler = joblib.load(SCALER_FILE)
    if FEATURES_FILE.exists():
        feature_names = joblib.load(FEATURES_FILE)
    return model, scaler, feature_names

@st.cache_data
def load_data():
    if DF_CLEAN_FILE.exists():
        df = pd.read_csv(DF_CLEAN_FILE)
        return df
    else:
        return None

model, scaler, feature_names = load_artifacts()
df = load_data()

# initialize prediction session state
if 'pred_label' not in st.session_state:
    st.session_state['pred_label'] = None
    st.session_state['pred_prob'] = None
    st.session_state['total_charges'] = None
    st.session_state['pred_is_churn'] = None

# --------------------
# Page config and header style
# --------------------
st.set_page_config(page_title="Customer Churn Analysis and Prediction", layout="wide")
st.markdown("<style>body{background-color: #ffffff}</style>", unsafe_allow_html=True)

# Top title
st.markdown(
    """
    <h1 style='
        text-align:center;
        color:#ffffff;
        background: linear-gradient(90deg, #0b3b66, #145DA0);
        padding:20px;
        border-radius:12px;
        margin-bottom:0.75rem;
        font-size:40px;
        font-weight:800;
        letter-spacing:1px;
        box-shadow:0 4px 12px rgba(0,0,0,0.25);
    '>
        Customer Churn Analysis and Prediction
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# --------------------
# Sidebar form for prediction
# --------------------
# (Removed entire sidebar form block)

# --------------------
# Dashboard main area
# --------------------
if df is None:
    st.error('Processed data file not found: data/processed/df_clean.csv. Dashboard visuals require this file.')
else:
    # Basic KPIs
    total_customers = len(df)
    churn_col = 'Churn_Flag' if 'Churn_Flag' in df.columns else ('Churn' if 'Churn' in df.columns else None)
    if churn_col is None:
        churn_count = 0
    else:
        churn_count = df[churn_col].sum()
    churn_pct = churn_count / total_customers * 100 if total_customers > 0 else 0
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['MonthlyCharges'].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label='Total Customers', value=f"{total_customers:,}")
    kpi2.metric(label='Churned %', value=f"{churn_pct:.2f}%")
    kpi3.metric(label='Avg Tenure', value=f"{avg_tenure:.2f}")
    kpi4.metric(label='Avg Monthly Charges', value=f"{avg_monthly:.2f}")

    st.markdown("---")

    # --- Prediction form placed in main area (centered) ---
    col_left, col_center, col_right = st.columns([0.05, 1.9, 0.05])
    with col_center:
        st.markdown('## Predict single customer churn')
        with st.form(key='predict_form_main'):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                Partner = st.selectbox("Partner", ["Yes", "No"], index=1)
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=200, value=12)
                MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], index=1)
                OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"], index=1)
                TechSupport = st.selectbox("Tech Support", ["Yes", "No"], index=1)
                StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"], index=1)
                MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.5)
                Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
            with c2:
                SeniorCitizen_input = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
                SeniorCitizen = 1 if SeniorCitizen_input == "Yes" else 0
                Dependents = st.selectbox("Dependents", ["Yes", "No"], index=1)
                PhoneService = st.selectbox("Phone Service", ["Yes", "No"], index=1)
                OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"], index=1)
                DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"], index=1)
                StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"], index=1)
                PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"], index=1)
                InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=2)
                PaymentMethod = st.selectbox("Payment Method", ["Credit card (automatic)", "Electronic check", "Mailed check"], index=1)

            submit_button = st.form_submit_button(label='Predict churn')

            # If form submitted, compute prediction now so the dashboard panel can show it in the same run
            if submit_button:
                # Build input row similar to training
                def build_row_from_form():
                    row = {c:0 for c in feature_names} if feature_names else { }
                    def set_if(k, v):
                        if feature_names and k in feature_names:
                            row[k] = v
                    if feature_names:
                        set_if('gender', 0 if gender=='Male' else 1)
                        set_if('SeniorCitizen', int(SeniorCitizen))
                        set_if('Partner', 1 if Partner=='Yes' else 0)
                        set_if('Dependents', 1 if Dependents=='Yes' else 0)
                        set_if('tenure', int(tenure))
                        set_if('PhoneService', 1 if PhoneService=='Yes' else 0)
                        set_if('OnlineSecurity', 1 if OnlineSecurity=='Yes' else 0)
                        set_if('OnlineBackup', 1 if OnlineBackup=='Yes' else 0)
                        set_if('DeviceProtection', 1 if DeviceProtection=='Yes' else 0)
                        set_if('TechSupport', 1 if TechSupport=='Yes' else 0)
                        set_if('StreamingTV', 1 if StreamingTV=='Yes' else 0)
                        set_if('StreamingMovies', 1 if StreamingMovies=='Yes' else 0)
                        set_if('PaperlessBilling', 1 if PaperlessBilling=='Yes' else 0)
                        set_if('MonthlyCharges', float(MonthlyCharges))
                        total_charges = float(tenure)*float(MonthlyCharges)
                        set_if('TotalCharges', float(total_charges))
                        ml_key = f"MultipleLines_{MultipleLines}"
                        if ml_key in row:
                            set_if(ml_key, 1)
                        int_key = f"InternetService_{InternetService}"
                        if int_key in row:
                            set_if(int_key,1)
                        c_key = f"Contract_{Contract}"
                        if c_key in row:
                            set_if(c_key,1)
                        pm_key = f"PaymentMethod_{PaymentMethod}"
                        if pm_key in row:
                            set_if(pm_key,1)
                    return pd.DataFrame([row], columns=feature_names) if feature_names else pd.DataFrame()

                if not model or not feature_names:
                    st.error('Model or feature names not found. Cannot predict. Save rf_model.pkl and feature_names.pkl into models/.')
                else:
                    X_user = build_row_from_form()
                    numeric_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges'] if c in X_user.columns]
                    try:
                        if scaler and numeric_cols:
                            X_user[numeric_cols] = scaler.transform(X_user[numeric_cols])
                    except Exception as e:
                        st.warning('Scaler transform failed, proceeding without scaling numeric cols.')
                        st.write(str(e))

                    try:
                        pred = model.predict(X_user)[0]
                        label = 'Likely to Churn' if int(pred)==1 else 'Not Likely to Churn'

                        # store prediction and computed values in session state for display in the dashboard area
                        total_charges = float(tenure) * float(MonthlyCharges)
                        try:
                            prob = None
                            if hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(X_user)[:,1][0]
                        except:
                            prob = None

                        st.session_state['pred_label'] = label
                        st.session_state['pred_prob'] = prob
                        st.session_state['total_charges'] = total_charges
                        st.session_state['pred_is_churn'] = (int(pred) == 1)

                    except Exception as e:
                        st.error('Prediction failed. See console output for details.')
                        st.write(str(e))

    # Row 1 charts
    colA, colB = st.columns([2,1])
    with colA:
        # Churn by Contract Type (grouped bar)
        if 'Contract' in df.columns and churn_col is not None:
            contract_counts = df.groupby(['Contract', churn_col]).size().unstack(fill_value=0)
            # ensure columns 0 and 1 present
            if 0 not in contract_counts.columns:
                contract_counts[0] = 0
            if 1 not in contract_counts.columns:
                contract_counts[1] = 0
            contract_counts = contract_counts[[0,1]]
            labels = contract_counts.index.tolist()
            x = np.arange(len(labels))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(x - width/2, contract_counts[0].values, width, label='No')
            ax.bar(x + width/2, contract_counts[1].values, width, label='Yes')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Count of customers')
            ax.set_title('Churn by Contract Type')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

    with colB:
        pass

    # Row 2 charts
    colC, colD, colE = st.columns([1.2,1,1])
    with colC:
        # Prediction result panel moved to the leftmost side
        if st.session_state.get('pred_label') is not None:
            st.markdown('### Prediction Result')
            prob = st.session_state.get('pred_prob')
            label = st.session_state.get('pred_label')
            is_churn = st.session_state.get('pred_is_churn')
            if is_churn:
                if prob is not None:
                    st.error(f"{label} (prob={prob:.2f})")
                else:
                    st.error(label)
            else:
                if prob is not None:
                    st.success(f"{label} (prob={prob:.2f})")
                else:
                    st.success(label)

            st.markdown('**Computed values:**')
            total = st.session_state.get('total_charges')
            if total is not None:
                st.metric(label='TotalCharges', value=f"{total:,.2f}")
            else:
                st.write('')
        else:
            st.write('')

    with colD:
        pass

    with colE:
        # Retention recommendations box
        st.markdown("### Retention Recommendations")
        st.markdown("1. Offer discounts for month-to-month customers who show high monthly charges.")
        st.markdown("2. Bundle Tech Support for Fiber optic customers.")
        st.markdown("3. Introduce loyalty programs after 6 months.")
        st.markdown("4. Target customers with high MonthlyCharges and low tenure for proactive offers.")

    st.markdown("---")

# --------------------
# EDA Insights (summary from exploratory analysis)
# --------------------
st.header('Key insights from exploratory data analysis (EDA)')

if df is None:
    st.write('No processed data available to show EDA insights.')
else:

    st.markdown('**Summary / Actionable insights:**')
    if 'Contract' in df.columns:
        st.markdown('- Month-to-month customers have a higher churn rate; consider retention offers or longer contract incentives.')
    if 'InternetService' in df.columns:
        st.markdown('- Customers on Fiber optic tend to churn more; review service satisfaction and targeted bundles.')
    st.markdown('- Customers with low tenure and high MonthlyCharges are at higher risk — target them with early retention incentives.')
    if 'PaymentMethod' in df.columns:
        st.markdown('- Customers using Electronic check tend to have higher churn; consider offering automatic payment discounts.')
    if 'SeniorCitizen' in df.columns:
        st.markdown('- Senior citizens show different behavior; ensure plans and communication are senior-friendly and clearly explained.')
