This project focuses on a telecom company aiming to understand and reduce customer churn using data-driven insights.

Customer churn is a major challenge for subscription-based businesses like telecom companies. Losing existing customers is often more expensive than acquiring new ones, making churn prediction a critical business goal.

The objective of this project is to analyze customer behavior and build a machine learning model that predicts the likelihood of a customer leaving the company (churning). By identifying at-risk customers early, businesses can take proactive actions — such as offering discounts, improving service quality, or personalizing plans — to improve customer retention.

This project includes the complete end-to-end process:
	•	Data exploration and visualization
	•	Data preprocessing and feature engineering
	•	Machine learning model development and evaluation
	•	Model explainability (SHAP-based insights)
	•	Deployment of an interactive Streamlit dashboard for real-time churn prediction and actionable insights

Download dataset from Kaggle:
https://www.kaggle.com/blastchar/telco-customer-churn
Place it in data/raw/telco_churn.csv

EDA Observations:

    From the boxplots, we observe that shorter tenure and higher monthly charges are major drivers of customer churn.
    Customers who churn typically have been with the company for less than a year, pay higher monthly rates, and therefore have lower total lifetime spending.
    These insights suggest that early retention efforts and pricing optimization could significantly reduce churn rates.

    We analyzed churn rates across key categorical features.
    Customers on month-to-month contracts and those paying via electronic check have the highest churn rates, while longer contracts and automated payment methods significantly reduce churn.
    Internet service type (Fiber) also shows higher churn compared to DSL users.

Fix's:

    After mapping Yes/No values we found NaNs from “No internet service” values; these were set to 0 (not subscribed)

### 🔧 Data Preprocessing
- Removed unwanted columns (`customerID`)
- Handled missing values in `TotalCharges`
- Encoded categorical variables using Label and One-Hot Encoding
- Replaced 'No internet service' entries with 'No'
- Scaled numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`
- Saved the scaler as `scaler.pkl` in the `/models` directory for future use


### Baseline Model — Logistic Regression
- **Accuracy:** 81%
- **Precision:** 0.67
- **Recall:** 0.55
- **F1-Score:** 0.60
- **Interpretation:**
  - The model performs well overall.
  - It correctly identifies 55% of churners.
  - A good starting point; we’ll aim to improve recall with ensemble models.


### 🪵 Model 2: Random Forest Classifier

| Metric | Value |
|--------|--------|
| Accuracy | 0.79 |
| Precision | 0.65 |
| Recall | 0.48 |
| F1-Score | 0.55 |
**Insights:**
- The Random Forest slightly underperformed compared to Logistic Regression.
- It has a strong ability to identify non-churners (Recall 0.91 for class 0) but missed some churners.
- Next step: tune hyperparameters and test XGBoost to improve recall.

“Note: ROC-AUC metric and probability-based threshold tuning were skipped intentionally to focus on core classification metrics — Accuracy, Precision, Recall, and F1-Score.”


### 🌲 Tuned Random Forest Classifier

| Metric | Value |
|--------|--------|
| Accuracy | **0.73** |
| Precision | **0.49** |
| Recall | **0.80** |
| F1-Score | **0.61** |

**Insights:**
- Recall improved from **0.48 → 0.80** after tuning — model now catches 8 out of 10 churners.
- Precision dropped slightly, which is acceptable for churn prediction.
- Final model shows a strong balance between business usefulness and model performance.
# 📊 Customer Churn Prediction Dashboard
### 📡 Telecom Industry — Machine Learning | End-to-End Project

This project presents a fully developed **Customer Churn Prediction System** for a telecom company, built with a complete **machine learning pipeline**, **EDA insights**, and an interactive **Streamlit dashboard**. The goal is to identify customers likely to churn so the business can take preventative actions like offering discounts, improving plan recommendations, or enhancing customer service.

---

## 🚀 Project Highlights
- Full end‑to‑end ML workflow (EDA → Preprocessing → Modeling → Deployment)
- Interactive **Streamlit App** for real‑time churn prediction
- Business‑oriented insights for decision making
- Tuned Random Forest model optimized for **high recall** (catches most churners)
- Clean and professional UX with actionable retention recommendations

---

## 📦 Dataset
**Source:** Kaggle Telco Customer Churn Dataset  
🔗 https://www.kaggle.com/blastchar/telco-customer-churn

Place the dataset in:
```
data/raw/telco_churn.csv
```

---

## 🧼 Data Preprocessing
Key data cleaning & preparation steps:
- Removed `customerID` column
- Handled missing values in `TotalCharges`
- Replaced "No internet service" & "No phone service" with "No" for consistency
- Encoded categorical features using Label Encoding + One‑Hot Encoding
- Scaled numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`
- Saved required deployment artifacts:
  - `rf_model.pkl`
  - `scaler.pkl`
  - `feature_names.pkl`

---

## 📊 Exploratory Data Analysis — Key Insights
### 🔍 Customer Behavior
- Customers with **short tenure** churn significantly more.
- Higher **monthly charges** correlate strongly with churn.
- Customers who churn have **lower lifetime value**.

### 📄 Categorical Insights
- **Month-to-month contract** users churn the most.
- **Electronic Check** payment method is associated with high churn.
- **Fiber Optic** internet users churn more than DSL customers.

### 💡 Business Insights (Actionable)
- Offer retention incentives to month‑to‑month customers.
- Encourage automatic payment methods to reduce churn.
- Improve service quality and plans for Fiber customers.
- High‑risk segment: **Low tenure + High monthly charges**.
- Electronic check users respond well to discounts or auto‑pay benefits.
- Senior citizens benefit from clear communication and simplified plans.

---

## 🤖 Model Training & Evaluation
### **1️⃣ Logistic Regression (Baseline)**
- Accuracy: **81%**
- Precision: 0.67
- Recall: 0.55
- F1‑Score: 0.60

### **2️⃣ Random Forest (Before Tuning)**
- Accuracy: **79%**
- Recall: 0.48
- Notes: Great at catching non‑churners but weaker at detecting churners.

### **3️⃣ Tuned Random Forest (Final Model)**
- Accuracy: **73%**
- Precision: 0.49
- Recall: **0.80**  
- F1‑Score: 0.61

**Why this model?**  
Churn prediction is recall‑sensitive — catching **8 out of 10 churners** is more valuable than maximizing precision.

---

## 🖥️ Streamlit App Features
### 🎛️ Real‑Time Prediction
- Input customer information
- App computes **TotalCharges** automatically
- Shows churn prediction + probability

### 📈 Dashboard Metrics
- Total customers
- Churn percentage
- Average tenure
- Average monthly charges

### 📄 EDA Insight Summary
- Text‑only insights (no graphs)
- Focused on business understanding

### 🎯 Recommendations
- Action items based on model & EDA

---

## 📂 Project Structure
```
customer-churn-prediction-dashboard/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used
- Python
- Pandas / NumPy
- Scikit‑learn
- Matplotlib
- Streamlit
- Joblib

---

## ▶️ Running the App
### 1) Install requirements
```
pip install -r requirements.txt
```

### 2) Launch the dashboard
```
streamlit run app/streamlit_app.py
```

---

## 📌 Summary
This project delivers:
- End‑to‑end ML engineering experience
- Real‑world telecom churn prediction
- Business‑driven insights
- Professional deployment‑ready dashboard
