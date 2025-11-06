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