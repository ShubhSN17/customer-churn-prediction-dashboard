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