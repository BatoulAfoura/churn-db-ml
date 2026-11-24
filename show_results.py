import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://admin:admin123@localhost:5432/churn_db")

# 1) ملخص التنبؤات المخزنة في جدول predictions
summary_query = """
SELECT
    model,
    predicted_churn,
    COUNT(*) AS customer_count,
    ROUND(AVG(churn_probability)::numeric, 4) AS avg_probability
FROM predictions
GROUP BY model, predicted_churn
ORDER BY predicted_churn DESC;
"""
summary_df = pd.read_sql(summary_query, engine)
print("=== Prediction Summary from PostgreSQL ===")
print(summary_df.to_string(index=False))

# 2) عرض أعلى 15 عميلاً من حيث احتمالية المغادرة مع معلوماتهم من جدول customers
top_risk_query = """
SELECT
    p."customerID",
    c."Contract",
    c."InternetService",
    c."tenure",
    c."MonthlyCharges",
    p.actual_churn,
    p.predicted_churn,
    ROUND(p.churn_probability::numeric, 4) AS churn_probability,
    p.model
FROM predictions p
JOIN customers c ON p."customerID" = c."customerID"
ORDER BY p.churn_probability DESC
LIMIT 15;
"""
top_risk_df = pd.read_sql(top_risk_query, engine)
print("\n=== Top 15 High-Risk Customers ===")
print(top_risk_df.to_string(index=False)) 