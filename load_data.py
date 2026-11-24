import pandas as pd
from sqlalchemy import create_engine

# الاتصال بقاعدة PostgreSQL (نفس بيانات docker-compose.yml)
engine = create_engine("postgresql+psycopg2://admin:admin123@localhost:5432/churn_db")

# قراءة الـ CSV (مرة وحدة فقط لتحميل البيانات)
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# TotalCharges فيها قيم فاضية نصية، نحوّلها لأرقام
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# تحميل البيانات لجدول customers
df.to_sql("customers", engine, if_exists="replace", index=False)

print("Loaded rows:", len(df))