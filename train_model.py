import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

engine = create_engine("postgresql+psycopg2://admin:admin123@localhost:5432/churn_db")

# 1) سحب البيانات من القاعدة بـ SQL
query = """
SELECT "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
       "tenure", "PhoneService", "InternetService", "Contract",
       "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
       "TotalCharges", "Churn"
FROM customers
"""
df = pd.read_sql(query, engine)

# 2) المعالجة المسبقة
df = df.dropna()
ids = df["customerID"]
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["customerID", "Churn"])
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3) تدريب نموذجين وقياس الأداء
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
best_name, best_f1, best_model = None, -1, None
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    f1 = f1_score(y_test, pred)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1,
    })
    print(name, results[-1])
    if f1 > best_f1:
        best_name, best_f1, best_model = name, f1, model

# 4) حفظ النتائج والتنبؤات في القاعدة
pd.DataFrame(results).to_sql("model_metrics", engine, if_exists="replace", index=False)

proba = best_model.predict_proba(X_test_s)[:, 1]
preds = pd.DataFrame({
    "customerID": id_test.values,
    "actual_churn": y_test.values,
    "predicted_churn": best_model.predict(X_test_s),
    "churn_probability": proba,
    "model": best_name,
})
preds.to_sql("predictions", engine, if_exists="replace", index=False)

print("Best model:", best_name)
print("Saved predictions:", len(preds))