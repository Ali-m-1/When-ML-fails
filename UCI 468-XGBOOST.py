import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv(r"C:\Users\latit\.local\online_shoppers_intention.csv")

# =========================
# 2. Basic preprocessing
# =========================

# Target variable: Revenue (True/False -> 1/0)
df["Revenue"] = df["Revenue"].astype(int)

# Encode categorical variables
categorical_cols = ["Month", "VisitorType"]

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# =========================
# 3. Features / target split
# =========================
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# =========================
# 4. Train-test split
# IMPORTANT: stratify for imbalance
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. XGBoost model
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# =========================
# 6. Train
# =========================
model.fit(X_train, y_train)

# =========================
# 7. Predictions
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# 8. Evaluation
# =========================
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== ROC-AUC ===")
print(roc_auc_score(y_test, y_prob))

