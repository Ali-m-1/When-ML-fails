import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv(r"C:\Users\latit\.local\online_shoppers_intention.csv")

# Target encoding
df["Revenue"] = df["Revenue"].astype(int)

# =========================
# 2. ENCODE CATEGORICAL FEATURES
# =========================
categorical_cols = ["Month", "VisitorType"]

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# =========================
# 3. SPLIT FEATURES / TARGET
# =========================
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. HANDLE CLASS IMBALANCE
# =========================
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos
print("scale_pos_weight =", scale_pos_weight)

# =========================
# 5. XGBOOST MODEL (WITH CORRECTION)
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

# =========================
# 6. TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# 7. PREDICTIONS
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# 8. EVALUATION
# =========================
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== ROC-AUC ===")
print(roc_auc_score(y_test, y_prob))