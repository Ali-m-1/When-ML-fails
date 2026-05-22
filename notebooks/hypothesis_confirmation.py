# --- CELLULE 1 : IMPORTATIONS ET CHARGEMENT DES DONNÉES ---
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, recall_score, precision_score, average_precision_score
from xgboost import XGBClassifier

# Chargez vos données de manière robuste depuis le dossier racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'online_shoppers_intention.csv'

df = pd.read_csv(DATA_PATH)

# ADDED: Match your notebook structure by creating X and y first
X = df.drop(columns=['Revenue', 'PageValues'])
y = df['Revenue']

print(f"Total sessions in dataset: {len(df)}")

# --- CELLULE 2 : THE SCIENTIFIC SPLIT (Replaces your original train_test_split) ---
# 1. Define the temporal split based on EDA
early_months = ['Feb', 'Mar', 'May', 'June', 'Jul']
late_months = ['Sep', 'Oct', 'Nov', 'Dec']

# 2. Split X and y using pandas masks
train_mask = X['Month'].isin(early_months)
test_mask = X['Month'].isin(late_months)

X_early = X[train_mask].copy()
y_early = y[train_mask].copy()

X_late_test = X[test_mask].copy()
y_late_test = y[test_mask].copy()

# 3. Create the Control split (In-Distribution) from early months
# We replace your original random split with this one!
RANDOM_STATE = 42
X_train_control, X_test_control, y_train_control, y_test_control = train_test_split(
    X_early, 
    y_early, 
    test_size=0.2, 
    stratify=y_early, 
    random_state=RANDOM_STATE
)

print(f"Training Data (Early Months): {len(X_train_control)} sessions")
print(f"Control Test (Early Months): {len(X_test_control)} sessions")
print(f"Shift Test (Late Months): {len(X_late_test)} sessions")


# --- CELLULE 3 : MODEL PIPELINE ---
# (Using your exact Pipeline from your uploaded notebook)
from sklearn.pipeline import Pipeline

categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            categorical_features
        )
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    ))
])

# --- CELLULE 4 : TRAINING ON EARLY MONTHS ---
model.fit(X_train_control, y_train_control)
print("Model successfully trained on early months (Control Train)!")


# --- CELLULE 5 : EVALUATION AND PROVING THE HYPOTHESIS ---
def evaluate_pipeline(pipeline, X_eval, y_eval, dataset_name):
    y_pred = pipeline.predict(X_eval)
    y_proba = pipeline.predict_proba(X_eval)[:, 1]
    
    recall = recall_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    pr_auc = average_precision_score(y_eval, y_proba)
    
    print(f"--- Results on : {dataset_name} ---")
    print(f"Recall (Minority Class) : {recall:.3f}")
    print(f"Precision             : {precision:.3f}")
    print(f"PR-AUC                : {pr_auc:.3f}\n")
    return recall, precision, pr_auc

# 1. Evaluate on Control Group (In-Distribution)
rec_ctrl, prec_ctrl, prauc_ctrl = evaluate_pipeline(
    model, X_test_control, y_test_control, "CONTROL TEST (Early Months - Random Split)"
)

# 2. Evaluate on Temporal Shift (Out-of-Distribution)
rec_shift, prec_shift, prauc_shift = evaluate_pipeline(
    model, X_late_test, y_late_test, "SHIFT TEST (Late Months - Temporal Split)"
)

# 3. Calculate Generalization Gap
print("=== OBSERVED GENERALIZATION GAP ===")
print(f"Recall Drop    : {(rec_ctrl - rec_shift)*100:.1f} %")
print(f"PR-AUC Drop    : {(prauc_ctrl - prauc_shift)*100:.1f} %")