import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import recall_score, precision_score, average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from pathlib import Path

# Chargez vos données de manière robuste depuis le dossier racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'online_shoppers_intention.csv'
df = pd.read_csv(DATA_PATH)

# 2. Apply the Correction (Drop Leakage AND Shortcut)
X_corrected = df.drop(columns=['Revenue', 'PageValues', 'Month'])
y_corrected = df['Revenue']

# 3. Standard Random Split (IID)
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X_corrected, 
    y_corrected, 
    test_size=0.2, 
    stratify=y_corrected, 
    random_state=RANDOM_STATE
)

# 4. Build the Pipeline
categorical_features = X_corrected.select_dtypes(include=['object', 'bool']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# Using your exact baseline parameters for a fair scientific comparison
model_corrected = Pipeline([
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

# 5. Train the Model
print("Training corrected model (No Month feature)...")
model_corrected.fit(X_train, y_train)

# 6. Evaluate
y_pred = model_corrected.predict(X_test)
y_proba = model_corrected.predict_proba(X_test)[:, 1]

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_proba)

print("\n=== RESULTS: CORRECTED MODEL ===")
print(f"Recall (Minority Class) : {recall:.3f}")
print(f"Precision             : {precision:.3f}")
print(f"PR-AUC                : {pr_auc:.3f}")

# 7. Check New Feature Importances
ohe = model_corrected.named_steps['preprocessing'].named_transformers_['cat']
encoded_cats = ohe.get_feature_names_out(categorical_features)
numericals = X_corrected.select_dtypes(include=['int64', 'float64']).columns.tolist()

all_features = np.concatenate([encoded_cats, numericals])
importances = model_corrected.named_steps['classifier'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n=== NEW TOP 5 FEATURES ===")
print(importance_df.head(5))