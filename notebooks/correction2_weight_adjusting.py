import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Chargez vos données de manière robuste depuis le dossier racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'online_shoppers_intention.csv'
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=['PageValues'])
X = df.drop(columns=['Revenue'])
y = df['Revenue']

# 2. Séparation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Pipeline de preprocessing
categorical_features = ['Month', 'VisitorType', 'Weekend']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# 4. Calcul des poids des features
# On doit d'abord "fit" le preprocessor pour connaître les noms des colonnes après OHE
preprocessor.fit(X_train)
encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([encoded_cat_names, X.select_dtypes(include=['number']).columns])

# Initialisation des poids : 1.0 par défaut (importance normale)
weights = np.ones(len(all_feature_names))

# Application d'un poids faible (0.1) à toutes les colonnes contenant "Month"
month_indices = [i for i, name in enumerate(all_feature_names) if 'Month' in name]
weights[month_indices] = 0.1 

print(f"Poids appliqués : {len(month_indices)} features 'Month' ont un poids de 0.1")

# 5. Entraînement avec les poids spécifiques
# On transforme manuellement les données pour injecter les poids dans le classifier
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

clf = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

# C'est ici que l'on limite l'influence :
clf.fit(X_train_transformed, y_train, feature_weights=weights)

# 6. Évaluation
y_pred = clf.predict(X_test_transformed)
y_proba = clf.predict_proba(X_test_transformed)[:, 1]

print(classification_report(y_test, y_pred))
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_proba)
}

print("\n=== COMPREHENSIVE PERFORMANCE METRICS ===")
for name, value in metrics.items():
    print(f"{name:<10}: {value:.4f}")