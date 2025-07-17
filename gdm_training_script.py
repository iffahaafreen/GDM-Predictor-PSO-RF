pip install pyswarms
pip install imblearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pyswarms.single.global_best import GlobalBestPSO
import pyswarms as ps
from pyswarms.discrete import BinaryPSO
from sklearn.base import clone
# Load dataset
df = pd.read_csv("gdm_oversampled.csv")
print(f"Loaded {df} successfully.")
# Drop rows with missing values
df.dropna(inplace=True)
print("Available columns:\n", df.columns.tolist())
df.info()
# Separate features and target
X = df.drop("GDM_Status", axis=1)
y = df["GDM_Status"]
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# Define objective function
def f_per_particle(m, alpha=0.88):
    total_features = m.shape[0]
    if np.count_nonzero(m) == 0:
        return 1.0  # Penalize particles that select no features
    X_selected = X_train[:, m==1]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_selected, y_train)
    preds = clf.predict(X_selected)
    acc = accuracy_score(y_train, preds)
    return alpha * (1 - acc) + (1 - alpha) * (np.count_nonzero(m) / total_features)

def f(X):
    n_particles = X.shape[0]
    j = [f_per_particle(X[i]) for i in range(n_particles)]
    return np.array(j)
# Initialize PSO
p=30
k=10
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': k,'p': 3}
dimensions = X_train.shape[1]
optimizer = BinaryPSO(n_particles=p, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=20, verbose=True)

# Selected features
selected_features = np.where(pos == 1)[0]
print("Selected features indices:", selected_features)
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]
# --- Random Forest Model ---
model = RandomForestClassifier(n_estimators=300,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    criterion='gini',
    bootstrap=False,
    random_state=42)
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)
y_proba = model.predict_proba(X_test_selected)[:, 1]

# --- Evaluation ---
report = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)

print("Classification Report:")
print(report)
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")

import joblib

# Save the trained model
joblib.dump(model, "GDM_PSO_RF_Model.joblib")

# Optionally save selected features (so you can reuse them)
import json
with open("selected_features.json", "w") as f:
    json.dump(selected_features.tolist(), f)

print("Model and selected features saved.")
