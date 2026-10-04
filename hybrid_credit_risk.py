import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# 1. GENERATE HIGH-DIMENSIONAL SIMULATED FINANCIAL RISK SPACE
print("[INFO] Generating high-dimensional consumer credit dataset...")
np.random.seed(42)
n_samples = 5000

# Generating a dataset containing real predictors, collinear metrics, and complete noise
data = {
    'debt_to_income_ratio': np.random.uniform(0.05, 4.5, n_samples),
    'revolving_utilization_wallet': np.random.uniform(0.0, 1.0, n_samples),
    'payment_stringency_score': np.random.randint(0, 100, n_samples),
    'delinquency_events_12m': np.random.poisson(0.3, n_samples),
    'length_of_credit_history': np.random.exponential(6, n_samples),
    'invariant_noise_field': np.ones(n_samples),               # Filter 1 target (Zero Variance)
    'gaussian_noise_field': np.random.normal(0, 5, n_samples),   # Filter 2 target (Low Mutual Info)
    'uniform_noise_field': np.random.uniform(1, 10, n_samples)   # Wrapper target (Low Multi-Model Utility)
}

# Inject explicit multicollinearity (Highly correlated duplicate feature)
data['revolving_util_collinear_dup'] = data['revolving_utilization_wallet'] * 1.02 + np.random.normal(0, 0.005, n_samples)

df = pd.DataFrame(data)

# Derive non-linear risk target mapping to represent realistic default trends
latent_default_risk = (
    df['revolving_utilization_wallet'] * 4.2 
    - df['debt_to_income_ratio'] * 1.6 
    - (df['payment_stringency_score'] / 20.0)
    + df['delinquency_events_12m'] * 2.8
)
# Fix explicit severe class imbalance (15% default profile representation)
df['is_default'] = (latent_default_risk > np.percentile(latent_default_risk, 85)).astype(int)

X = df.drop('is_default', axis=1)
y = df['is_default']

# Split data using stratify to maintain default ratios across domains
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[STAGE 1] Initial dimension baseline matrix shape: {X_train.shape}")

# 2. TWO-TIER HYBRID FEATURE SELECTION PIPELINE
print("\n[STAGE 2] Executing Hybrid Feature Selection Framework...")

# Step A: Filter Phase 1 (Variance Thresholding)
selector_variance = VarianceThreshold(threshold=0.01)
X_train_v = selector_variance.fit_transform(X_train)
X_test_v = selector_variance.transform(X_test)
features_v = X_train.columns[selector_variance.get_support()]
print(f" -> Retained after Variance Thresholding: {list(features_v)}")

# Step B: Filter Phase 2 (Mutual Information Ranking)
mi_coefficients = mutual_info_classif(X_train_v, y_train, random_state=42)
mi_mapping = pd.DataFrame({'Feature': features_v, 'MI': mi_coefficients})
selected_mi_features = mi_mapping.nlargest(6, 'MI')['Feature'].tolist()
print(f" -> Retained after Mutual Information Filtering: {selected_mi_features}")

X_train_mi = pd.DataFrame(X_train_v, columns=features_v)[selected_mi_features]
X_test_mi = pd.DataFrame(X_test_v, columns=features_v)[selected_mi_features]

# Step C: Wrapper Phase (Recursive Feature Elimination via Random Forest)
base_rf = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)
wrapper_rfe = RFE(estimator=base_rf, n_features_to_select=4, step=1)

X_train_final = wrapper_rfe.fit_transform(X_train_mi, y_train)
X_test_final = wrapper_rfe.transform(X_test_mi)
optimized_features = X_train_mi.columns[wrapper_rfe.get_support()].tolist()
print(f" -> Optimized Features Selected by Hybrid Execution: {optimized_features}")

# Multi-model normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

# 3. ENSEMBLE MODEL TRAINING (XGBOOST CLASSIFIER)
print("\n[STAGE 3] Training Hyperparameter-Tuned XGBoost Optimization Engine...")
xgb_engine = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.06,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=5.6,  # Handles class imbalance math smoothly
    random_state=42,
    eval_metric='logloss'
)
xgb_engine.fit(X_train_scaled, y_train)

# 4. QUANTITATIVE PERFORMANCE EVALUATION LOGGING
print("\n=== MODEL EVALUATION PIPELINE REPORT ===")
predictions = xgb_engine.predict(X_test_scaled)
probabilities = xgb_engine.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, predictions))
print(f"Receiver Operating Characteristic (AUROC) Score: {roc_auc_score(y_test, probabilities):.4f}")

print("\n=== CONFUSION MATRIX RESOLUTION LOGS ===")
c_matrix = confusion_matrix(y_test, predictions)
print(f"Confusion Matrix Array Data Output:\n {c_matrix}")
