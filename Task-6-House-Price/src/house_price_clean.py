"""
================================================================================
ADVANCED HOUSE PRICE PREDICTION PIPELINE
================================================================================
Author: Abdul Rafay
Version: Professional ML Pipeline Edition
Date: May 2026

IMPROVEMENTS OVER PREVIOUS VERSION:
[OK] Proper categorical encoding
[OK] Log transformation of target
[OK] Outlier handling
[OK] Pipeline architecture
[OK] ColumnTransformer usage
[OK] Better feature engineering
[OK] Model-specific preprocessing
[OK] Advanced evaluation
[OK] Residual analysis
[OK] Stronger ML engineering practices

CURRENT PERFORMANCE SUMMARY:
- Gradient Boosting is the strongest current model.
- Random Forest is the second-best current model.
- Linear models are kept as baselines and need improvement.

================================================================================
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

warnings.filterwarnings('ignore')

# =============================================================================
# CREATE PROJECT DIRECTORIES
# =============================================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 90)
print("ADVANCED HOUSE PRICE PREDICTION PIPELINE")
print("=" * 90)

# =============================================================================
# STEP 1 - LOAD DATA
# =============================================================================

print("\n[1/10] LOADING DATA")
print("-" * 60)

df = pd.read_csv("data/data.csv")

print(f"Original Dataset Shape: {df.shape}")

# Remove invalid prices
df = df[df["price"] > 0]

print(f"After removing invalid prices: {df.shape}")

# =============================================================================
# STEP 2 - DATA CLEANING & OUTLIER HANDLING
# =============================================================================

print("\n[2/10] DATA CLEANING & OUTLIER HANDLING")
print("-" * 60)

# Remove extreme outliers using IQR

Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

before_outliers = len(df)

df = df[
    (df["price"] >= lower_bound) &
    (df["price"] <= upper_bound)
]

after_outliers = len(df)

print(f"Removed Outliers: {before_outliers - after_outliers}")
print(f"Remaining Samples: {after_outliers}")

# =============================================================================
# STEP 3 - FEATURE ENGINEERING
# =============================================================================

print("\n[3/10] FEATURE ENGINEERING")
print("-" * 60)

# Property age
df["house_age"] = 2026 - df["yr_built"]

# Renovation age
df["renovation_age"] = np.where(
    df["yr_renovated"] == 0,
    0,
    2026 - df["yr_renovated"]
)

# Total rooms
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

# Total square footage
df["total_sqft"] = df["sqft_living"] + df["sqft_basement"]

# Price per room approximation
df["sqft_per_room"] = df["sqft_living"] / (df["total_rooms"] + 1)

# Has renovation
df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)

# Waterfront flag
df["has_waterfront"] = df["waterfront"].astype(int)

# View quality
df["good_view"] = (df["view"] >= 2).astype(int)

print("[OK] Advanced features created")

# =============================================================================
# STEP 4 - TARGET TRANSFORMATION
# =============================================================================

print("\n[4/10] TARGET TRANSFORMATION")
print("-" * 60)

# Log transform target
df["log_price"] = np.log1p(df["price"])

print("[OK] Applied log transformation to target variable")

# =============================================================================
# STEP 5 - EDA VISUALIZATIONS
# =============================================================================

print("\n[5/10] VISUALIZATION & EDA")
print("-" * 60)

# Figure 1 - Price Distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df["price"], bins=50)
plt.title("Original Price Distribution")

plt.subplot(1, 2, 2)
sns.histplot(df["log_price"], bins=50)
plt.title("Log-Transformed Price Distribution")

plt.tight_layout()
plt.savefig("outputs/1_price_distribution.png", dpi=300)
plt.close()

# Figure 2 - Correlation Heatmap
numeric_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(16, 12))

corr_matrix = df[numeric_cols].corr()

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()

plt.savefig("outputs/2_correlation_heatmap.png", dpi=300)
plt.close()

# =============================================================================
# STEP 6 - PREPARE FEATURES
# =============================================================================

print("\n[6/10] PREPARING FEATURES")
print("-" * 60)

target = "log_price"

drop_columns = [
    "price",
    "log_price"
]

X = df.drop(columns=drop_columns)
y = df[target]

# Numerical & categorical columns
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

print(f"Numerical Features: {len(numerical_features)}")
print(f"Categorical Features: {len(categorical_features)}")

# =============================================================================
# STEP 7 - PREPROCESSING PIPELINE
# =============================================================================

print("\n[7/10] BUILDING PREPROCESSING PIPELINE")
print("-" * 60)

# Numerical pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Full preprocessor
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

print("[OK] Professional preprocessing pipeline built")

# =============================================================================
# STEP 8 - TRAIN MODELS
# =============================================================================

print("\n[8/10] TRAINING MODELS")
print("-" * 60)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models dictionary
models = {
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(alpha=0.0001),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
}

results = []

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred_log = pipeline.predict(X_test)

    # Reverse log transform
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    # Metrics
    mae = mean_absolute_error(y_actual, y_pred)

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    r2 = r2_score(y_actual, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "CV Mean": round(cv_scores.mean(), 4)
    })

    trained_models[name] = pipeline

    print(f"   [OK] R2 Score: {r2:.4f}")
    print(f"   [OK] MAE: ${mae:,.0f}")

# =============================================================================
# STEP 9 - RESULTS & VISUALIZATION
# =============================================================================

print("\n[9/10] EVALUATION & VISUALIZATION")
print("-" * 60)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\nMODEL PERFORMANCE:")
print(results_df)

# Save results
results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

# Best model
best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print(f"\nBEST MODEL: {best_model_name}")

# Final predictions
best_pred_log = best_model.predict(X_test)

best_pred = np.expm1(best_pred_log)

actual_prices = np.expm1(y_test)

# Actual vs Predicted Plot
plt.figure(figsize=(8, 8))

plt.scatter(
    actual_prices,
    best_pred,
    alpha=0.5
)

plt.plot(
    [actual_prices.min(), actual_prices.max()],
    [actual_prices.min(), actual_prices.max()],
    'r--',
    linewidth=2
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title(f"{best_model_name} - Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    "outputs/3_actual_vs_predicted.png",
    dpi=300
)

plt.close()

# Residual Plot
residuals = actual_prices - best_pred

plt.figure(figsize=(10, 5))

plt.scatter(best_pred, residuals, alpha=0.5)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")

plt.title("Residual Analysis")

plt.tight_layout()

plt.savefig(
    "outputs/4_residual_analysis.png",
    dpi=300
)

plt.close()

# =============================================================================
# STEP 10 - SAVE ARTIFACTS
# =============================================================================

print("\n[10/10] SAVING ARTIFACTS")
print("-" * 60)

joblib.dump(
    best_model,
    "models/best_house_price_model.pkl"
)

print("[OK] Model saved")

# Save processed feature names
joblib.dump(
    numerical_features,
    "models/numerical_features.pkl"
)

joblib.dump(
    categorical_features,
    "models/categorical_features.pkl"
)

print("[OK] Feature metadata saved")

print("\n" + "=" * 90)
print("[OK] ADVANCED HOUSE PRICE PREDICTION PIPELINE COMPLETED")
print("=" * 90)

print("\nGENERATED FILES:")

print("\nOUTPUTS:")
print("   1_price_distribution.png")
print("   2_correlation_heatmap.png")
print("   3_actual_vs_predicted.png")
print("   4_residual_analysis.png")
print("   model_results.csv")

print("\nMODELS:")
print("   best_house_price_model.pkl")
print("   numerical_features.pkl")
print("   categorical_features.pkl")

print("\nPROFESSIONAL IMPROVEMENTS ACHIEVED:")
print("   [OK] Proper ML pipeline architecture")
print("   [OK] Log target transformation")
print("   [OK] Categorical encoding")
print("   [OK] Outlier handling")
print("   [OK] Residual diagnostics")
print("   [OK] Advanced feature engineering")
print("   [OK] Model comparison")
print("   [OK] Production-style preprocessing")
