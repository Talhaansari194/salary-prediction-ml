import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# from google.colab import files

# uploaded = files.upload()

df = pd.read_csv("DataScience_salaries_2025.csv")
print("Dataset shape:", df.shape)
print(df.head())


print("\n--- Data Info ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Columns ---")
print(df.columns.tolist())

# ===============================
# EXPLORATORY DATA ANALYSIS
# ===============================

sns.histplot(df['salary_in_usd'], bins=30, kde=True)
plt.title("Salary Distribution")
plt.xlabel("Salary in USD")
plt.ylabel("Frequency")
plt.show()

avg_salary_exp = df.groupby('experience_level')['salary_in_usd'].mean().reset_index()

sns.barplot(x='experience_level', y='salary_in_usd', data=avg_salary_exp)
plt.title("Average Salary by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Average Salary (USD)")
plt.show()

avg_salary_company = df.groupby('company_size')['salary_in_usd'].mean().reset_index()

sns.barplot(x='company_size', y='salary_in_usd', data=avg_salary_company)
plt.title("Average Salary by Company Size")
plt.show()

top_jobs = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10)

top_jobs.plot(kind='bar', figsize=(10,5))
plt.title("Top 10 Highest Paying Job Titles")
plt.ylabel("Average Salary")
plt.show()

numeric_df = df.select_dtypes(include=['int64','float64'])

sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(x='remote_ratio', y='salary_in_usd', data=df)
plt.title("Remote Work Ratio vs Salary")
plt.show()

salary_year = df.groupby('work_year')['salary_in_usd'].mean()

salary_year.plot(marker='o')
plt.title("Average Salary Over Time")
plt.ylabel("Salary in USD")
plt.show()

# ============================================================
# BASELINE PREPROCESSING
# ============================================================
# This is the shared base used by every model section below.
# Improvements are each applied in isolation further down.

features = [
    'experience_level', 'employment_type', 'job_title',
    'employee_residence', 'remote_ratio', 'company_location', 'company_size'
]
 
X_base = pd.get_dummies(df[features], drop_first=True)
y_base = df['salary_in_usd']
 
print(f"\nFeature matrix shape: {X_base.shape}")
 
X_train, X_test, y_train, y_test = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42
)
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
 
print(f"Training samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

# =======================
# BASELINE MODELS
# =======================

# Helper to print cleanly
def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {name:<20}  R² = {r2:.3f}   RMSE = {rmse:.0f}")
    

# Linear Regression
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print_metrics("Linear Regression", y_test, y_pred_lr)

# KNN (k = 5)
knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print_metrics("KNN (k=5)", y_test, y_pred_knn)

# KNN - tune K
print("\n Tuning KNN (testing k=1 to 20)...")
knn_rmse_vals = []
for k in range(1, 21):
    knn_k = KNeighborsRegressor(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    pred_k = knn_k.predict(X_test_scaled)
    knn_rmse_vals.append(np.sqrt(mean_squared_error(y_test, pred_k)))
    
best_k = int(np.argmin(knn_rmse_vals)) + 1
print(f"  Best K = {best_k}")
 
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), knn_rmse_vals, marker='o')
plt.title("KNN Tuning: RMSE vs K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("RMSE")
plt.grid(True)
plt.tight_layout()
plt.show()
 
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_knn_best = knn_best.predict(X_test_scaled)
print_metrics(f"KNN (tuned, k={best_k})", y_test, y_pred_knn_best)

# =============================================================================
# IMPROVEMENT A: LOG TRANSFORM THE TARGET
# =============================================================================
# Salary is right-skewed, meaning that there are more jobs with lower salaries. 
# A log transform makes the target more symmetric, which benefits linear models. 
# Predictions are converted back to dollars with expm1() for evaluation.

print("\n" + "="*75)
print("IMPROVEMENT A: LOG TRANSFORM")
print("="*75)
print(f" Salary skewness (raw): {y_base.skew():.3f}")
print(f" Salary skewness (log): {np.log1p(y_base).skew():.3f}")

y_log = np.log1p(y_base)

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
    X_base, y_log, test_size=0.2, random_state=42
)
scaler_A = StandardScaler()
X_train_A_s = scaler_A.fit_transform(X_train_A)
X_test_A_s = scaler_A.transform(X_test_A)

y_test_A_orig = np.expm1(y_test_A)

lr_A = LinearRegression().fit(X_train_A_s, y_train_A)
print_metrics("LR (log)", y_test_A_orig, np.expm1(lr_A.predict(X_test_A_s)))

knn_A = KNeighborsRegressor(n_neighbors=5).fit(X_train_A_s, y_train_A)
print_metrics("KNN k=5 (log)", y_test_A_orig, np.expm1(knn_A.predict(X_test_A_s)))

# =============================================================================
# IMPROVEMENT B: OUTLIER REMOVAL
# =============================================================================
# Salaries above the 99th percentile (~$379k) are extreme values that
# inflate RMSE and distort model training. We remove them and retrain.

print("\n" + "="*75)
print("IMPROVEMENT B: OUTLIER REMOVAL (cap at 99th percentile)")
print("="*75)

q99 = y_base.quantile(0.99)
mask_B = y_base <= q99
print(f" 99th percentile cutoff: ${q99:,.0f}")
print(f" Rows removed: {(~mask_B).sum()} of {len(y_base)}")

X_B = X_base[mask_B]
y_B = y_base[mask_B]

X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B, y_B, test_size=0.2, random_state=42
)
scaler_B = StandardScaler()
X_train_B_s = scaler_B.fit_transform(X_train_B)
X_test_B_s = scaler_B.transform(X_test_B)

lr_B = LinearRegression().fit(X_train_B_s, y_train_B)
print_metrics("LR (outliers removed)", y_test_B, lr_B.predict(X_test_B_s))
 
knn_B = KNeighborsRegressor(n_neighbors=5).fit(X_train_B_s, y_train_B)
print_metrics("KNN k=5 (outliers removed)", y_test_B, knn_B.predict(X_test_B_s))

# =============================================================================
# IMPROVEMENT C: DECISION TREE
# =============================================================================
# Decision Trees split data on feature thresholds without assuming linearity.
# We test three depth settings to explore the bias-variance tradeoff:
# unconstrained (likely overfit), depth=5, and depth=10.
 
print("\n" + "="*75)
print("IMPROVEMENT C: DECISION TREE")
print("="*75)
 
for depth, label in [(None, "no limit"), (5, "depth=5"), (10, "depth=10")]:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    print_metrics(f"Decision Tree ({label})", y_test, dt.predict(X_test))

# =============================================================================
# IMPROVEMENT D: RANDOM FOREST
# =============================================================================
# Random Forest builds 100 independent decision trees on random subsets of
# data and features, then averages their predictions. This reduces the
# overfitting of a single tree. Also outputs feature importances.
 
print("\n" + "="*75)
print("IMPROVEMENT D: RANDOM FOREST")
print("="*75)
 
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print_metrics("Random Forest (100 trees)", y_test, rf.predict(X_test))
 
# Feature importances
importances = pd.Series(rf.feature_importances_, index=X_base.columns)
top15 = importances.sort_values(ascending=False).head(15)
 
print("\n  Top 15 features by importance:")
for feat, imp in top15.items():
    print(f"    {feat:<45} {imp:.4f}")
 
plt.figure(figsize=(8, 5))
top15.sort_values().plot(kind='barh', color='steelblue')
plt.title("Random Forest — Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# =============================================================================
# IMPROVEMENT E: GRADIENT BOOSTING
# =============================================================================
# Gradient Boosting builds trees sequentially: each new tree corrects the
# residual errors of the previous one. Strong on tabular data but slower
# to train than Random Forest.
 
print("\n" + "="*75)
print("IMPROVEMENT E: GRADIENT BOOSTING")
print("="*75)
 
gb = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb.fit(X_train, y_train)
print_metrics("Gradient Boosting", y_test, gb.predict(X_test))

# =============================================================================
# IMPROVEMENT F: FEATURE SELECTION
# =============================================================================
# Using Random Forest importances, we select the top 10 features and
# retrain both Linear Regression and Random Forest on this leaner set.
 
print("\n" + "="*75)
print("IMPROVEMENT F: FEATURE SELECTION (top 10 by RF importance)")
print("="*75)
 
top10_features = importances.sort_values(ascending=False).head(10).index
print(f"  Selected features: {list(top10_features)}")
print(f"  Reduced from {X_base.shape[1]} columns to {len(top10_features)}")
 
X_train_G = X_train[top10_features]
X_test_G  = X_test[top10_features]
 
scaler_G = StandardScaler()
X_train_G_s = scaler_G.fit_transform(X_train_G)
X_test_G_s  = scaler_G.transform(X_test_G)
 
lr_G = LinearRegression().fit(X_train_G_s, y_train)
print_metrics("LR (top 10 features)", y_test, lr_G.predict(X_test_G_s))
 
rf_G = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_G.fit(X_train_G, y_train)
print_metrics("RF (top 10 features)", y_test, rf_G.predict(X_test_G))
