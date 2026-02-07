import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# ----------------------------
# CONFIG (easy to tweak for pass/fail)
# ----------------------------
MODEL_PATH = "data/model.joblib"
DATA_PATH = "data/day_2012.csv"
TARGET_COL = "count"          # CHANGE if your target column name differs

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ----------------------------
# LOAD TRAINED MODEL (Task 1)
# ----------------------------
model = joblib.load(MODEL_PATH)

# ----------------------------
# MODEL PERFORMANCE
# ----------------------------
y_pred = model.predict(X)
rmse_model = mean_squared_error(y, y_pred, squared=False)

# ----------------------------
# BASELINE MODEL (Simple Linear Regression)
# ----------------------------
baseline = LinearRegression()
baseline.fit(X, y)
y_base_pred = baseline.predict(X)
rmse_baseline = mean_squared_error(y, y_base_pred, squared=False)

print(f"Model RMSE: {rmse_model:.4f}")
print(f"Baseline RMSE: {rmse_baseline:.4f}")

# ----------------------------
# QUALITY GATE
# ----------------------------
threshold = 0.95 * rmse_baseline

print(f"Quality Gate Threshold: {threshold:.4f}")

assert rmse_model <= threshold, (
    f"❌ Quality Gate FAILED: RMSE {rmse_model:.4f} "
    f"is worse than threshold {threshold:.4f}"
)

print("✅ Quality Gate PASSED")
