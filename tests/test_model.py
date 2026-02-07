# # import joblib
# # import pandas as pd
# # from sklearn.metrics import mean_squared_error, root_mean_squared_error
# # from sklearn.linear_model import LinearRegression


# # # CONFIGURATION

# # MODEL_PATH = "data/model.joblib"               
# # DATA_PATH = "data/day_2012.csv"      
# # TARGET_COL = "cnt"                   


# # # LOAD EVALUATION DATA

# # data = pd.read_csv(DATA_PATH)

# # cols_to_drop = [TARGET_COL, 'dteday'] 
# # X = data.drop(columns=cols_to_drop)
# # y = data[TARGET_COL]

# # # LOAD TRAINED MODEL

# # trained_model = joblib.load(MODEL_PATH)


# # # EVALUATE TRAINED MODEL

# # y_pred = trained_model.predict(X)
# # rmse_model = root_mean_squared_error(y, y_pred)


# # # BASELINE MODEL

# # baseline_model = LinearRegression()
# # baseline_model.fit(X, y)
# # y_baseline_pred = baseline_model.predict(X)
# # rmse_baseline = root_mean_squared_error(y, y_baseline_pred)


# # # QUALITY GATE

# # threshold = 0.95 * rmse_baseline

# # print("===== MODEL QUALITY GATE CHECK =====")
# # print(f"Trained Model RMSE  : {rmse_model:.4f}")
# # print(f"Baseline Model RMSE : {rmse_baseline:.4f}")
# # print(f"Quality Threshold  : {threshold:.4f}")

# # assert rmse_model <= threshold, (
# #     f"QUALITY GATE FAILED \n"
# #     f"Model RMSE ({rmse_model:.4f}) is worse than threshold ({threshold:.4f})"
# # )

# # print("QUALITY GATE PASSED ")


# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.metrics import root_mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# # CONFIGURATION
# MODEL_PATH = "data/model.joblib" 
# DATA_PATH = "data/day_2012.csv"
# TARGET_COL = "cnt"

# # 1. LOAD AND PREPROCESS DATA EXACTLY LIKE THE NOTEBOOK
# data = pd.read_csv(DATA_PATH)

# # Replicate notebook preprocessing (Date extraction)
# data['dteday'] = pd.to_datetime(data['dteday'], dayfirst=True)
# data['month'] = data['dteday'].dt.month
# data['weekday'] = data['dteday'].dt.weekday
# data['day'] = data['dteday'].dt.day

# # Match feature selection from notebook
# features = [
#     'season', 'mnth', 'holiday', 'weekday', 'workingday',
#     'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
# ]
# X = data[features]
# y = data[TARGET_COL]

# # 2. REPLICATE THE SPLIT (80/20, Random State 42)
# X_train_2012, X_test_2012, y_train_2012, y_test_2012 = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 3. EVALUATE TRAINED MODEL ON TEST SET
# trained_model = joblib.load(MODEL_PATH)
# y_pred = trained_model.predict(X_test_2012)
# rmse_model = root_mean_squared_error(y_test_2012, y_pred)

# # 4. REPLICATE BASELINE LOGIC (LR on 50% of training data)
# X_train_subset, _, y_train_subset, _ = train_test_split(
#     X_train_2012, y_train_2012, train_size=0.5, random_state=42
# )
# baseline_model = LinearRegression()
# baseline_model.fit(X_train_subset, y_train_subset)

# # Evaluate baseline on the same test set
# y_baseline_pred = baseline_model.predict(X_test_2012)
# rmse_baseline = root_mean_squared_error(y_test_2012, y_baseline_pred)

# # 5. QUALITY GATE
# threshold = 0.95 * rmse_baseline

# print("===== MODEL QUALITY GATE CHECK =====")
# print(f"Trained Model RMSE  : {rmse_model:.4f}")
# print(f"Baseline Model RMSE : {rmse_baseline:.4f}")
# print(f"Quality Threshold   : {threshold:.4f}")

# assert rmse_model <= threshold, (
#     f"QUALITY GATE FAILED \n"
#     f"Model RMSE ({rmse_model:.4f}) is worse than threshold ({threshold:.4f})"
# )

# print("QUALITY GATE PASSED")

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
MODEL_PATH = "data/model.joblib" 
DATA_PATH = "data/day_2012.csv"
TARGET_COL = "cnt"

# 1. LOAD DATA
data = pd.read_csv(DATA_PATH)

# 2. FEATURE ENGINEERING (Must match your training pipeline)
# Ensure columns used in training are recreated here
if 'dteday' in data.columns:
    data['dteday'] = pd.to_datetime(data['dteday'], dayfirst=True)
    data['mnth'] = data['dteday'].dt.month
    data['weekday'] = data['dteday'].dt.weekday

# Define the exact features used in your MLflow run
features = [
    'season', 'mnth', 'holiday', 'weekday', 'workingday',
    'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
]
X = data[features]
y = data[TARGET_COL]

# 3. SPLIT DATA (Replicate the 80/20 split used in training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. DYNAMIC BASELINE CALCULATION
# This replicates your "Linear_Regression_Worse_Performance" run logic:
# It takes 50% of the training partition to train the baseline.
X_train_subset, _, y_train_subset, _ = train_test_split(
    X_train, y_train, train_size=0.5, random_state=42
)

baseline_lr = LinearRegression()
baseline_lr.fit(X_train_subset, y_train_subset)

# Evaluate baseline on the test set
y_baseline_pred = baseline_lr.predict(X_test)
rmse_baseline = root_mean_squared_error(y_test, y_baseline_pred)

# 5. EVALUATE YOUR SAVED MODEL
trained_model = joblib.load(MODEL_PATH)
y_model_pred = trained_model.predict(X_test)
rmse_model = root_mean_squared_error(y_test, y_model_pred)

# 6. QUALITY GATE
# Threshold is 5% better than the dynamic baseline
threshold = 0.95 * rmse_baseline

print("---")
print(f"Dynamic Baseline RMSE : {rmse_baseline:.4f}")
print(f"Trained Model RMSE    : {rmse_model:.4f}")
print(f"Target Threshold      : {threshold:.4f}")
print("---")

assert rmse_model <= threshold, (
    f"QUALITY GATE FAILED: Model RMSE ({rmse_model:.2f}) "
    f"is not at least 5% better than Baseline RMSE ({rmse_baseline:.2f})"
)

print("QUALITY GATE PASSED: Model outperformed the dynamic baseline.")