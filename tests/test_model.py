# # # # import joblib
# # # # import pandas as pd
# # # # from sklearn.metrics import mean_squared_error, root_mean_squared_error
# # # # from sklearn.linear_model import LinearRegression


# # # # # CONFIGURATION

# # # # MODEL_PATH = "data/model.joblib"               
# # # # DATA_PATH = "data/day_2012.csv"      
# # # # TARGET_COL = "cnt"                   


# # # # # LOAD EVALUATION DATA

# # # # data = pd.read_csv(DATA_PATH)

# # # # cols_to_drop = [TARGET_COL, 'dteday'] 
# # # # X = data.drop(columns=cols_to_drop)
# # # # y = data[TARGET_COL]

# # # # # LOAD TRAINED MODEL

# # # # trained_model = joblib.load(MODEL_PATH)


# # # # # EVALUATE TRAINED MODEL

# # # # y_pred = trained_model.predict(X)
# # # # rmse_model = root_mean_squared_error(y, y_pred)


# # # # # BASELINE MODEL

# # # # baseline_model = LinearRegression()
# # # # baseline_model.fit(X, y)
# # # # y_baseline_pred = baseline_model.predict(X)
# # # # rmse_baseline = root_mean_squared_error(y, y_baseline_pred)


# # # # # QUALITY GATE

# # # # threshold = 0.95 * rmse_baseline

# # # # print("===== MODEL QUALITY GATE CHECK =====")
# # # # print(f"Trained Model RMSE  : {rmse_model:.4f}")
# # # # print(f"Baseline Model RMSE : {rmse_baseline:.4f}")
# # # # print(f"Quality Threshold  : {threshold:.4f}")

# # # # assert rmse_model <= threshold, (
# # # #     f"QUALITY GATE FAILED \n"
# # # #     f"Model RMSE ({rmse_model:.4f}) is worse than threshold ({threshold:.4f})"
# # # # )

# # # # print("QUALITY GATE PASSED ")


# # import joblib
# # import pandas as pd
# # import numpy as np
# # from sklearn.metrics import root_mean_squared_error
# # from sklearn.linear_model import LinearRegression
# # from sklearn.model_selection import train_test_split

# # # CONFIGURATION
# # MODEL_PATH = "data/model.joblib" 
# # DATA_PATH = "data/day_2012.csv"
# # TARGET_COL = "cnt"

# # # 1. LOAD AND PREPROCESS DATA EXACTLY LIKE THE NOTEBOOK
# # data = pd.read_csv(DATA_PATH)

# # # Replicate notebook preprocessing (Date extraction)
# # data['dteday'] = pd.to_datetime(data['dteday'], dayfirst=True)
# # data['month'] = data['dteday'].dt.month
# # data['weekday'] = data['dteday'].dt.weekday
# # data['day'] = data['dteday'].dt.day

# # # Match feature selection from notebook
# # features = [
# #     'season', 'mnth', 'holiday', 'weekday', 'workingday',
# #     'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
# # ]
# # X = data[features]
# # y = data[TARGET_COL]

# # # 2. REPLICATE THE SPLIT (80/20, Random State 42)
# # X_train_2012, X_test_2012, y_train_2012, y_test_2012 = train_test_split(
# #     X, y, test_size=0.2, random_state=42
# # )

# # # 3. EVALUATE TRAINED MODEL ON TEST SET
# # trained_model = joblib.load(MODEL_PATH)
# # y_pred = trained_model.predict(X_test_2012)
# # rmse_model = root_mean_squared_error(y_test_2012, y_pred)

# # # 4. REPLICATE BASELINE LOGIC (LR on 50% of training data)
# # X_train_subset, _, y_train_subset, _ = train_test_split(
# #     X_train_2012, y_train_2012, train_size=0.5, random_state=42
# # )
# # baseline_model = LinearRegression()
# # baseline_model.fit(X_train_subset, y_train_subset)

# # # Evaluate baseline on the same test set
# # y_baseline_pred = baseline_model.predict(X_test_2012)
# # rmse_baseline = root_mean_squared_error(y_test_2012, baseline_model.predict(X_test_2012))

# # # 5. QUALITY GATE
# # threshold = 0.95 * rmse_baseline

# # print("===== MODEL QUALITY GATE CHECK =====")
# # print(f"Trained Model RMSE  : {rmse_model:.4f}")
# # print(f"Baseline Model RMSE : {rmse_baseline:.4f}")
# # print(f"Quality Threshold   : {threshold:.4f}")

# # assert rmse_model <= threshold, (
# #     f"QUALITY GATE FAILED \n"
# #     f"Model RMSE ({rmse_model:.4f}) is worse than threshold ({threshold:.4f})"
# # )

# # print("QUALITY GATE PASSED")

# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# # =====================
# # CONFIGURATION
# # =====================
# MODEL_PATH = "data/model.joblib"
# DATA_PATH = "data/day_2012.csv"
# TARGET_COL = "cnt"

# # =====================
# # 1. LOAD AND PREPROCESS DATA (MATCH NOTEBOOK)
# # =====================
# data = pd.read_csv(DATA_PATH)

# # Replicate notebook preprocessing
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

# # =====================
# # 2. REPLICATE TRAIN / TEST SPLIT (80 / 20)
# # =====================
# X_train_2012, X_test_2012, y_train_2012, y_test_2012 = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )

# # =====================
# # 3. EVALUATE TRAINED MODEL ON TEST SET
# # =====================
# trained_model = joblib.load(MODEL_PATH)

# y_pred = trained_model.predict(X_test_2012)
# rmse_model = np.sqrt(mean_squared_error(y_test_2012, y_pred))

# # =====================
# # 4. REPLICATE BASELINE LOGIC (MLflow: LR on 50% of training data)
# # =====================
# X_train_subset, _, y_train_subset, _ = train_test_split(
#     X_train_2012,
#     y_train_2012,
#     train_size=0.5,
#     random_state=42
# )

# baseline_model = LinearRegression()
# baseline_model.fit(X_train_subset, y_train_subset)

# # Evaluate baseline on SAME test set
# y_baseline_pred = baseline_model.predict(X_test_2012)
# rmse_baseline = np.sqrt(mean_squared_error(y_test_2012, y_baseline_pred))

# # =====================
# # 5. QUALITY GATE
# # =====================
# threshold = 0.95 * rmse_baseline

# print("===== MODEL QUALITY GATE CHECK =====")
# print(f"Trained Model RMSE  : {rmse_model:.4f}")
# print(f"Baseline Model RMSE : {rmse_baseline:.4f}")
# print(f"Quality Threshold   : {threshold:.4f}")

# assert rmse_model <= threshold, (
#     f"QUALITY GATE FAILED\n"
#     f"Model RMSE ({rmse_model:.4f}) is worse than threshold ({threshold:.4f})"
# )

# print("QUALITY GATE PASSED ")

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# =====================
# CONFIGURATION
# =====================

MODEL_PATH = "data/model.joblib"

# Use the SAME data as notebook baseline
DATA_PATH = "data/day_2011.csv"   # <-- change to your 2011 file name
TARGET_COL = "cnt"

# =====================
# 1. LOAD AND PREPROCESS DATA (MATCH NOTEBOOK)
# =====================

data = pd.read_csv(DATA_PATH)

# Notebook preprocessing: derive date parts, but baseline LR actually uses:
# season, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed
data["dteday"] = pd.to_datetime(data["dteday"], dayfirst=True)
data["month"] = data["dteday"].dt.month
data["weekday"] = data["dteday"].dt.weekday
data["day"] = data["dteday"].dt.day

features = [
    "season",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

X = data[features]
y = data[TARGET_COL]

# =====================
# 2. TRAIN / TEST SPLIT (80 / 20, RS=42) – SAME AS NOTEBOOK
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# 3. EVALUATE TRAINED RF MODEL ON TEST SET
# =====================

trained_model = joblib.load(MODEL_PATH)
y_pred = trained_model.predict(X_test)
rmse_model = np.sqrt(mean_squared_error(y_test, y_pred))

# =====================
# 4. BASELINE: LR ON 50% OF TRAINING DATA (MATCH NOTEBOOK "WORSE" RUN)
# =====================

X_train_subset, _, y_train_subset, _ = train_test_split(
    X_train, y_train, train_size=0.5, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(X_train_subset, y_train_subset)

y_baseline_pred = baseline_model.predict(X_test)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_baseline_pred))

# =====================
# 5. QUALITY GATE
# =====================

threshold = 0.95 * rmse_baseline

print("===== MODEL QUALITY GATE CHECK =====")
print(f"Trained Model RMSE : {rmse_model:.4f}")
print(f"Baseline Model RMSE : {rmse_baseline:.4f}")
print(f"Quality Threshold : {threshold:.4f}")

assert rmse_model <= threshold, (
    "QUALITY GATE FAILED\n"
    f"Model RMSE ({rmse_model:.4f}) is worse than threshold ({threshold:.4f})"
)

print("QUALITY GATE PASSED")
