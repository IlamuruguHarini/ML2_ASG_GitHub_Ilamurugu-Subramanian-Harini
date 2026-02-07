import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression


# CONFIGURATION

MODEL_PATH = "data/model.joblib"               
DATA_PATH = "data/day_2012.csv"      
TARGET_COL = "cnt"                   


# LOAD EVALUATION DATA

data = pd.read_csv(DATA_PATH)

X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

# LOAD TRAINED MODEL

trained_model = joblib.load(MODEL_PATH)


# EVALUATE TRAINED MODEL

y_pred = trained_model.predict(X)
rmse_model = root_mean_squared_error(y, y_pred)


# BASELINE MODEL

baseline_model = LinearRegression()
baseline_model.fit(X, y)
y_baseline_pred = baseline_model.predict(X)
rmse_baseline = root_mean_squared_error(y, y_baseline_pred)


# QUALITY GATE

threshold = 0.95 * rmse_baseline

print("===== MODEL QUALITY GATE CHECK =====")
print(f"Trained Model RMSE  : {rmse_model:.4f}")
print(f"Baseline Model RMSE : {rmse_baseline:.4f}")
print(f"Quality Threshold  : {threshold:.4f}")

assert rmse_model <= threshold, (
    f"QUALITY GATE FAILED \n"
    f"Model RMSE ({rmse_model:.4f}) is worse than threshold ({threshold:.4f})"
)

print("QUALITY GATE PASSED ")
