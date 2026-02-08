import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# CONFIGURATION

MODEL_PATH = "data/model.joblib"      
BASELINE_PATH = "data/day_2011.csv"   
TARGET_COL = "cnt"


# 1. BASELINE DATA (2011) FOR BASELINE RMSE

baseline_data = pd.read_csv(BASELINE_PATH)
baseline_data["dteday"] = pd.to_datetime(baseline_data["dteday"], dayfirst=True)
baseline_data["month"] = baseline_data["dteday"].dt.month
baseline_data["weekday"] = baseline_data["dteday"].dt.weekday
baseline_data["day"] = baseline_data["dteday"].dt.day

features = ["season", "mnth", "holiday", "weekday", "workingday", 
            "weathersit", "temp", "atemp", "hum", "windspeed"]

X_baseline = baseline_data[features]
y_baseline = baseline_data[TARGET_COL]

# 2011 baseline split 
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline, y_baseline, test_size=0.2, random_state=42
)


# 2. BASELINE: LR on 50% of 2011 training data

X_train_subset, _, y_train_subset, _ = train_test_split(
    X_train_base, y_train_base, train_size=0.5, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(X_train_subset, y_train_subset)
rmse_baseline = np.sqrt(mean_squared_error(y_test_base, baseline_model.predict(X_test_base)))


# 3. LOAD TRAINED MODEL trained on 2012

trained_model = joblib.load(MODEL_PATH)


# 4. EVALUATE TRAINED MODEL on 2012 data 

data_2012 = pd.read_csv("data/day_2012.csv")  
data_2012["dteday"] = pd.to_datetime(data_2012["dteday"], dayfirst=True)
data_2012["month"] = data_2012["dteday"].dt.month
data_2012["weekday"] = data_2012["dteday"].dt.weekday
data_2012["day"] = data_2012["dteday"].dt.day

X_2012 = data_2012[features]
y_2012 = data_2012[TARGET_COL]

X_train_2012, X_test_2012, y_train_2012, y_test_2012 = train_test_split(
    X_2012, y_2012, test_size=0.2, random_state=42
)

y_pred = trained_model.predict(X_test_2012)
rmse_model = np.sqrt(mean_squared_error(y_test_2012, y_pred))


# 5. QUALITY GATE

threshold = rmse_baseline

print("===== MODEL QUALITY GATE CHECK =====")
print(f"Trained Model RMSE (2012): {rmse_model:.4f}")
print(f"Baseline Model RMSE (2011): {rmse_baseline:.4f}")
print(f"Quality Threshold: {threshold:.4f}")

assert rmse_model <= threshold, (
    "QUALITY GATE FAILED\n"
    f"Model RMSE ({rmse_model:.4f}) > threshold ({threshold:.4f})"
)

print("QUALITY GATE PASSED")
