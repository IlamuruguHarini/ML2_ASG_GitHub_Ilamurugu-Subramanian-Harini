import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load trained model
model = joblib.load("model.joblib")


# Load test dataset (2012)
df_2012 = pd.read_csv("data/day_2012.csv")

# Convert date
df_2012["dteday"] = pd.to_datetime(df_2012["dteday"], dayfirst=True)

# Feature engineering (same as training)
df_2012["month"] = df_2012["dteday"].dt.month
df_2012["weekday"] = df_2012["dteday"].dt.weekday
df_2012["day"] = df_2012["dteday"].dt.day
df_2012.drop(columns=["dteday"], inplace=True)


# Features and target
features = [
    "season", "mnth", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed"
]

target = "cnt"

X_test = df_2012[features]
y_test = df_2012[target]


# Predictions
preds = model.predict(X_test)


# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("Model Evaluation on 2012 Data")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.3f}")


# Simple quality gate (CI/CD check)
assert r2 > 0.75, " Model performance below acceptable threshold"
print(" Model performance check passed")
