import pytest
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def test_model_performance_vs_dynamic_baseline():
    # 1. LOAD DATA
    # We use 2011 to establish the baseline and 2012 to test the new model
    df_2011 = pd.read_csv("data/Dataset/day_2011.csv") # Original data
    df_2012 = pd.read_csv("data/Dataset/day_2012.csv") # New data
    
    # 2. CALCULATE DYNAMIC BASELINE (No hard-coding)
    # Train a simple LR on 2011 data to get the real-time baseline
    X_base = df_2011.drop(columns=["cnt", "dteday"], errors="ignore")
    y_base = df_2011["cnt"]
    
    base_model = LinearRegression()
    base_model.fit(X_base, y_base)
    base_rmse = np.sqrt(mean_squared_error(y_base, base_model.predict(X_base)))
    
    # 3. TEST RF MODEL
    _, test_df = train_test_split(df_2012, test_size=0.2, random_state=42)
    
    X_test = test_df.drop(columns=["cnt", "dteday"], errors="ignore")
    y_test = test_df["cnt"]
    
    rf_model = joblib.load("data/BikeSharing_RF_Model_2012.joblib")
    current_rmse = np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))
    
    # 4. THE QUALITY GATE (0.95 * Calculated Baseline)
    threshold = base_rmse * 0.95
    
    print(f"\nCalculated Baseline RMSE: {base_rmse:.2f}")
    print(f"Target Threshold (95%): {threshold:.2f}")
    print(f"Model Validation RMSE: {current_rmse:.2f}")

    # 5. ASSERTION
    assert current_rmse <= threshold, f"Model RMSE ({current_rmse:.2f}) is not 5% better than Baseline ({threshold:.2f})"