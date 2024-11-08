import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#ostad age ino mikhonid bedonid nesf inaro khodam neveshtam chatgpt error midad hey -_-
df = pd.read_csv("data/cleaned_data.csv", index_col=0, parse_dates=['datetime'])
df_hourly = pd.read_csv("data/df_hourly.csv", index_col=0, parse_dates=True)
df_daily = pd.read_csv("data/df_daily.csv", index_col=0, parse_dates=True)

import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_data.csv", index_col=0, parse_dates=['datetime'])
df_hourly = pd.read_csv("data/df_hourly.csv", index_col=0, parse_dates=True)

target_column = 'Global_active_power'

# Create lag features
def create_lag_features(df, target, lags=24):
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[target].shift(lag)
    df.dropna(inplace=True)  # Drop rows with missing values due to lag
    return df

df_hourly = create_lag_features(df_hourly, target_column, lags=24)

# Prepare features (X) and target (y)
X = df_hourly.drop(columns=[target_column])  # Features: All columns except target
y = df_hourly[target_column]  # Target: The 'global_active_power' column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the XGBoost model
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=5)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)

print(f"XGBoost RMSE: {rmse_xgb}")

# Plot the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred_xgb, label="Predicted", linestyle='--')
plt.legend()
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Global Active Power")
plt.show()