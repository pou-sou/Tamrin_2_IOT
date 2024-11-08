import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Step 1: Load data
df = pd.read_csv("data/cleaned_data.csv", index_col=0, parse_dates=['datetime'])
df_hourly = pd.read_csv("data/df_hourly.csv", index_col=0, parse_dates=True)
df_daily = pd.read_csv("data/df_daily.csv", index_col=0, parse_dates=True)

# Step 2: Choose the target variable
target_column = 'Global_active_power'  # You can choose the column that you're forecasting

# Step 3: Prepare the data (use hourly data for this example)
df_hourly_target = df_hourly[target_column]

# Step 4: Check stationarity of the time series (ARIMA works best on stationary data)
from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(df_hourly_target)
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")

# If p-value is greater than 0.05, the series is not stationary. In this case, we can take the difference to make it stationary.

# Step 5: Differencing to make the series stationary if necessary (d=1)
df_hourly_target_diff = df_hourly_target.diff().dropna()

# Check again for stationarity after differencing
adf_test_diff = adfuller(df_hourly_target_diff)
print(f"ADF Statistic (Differenced): {adf_test_diff[0]}")
print(f"p-value (Differenced): {adf_test_diff[1]}")

# Step 6: Fit the ARIMA model (you can adjust p, d, q based on your needs)
# (1, 1, 1) is a common starting point for ARIMA models

arima_model = sm.tsa.ARIMA(df_hourly_target, order=(1, 1, 1))
arima_model_fit = arima_model.fit()

# Step 7: Check the summary of the ARIMA model
print(arima_model_fit.summary())

# Step 8: Make predictions
# Forecasting the next 'steps' (e.g., next 30 hours)
forecast_steps = 30
forecast = arima_model_fit.forecast(steps=forecast_steps)

# Step 9: Plot only the ARIMA forecast
plt.figure(figsize=(10, 6))
plt.plot(pd.date_range(df_hourly_target.index[-1], periods=forecast_steps+1, freq='H')[1:], forecast, label="ARIMA Forecast", linestyle='--')
plt.legend()
plt.title("ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Global Active Power")
plt.show()

# Step 10: Evaluate the model (optional, if you have a test set)
# If you have a test set, you can evaluate using RMSE or MSE
y_test = df_hourly_target[-forecast_steps:]  # Example: using the last 'forecast_steps' data for evaluation
mse_arima = mean_squared_error(y_test, forecast)
rmse_arima = np.sqrt(mse_arima)

print(f"ARIMA RMSE: {rmse_arima}")
