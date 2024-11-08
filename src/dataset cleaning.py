import pandas as pd
import numpy as np

# Path to the dataset
data_path = "data/household_power_consumption.txt"

# Load the data
df = pd.read_csv(data_path, sep=';')

# Print the column names to debug (check for any extra spaces or hidden characters)
print("Column names before cleaning:", df.columns)

# Remove extra spaces in column names (if any)
df.columns = df.columns.str.strip()

# Replace '?' with NaN for proper handling of missing values
df.replace('?', np.nan, inplace=True)

# Drop rows with NaN values
df.dropna(inplace=True)

# Convert 'Date' and 'Time' columns to datetime and set as index
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('datetime', inplace=True)

# Print the column names again to ensure everything is correct
print("Columns after cleaning:", df.columns)

# Use the correct column name ('Global_active_power' with the correct case)
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# Resample to hourly or daily data, depending on your forecasting goals
df_hourly = df['Global_active_power'].resample('H').mean()  # Hourly
df_daily = df['Global_active_power'].resample('D').mean()   # Daily


