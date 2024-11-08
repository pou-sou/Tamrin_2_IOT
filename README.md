# Energy Consumption Prediction Using Machine Learning Models

This repository contains code to predict household energy consumption based on historical data using various machine learning models and time series forecasting techniques.

## Requirements

To run the code, you'll need to install the following dependencies:

- Python 3.7+
- pandas
- numpy
- statsmodels
- scikit-learn
- xgboost
- matplotlib

## Forecasting Modules
The cleaned data is then used in four different modules for energy consumption prediction:

- ARIMA (AutoRegressive Integrated Moving Average): ARIMA is a classical time series forecasting model that is best suited for predicting univariate time series data that exhibit trends or seasonality.
- Linear Regression: A simple regression model used to predict the target variable (Global_active_power) based on past values and features.
- Random Forest: A powerful ensemble learning method used for both classification and regression tasks. Random Forest models the relationship between input features and the target variable.
- XGBoost (Extreme Gradient Boosting): A gradient boosting algorithm that is efficient and widely used in machine learning competitions. XGBoost provides strong predictive performance for regression tasks.

## Conclusion

After comparing all four models, XGBoost proved to be the best for predicting household energy consumption, delivering the most accurate results. It can effectively capture the complexities and nonlinear relationships within the data. However, depending on the dataset and the complexity of the problem, **ARIMA** might also be a strong contender, especially for time series forecasting.

### Files in this Repository:
- [XGBoost forecasting.py](https://github.com/pou-sou/Tamrin_2_IOT/blob/main/src/XGBoost%20forecasting.py) used for XGBoost forecasting module
!alt(./assets/XGBoost Prediction results.png)
- [arima forecasting.py](https://github.com/pou-sou/Tamrin_2_IOT/blob/main/src/arima%20forecasting.py)used for XGBoost forecasting module
- [linear and random forest forecasting](https://github.com/pou-sou/Tamrin_2_IOT/blob/main/src/linear%20and%20random%20forest%20forecasting.py)used for linear and random forest forecasting modules
- [dataset cleaning.py](https://github.com/pou-sou/Tamrin_2_IOT/blob/main/src/dataset%20cleaning.py)the code used to clean the data
- [cleaned_data.csv](https://github.com/pou-sou/Tamrin_2_IOT/blob/main/data/cleaned_data.csv) result of dataset cleaning.py removed nulls and incomplete data
- [household power consumption](https://github.com/pou-sou/Tamrin_2_IOT/blob/main/data/household_power_consumption.txt) raw data i got from [the UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) 

### Data Cleaning

Before applying any machine learning or forecasting models, we first clean and preprocess the data. The dataset contains household electric power consumption measurements with the following columns:

- Date
- Time
- Global_active_power
- Global_reactive_power
- Voltage
- Global_intensity
- Sub_metering_1
- Sub_metering_2
- Sub_metering_3

### Data Cleaning Steps:

- Replacing missing values: We replaced missing values (?) with NaN and removed rows with missing data.
- Date and Time Processing: Combined the Date and Time columns into a single datetime column, which was then set as the index for easy time series analysis.
- Numerical Conversion: Converted all relevant columns to numeric values, ensuring that they could be used for modeling.
- Hourly and Daily Resampling: The data was resampled to hourly and daily aggregates to create two datasets: df_hourly.csv and df_daily.csv.

