# Developed By : Rama E.K. Lekshmi
# Register Number : 212222240082
# Date: 
# Ex.No: 6               HOLT WINTERS METHOD

### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
```
Load the AirPassengers dataset
```py
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'])
```
Set Month as the index
```py
data.set_index('Month', inplace=True)
```
Use '#Passengers' column
```py
monthly_data = data['#Passengers']
```
Scale the data using MinMaxScaler
```py
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(monthly_data.values.reshape(-1, 1)).flatten(), 
                        index=monthly_data.index)
```
Split into training and testing sets (80% train, 20% test)
```py
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
```
Fit the Holt-Winters additive model on training data
```py
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()

Forecast for the test data length
```py
test_predictions_add = model_add.forecast(steps=len(test_data))
```
Evaluate model performance on test data
```py
mae = mean_absolute_error(test_data, test_predictions_add)
rmse = mean_squared_error(test_data, test_predictions_add, squared=False)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
```
Plot 1: Train, Test, and Test Predictions
```py
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='Train', color='black')
plt.plot(test_data, label='Test', color='green')
plt.plot(test_predictions_add, label='Prediction', color='red')
plt.title('Holt-Winters Additive Forecast - Train vs. Test Predictions')
plt.legend(loc='best')
plt.grid('True')
plt.show()
```
Fit the final model on the entire dataset (additive trend & seasonality)
```py
final_model = ExponentialSmoothing(monthly_data, trend='add', seasonal='add', seasonal_periods=12).fit()
```
Forecast next 12 months
```py
forecast = final_model.forecast(steps=12)
```
Plot Historical Data with 12-Month Forecast
```py
plt.figure(figsize=(12, 8))
monthly_data.plot(label='Observed', legend=True)
forecast.plot(label='Forecast', legend=True)
plt.title('Holt-Winters Additive Forecast - Next 12 Months')
plt.xlabel('Date')
plt.ylabel('Passengers Count')
plt.grid('True')
plt.show()
```
Output final predictions
```py
print("Final Predictions for the next 12 months:")
print(forecast)
```

### OUTPUT:


TEST_PREDICTION

![image](https://github.com/user-attachments/assets/8aa6dfb3-251e-4691-b271-4e769ead952e)

FINAL_PREDICTION

![image](https://github.com/user-attachments/assets/b9948f51-f138-4671-91ed-80af540380e7)


![image](https://github.com/user-attachments/assets/69c88426-3a84-4b9e-ab7a-a3cb9737ed69)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
