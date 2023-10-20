# E:\programming\Project\Portfolio\ph_stock_prediction\ph_stock_prediction.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import os

def download_stock_data():
    psei = yf.Ticker("PSEI.PS")
    data = psei.history(period="5y")
    data.to_csv("ph_stock_prediction.csv")

if not os.path.exists("ph_stock_prediction.csv"):
    download_stock_data()

data = pd.read_csv("ph_stock_prediction.csv")
data.dropna(inplace=True)

low, high = data['Close'].quantile([0.01, 0.99])
data = data[(data['Close'] >= low) & (data['Close'] <= high)]

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA25'] = data['Close'].rolling(window=25).mean()

data['STD'] = data['Close'].rolling(window=25).std()
data['Upper'] = data['MA25'] + (data['STD'] * 2)
data['Lower'] = data['MA25'] - (data['STD'] * 2)

X = data[['MA5', 'MA25', 'STD', 'Upper', 'Lower']].shift(1).dropna()
y = data['Close'].loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

clf = GridSearchCV(RandomForestRegressor(), params, cv=5)
clf.fit(X_train, y_train)

model = clf.best_estimator_

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Score: {train_score}")
print(f"Test Score: {test_score}")

y_pred = model.predict(X_test)

future_days = 10
X_future = X_test[-future_days:].copy()
for i in range(future_days):
    next_day_values = X_future.iloc[-1].values + np.array([y_pred[-1], y_pred[-1], 0, 0, 0])
    new_row = pd.DataFrame([next_day_values], columns=X.columns, index=[X_future.index[-1] + pd.DateOffset(1)])
    X_future = pd.concat([X_future, new_row])
y_future_pred = model.predict(X_future[-future_days:])

plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test, label='True', color='blue')
plt.plot(data.index[-len(y_pred):], y_pred, label='Predicted', color='red')
plt.plot(pd.date_range(data.index[-1] + pd.DateOffset(1), periods=future_days), y_future_pred, label='Future Prediction', color='green')
plt.title('PSEi Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('psei_stock_price_prediction.png')  # グラフを保存
plt.show()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
