### Stock Price Prediction Model

# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import yfinance as yf

# Step 1: Fetch Dataset
ticker = 'AAPL'  # Replace with desired stock symbol
start_date = '2020-01-01'
end_date = '2023-01-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Save the dataset as CSV
data.to_csv('stock_data.csv')

# Step 2: Preprocess Data
print(data.head())
print("\nData Shape:", data.shape)

# Extract 'Close' price and preprocess it
prices = data['Close'].values
prices = prices.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Step 3: Create Sequences
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(prices_scaled)):
    X.append(prices_scaled[i-sequence_length:i, 0])
    y.append(prices_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into training and testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 4: Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('stock_price_model.h5')

# Step 6: Make Predictions
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 7: Visualize Results
plt.figure(figsize=(14, 6))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(test_predictions, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Step 8: Evaluate Model
mse = mean_squared_error(actual_prices, test_predictions)
r2 = r2_score(actual_prices, test_predictions)
print('Mean Squared Error:', mse)
print('R-Squared:', r2)

# Save results as CSV
results = pd.DataFrame({'Actual': actual_prices.flatten(), 'Predicted': test_predictions.flatten()})
results.to_csv('predicted_vs_actual.csv', index=False)

# Additional Plots
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()