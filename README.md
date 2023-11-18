# # stock-forecasting


**Stock Price Prediction Web App**
This web app predicts stock prices using an LSTM (Long Short-Term Memory) model. It leverages historical stock data fetched from Yahoo Finance and visualizes the actual closing prices, along with the 100-day and 200-day moving averages.


**Overview**
Data Source: Yahoo Finance
Time Range: '2000-01-01' to '2023-11-18'
Model: LSTM-based Stock Price Prediction
Web Framework: Streamlit


**Features**
Actual Closing Prices: View the historical closing prices of the selected stock.
Moving Averages: Visualize the 100-day and 200-day moving averages to identify trends.
LSTM Predictions: Explore LSTM model predictions for future stock prices.


**File Structure**
app.py: Streamlit web app script.
requirements.txt: List of dependencies.
models/: Directory containing trained LSTM model files.
data/: Directory containing stock data fetched from Yahoo Finance.


**Usage**
Enter the stock ticker symbol in the provided input field.
Explore the actual closing prices, moving averages, and LSTM predictions.
Gain insights into historical trends and potential future price movements.



**Acknowledgments**
Stock data provided by Yahoo Finance.
LSTM model implementation inspired by Keras.
