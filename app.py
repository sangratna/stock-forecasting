import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

start = '2000-01-01'
end = '2023-11-18'

st.title('Stock trend prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)
df.head()

st.subheader('Data from 2000 - 2023')
st.write(df.describe())
st.subheader('Closing price vs Time chart')
fig, ax = plt.subplots(figsize=(15, 9))
sns.set_style({'axes.facecolor': '#E0FFFF'})
ax.plot(df['Close'], c='navy', linewidth=2)
ax.set_title('Stock Closing Prices Over Time', color='navy', fontweight='bold', size=15)
ax.set_xlabel('Date', color='navy', fontweight='bold', size=15)
ax.set_ylabel('Closing Price', color='navy', fontweight='bold', size=15)
ax.tick_params(axis='both', which='both', width=4, labelcolor='navy', labelsize=15)

# Increase the thickness of the spines
ax.spines['bottom'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)

# Set the background color
fig.patch.set_facecolor('#FBEEE6')

# Display the plot in Streamlit
st.pyplot(fig)
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
st.subheader('Closing Price vs 100 days & 200 days moving average')
fig1, ax1 = plt.subplots(figsize=(15, 9))
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Plotting Closing Price and 100-day Moving Average
ax1.plot(df['Close'], c='navy', linewidth=2, label='Closing Price')
ax1.plot(ma100, c='green', linewidth=3, label='100-day MA')
ax1.set_ylabel('Price', color='navy', fontweight='bold', size=15)
ax1.tick_params(axis='both', which='both', labelsize=12, colors='navy', width=2)
ax1.legend()

fig1.patch.set_facecolor('#FBEEE6')
plt.subplots_adjust(hspace=0.3)  # Adjust vertical space between subplots
st.pyplot(fig1) 

fig2, ax2 = plt.subplots(figsize=(15, 9))
ax2.plot(df['Close'], c='navy', linewidth=2, label='Closing Price')
ax2.plot(ma200, c='blue', linewidth=3, label='200-day MA')
ax2.set_xlabel('Date', color='navy', fontweight='bold', size=15)
ax2.set_ylabel('Price', color='navy', fontweight='bold', size=15)
ax2.tick_params(axis='both', which='both', labelsize=12, colors='navy', width=2)
ax2.legend()

# Set background color
fig2.patch.set_facecolor('#FBEEE6')
plt.subplots_adjust(hspace=0.3)  # Adjust vertical space between subplots
st.pyplot(fig2) 

tr_data =  pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
ts_data = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

tr_data_array = scaler.fit_transform(tr_data)
ts_data_array = scaler.fit_transform(ts_data)


import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import zipfile
import logging
from keras.models import load_model

# Unzip the Keras model
with zipfile.ZipFile('keras_model.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load the Keras model
model = load_model('keras_model')
# try:
#     model = load_model('keras_model')
#     logging.info('Model loaded successfully!')
# except Exception as e:
#     logging.error(f"Error loading the model: {e}")
# model = load_model('keras_model')

past_100_days = tr_data.tail(100)
final_df = past_100_days.append(ts_data, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor
import matplotlib.pyplot as plt

# Assuming you already have y_test and y_pred defined

# Create a single plot

fig4, ax4 = plt.subplots(figsize=(15, 9))
ax4.plot(y_test, c='blue', linewidth=3, label='Actual Price')  # Assuming y_test is defined
ax4.plot(y_pred, c='green', linewidth=3, label='Prediction')  # Assuming y_pred is defined

# Customize the plot
ax4.set_ylabel('Price', color='navy', fontweight='bold', size=15)
ax4.tick_params(axis='both', which='both', labelsize=12, colors='navy', width=2)
ax4.legend()
fig4.patch.set_facecolor('#FBEEE6')
# Show the plot
st.pyplot(fig4)



# Or use st.markdown for more formatting options
st.markdown("**Note:** X- axis in last plot shows index number in test dataset.")

st.subheader('Made with love By "SBG"')