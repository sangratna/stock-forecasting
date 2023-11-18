import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

start = '2000-01-01'
end = '2023-11-18'

st.title('STOCK TREND PREDICTION')
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

st.subheader('Closing price vs Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()  # Corrected
ma200 = df['Close'].rolling(200).mean()  # Corrected
fig1, ax1 = plt.subplots(figsize=(15, 9))
ax1.plot(ma100, c='green', linewidth=2, label='100-day MA')  # Added label
ax1.plot(ma200, c='blue', linewidth=2, label='200-day MA')  # Added label
ax1.plot(df['Close'], c='navy', linewidth=2, label='Closing Price')  # Added label

# Set labels and ticks for ax1
ax1.set_xlabel('Date', color='navy', fontweight='bold', size=18)
ax1.set_ylabel('Closing Price', color='navy', fontweight='bold', size=18)
ax1.tick_params(axis='both', which='both', labelsize=15, colors='navy', width=2)


# Set the background color for Seaborn plot
sns.set_style({'axes.facecolor': '#E0FFFF'})
ax1.spines['bottom'].set_linewidth(3)
ax1.spines['top'].set_linewidth(3)
ax1.spines['right'].set_linewidth(3)
ax1.spines['left'].set_linewidth(3)
fig1.patch.set_facecolor('#FBEEE6')

# Add legend for ax1
ax1.legend()

st.pyplot(fig1)

tr_data =  pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
ts_data = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
tr_data_array = scaler.fit_transform(tr_data)
ts_data_array = scaler.fit_transform(ts_data)

from keras.models import load_model
model = load_model ('keras_model')
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
scaler.scale_
scale_factor = 1/.00575159
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor

st.subheader('FINAL PREDICTION')
sns.set_style({'axes.facecolor': '#E0FFFF'})
fig, ax = plt.subplots(figsize=(15, 9))


ax.plot(y_test, c='blue', linewidth=2, label='Original price')
ax.plot(y_pred, c='red', linewidth=2, label='Predicted price')
ax.set_xlabel('Time', fontweight='bold', size=15)
ax.set_ylabel('Price', fontweight='bold', size=15)
ax.yaxis.tick_left()
ax.xaxis.tick_bottom()
ax.tick_params(axis='both', which='both', width=4)
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax1.tick_params(axis='both', which='both', labelsize=15, colors='navy', width=2)
ax.tick_params(axis='x', labelsize=15, labelcolor='black', width=2, which='both', direction='out')
ax.tick_params(axis='y', labelsize=15, labelcolor='black', width=2, which='both', direction='out')

fig.patch.set_facecolor('#FBEEE6')

# Add legend
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)
st.text("READ IT")

# Or use markdown for more styling options
st.markdown("""
    ## Note
    In final prediction plot x axis represents the instances of test dataset not actual time
    for long term investment follow second plot  
""")

# You can also use HTML for styling

st.subheader('Made with love By "SBG"')