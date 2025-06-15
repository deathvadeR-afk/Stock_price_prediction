import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from keras.models import load_model

st.title('Stock Trend Prediction')

stock = st.text_input('Enter Stock Ticker', 'NVDA')

from datetime import datetime
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

stock_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(stock_data)

splitting_len = int(len(stock_data) * 0.70)
x_test = x_test = stock_data[["Close"]].iloc[splitting_len:]


def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Red')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and Moving Average for 250 days')
stock_data['Moving Average for 250 days'] = stock_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), stock_data['Moving Average for 250 days'], stock_data, 0))

st.subheader('Original Close Price and Moving Average for 250 days')
stock_data['Moving Average for 200 days'] = stock_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), stock_data['Moving Average for 250 days'], stock_data, 0))

st.subheader('Original Close Price and Moving Average for 250 days')
stock_data['Moving Average for 100 days'] = stock_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), stock_data['Moving Average for 250 days'], stock_data, 0))

st.subheader('Original Close Price and Moving Average for 100 days and 200 days')
st.pyplot(plot_graph((15,6), stock_data['Moving Average for 250 days'], stock_data, 1, stock_data['Moving Average for 250 days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[["Close"]])
x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i, 0])
    y_data.append(scaled_data[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inverse_predictions = scaler.inverse_transform(predictions)
inverse_y_tests = scaler.inverse_transform(y_data.reshape(-1, 1))


plotting_data = pd.DataFrame(
    {
        "Original_test_data": inverse_y_tests.reshape(-1),
        "predictions": inverse_predictions.reshape(-1)
    },
    index = stock_data.index[splitting_len+100:]
)
st.subheader('Original values vs Predicted values')
st.write(plotting_data)

st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([stock_data.Close[:splitting_len+100],plotting_data], axis=0))
plt.legend(["Data-not used", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)