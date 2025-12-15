import yfinance as yf
import streamlit as st
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import Sequential, layers
from math import sqrt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from langchain_core.tools import tool
import math
from keras.src.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

@tool
def tool_lstm_predictions(start_date: dt.date, end_date: dt.date, ticker: str):
    '''This tool allows you to predict stock prices using LSTM'''
    
    df = yf.download(ticker, start=start_date, end=end_date)
    data = df.filter(['Close'])
    dataset = data.values
    train_data_len = math.ceil(len(dataset) * .8)
    st.write(train_data_len)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training data
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    st.write("Values of x_train:", x_train)

    # Check the shape of x_train
    st.write("Shape of x_train:", x_train.shape)

    # Ensure x_train is not empty and has the correct dimensions
    if x_train.shape[0] == 0 or x_train.shape[1] == 0:
        st.error("x_train is empty or has incorrect dimensions.")
        return

    # LSTM network
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=5)

    # Create testing dataset
    test_data = scaled_data[train_data_len - 60:, :]
    x_test = [test_data[i-60:i, 0] for i in range(60, len(test_data))]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(mean_squared_error(data[train_data_len:].values, predictions))

    # Split data into train and valid datasets
    train = data[:train_data_len]
    valid = data[train_data_len:]
    valid['Predictions'] = predictions

    # The interactive graph
    fig = make_subplots()

    # Add traces for the training data
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))

    # Add traces for the validation data
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Valid'))

    # Add traces for the predictions
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Prediction'))

    # Update the layout
    fig.update_layout(
        title=f"{ticker.upper()} Close Price",
        xaxis_title='Date',
        yaxis_title='Close Price (USD)',
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='x'
    )

    # Show the figure
    st.plotly_chart(fig)

    # Predict next day price
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test_next = np.array([last_60_days_scaled])
    X_test_next = np.reshape(X_test_next, (X_test_next.shape[0], X_test_next.shape[1], 1))
    predicted_price_next_day = scaler.inverse_transform(model.predict(X_test_next))[0][0]
    print(f"The predicted price for the next trading day is: {predicted_price_next_day:.2f}")

def normal_lstm_predictions(start_date, end_date, ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    data = df.filter(['Close'])
    dataset = data.values
    train_data_len = math.ceil(len(dataset) * .8)
    st.write(train_data_len)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training data
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    st.write("Values of x_train:", x_train)

    # Check the shape of x_train
    st.write("Shape of x_train:", x_train.shape)

        # Ensure x_train is not empty and has the correct dimensions
    if x_train.shape[0] == 0 or x_train.shape[1] == 0:
        st.error("x_train is empty or has incorrect dimensions.")
        return

    # LSTM network
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=5)

    # Create testing dataset
    test_data = scaled_data[train_data_len - 60:, :]
    x_test = [test_data[i-60:i, 0] for i in range(60, len(test_data))]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(mean_squared_error(data[train_data_len:].values, predictions))

    # Split data into train and valid datasets
    train = data[:train_data_len]
    valid = data[train_data_len:].copy()  # Ensure valid is a copy of the original DataFrame
    valid.loc[:, 'Predictions'] = predictions

    # The interactive graph
    fig = make_subplots()

    # Add traces for the training data
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))

    # Add traces for the validation data
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Valid'))

    # Add traces for the predictions
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Prediction'))

    # Update the layout
    fig.update_layout(
        title=f"{ticker.upper()} Close Price",
        xaxis_title='Date',
        yaxis_title='Close Price (USD)',
        legend=dict(x=0, y=1, traceorder='normal'),
        hovermode='x'
    )

    # Show the figure
    st.plotly_chart(fig)

    # Predict next day price
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test_next = np.array([last_60_days_scaled])
    X_test_next = np.reshape(X_test_next, (X_test_next.shape[0], X_test_next.shape[1], 1))
    predicted_price_next_day = scaler.inverse_transform(model.predict(X_test_next))[0][0]
    print(f"The predicted price for the next trading day is: {predicted_price_next_day:.2f}")