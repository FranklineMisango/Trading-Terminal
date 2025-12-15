# Set plot size
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from matplotlib.pylab import rcParams
import datetime as dt
import plotly.graph_objects as go
from fastai.tabular.all import add_datepart
from sklearn.preprocessing import MinMaxScaler
from langchain_core.tools import tool
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


@tool
def tool_stock_recommendations_using_moving_averages(start_date : dt.date, end_date : dt.date, ticker: str):
    '''This tool allows you to analyze stock prices using moving'''
    rcParams['figure.figsize'] = 20, 10

    # Download historical data for Google stock
    data = yf.download(ticker, start_date, end_date)

    # Moving Average and other calculations
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for close price, 50-day MA, and 200-day MA
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{ticker} Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50 Day MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], mode='lines', name='200 Day MA'))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Prices with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Display the interactive chart using Streamlit
    st.plotly_chart(fig)        

    # Preprocessing for Linear Regression and k-Nearest Neighbors
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data = add_datepart(data, 'Date')
    data.drop('Elapsed', axis=1, inplace=True)  # Remove Elapsed column

    # Scaling data to fit in graph
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.drop(['Close'], axis=1))

    # Train-test split
    train_size = int(len(data) * 0.8)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    # Separate features and target variable
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]

    # Initialize SimpleImputer to handle missing values
    imputer = SimpleImputer(strategy='mean')

    # Fit and transform the imputer on the training data
    X_train_imputed = imputer.fit_transform(X_train)

    # Transform the testing data using the fitted imputer
    X_test_imputed = imputer.transform(X_test)

    # Initialize and fit Linear Regression model on the imputed data
    lr_model = LinearRegression()
    lr_model.fit(X_train_imputed, y_train)
    lr_score = lr_model.score(X_test_imputed, y_test)
    lr_predictions = lr_model.predict(X_test_imputed)
    lr_printing_score = f"Linear Regression Score: {lr_score}"
    st.write(lr_printing_score)

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for actual and predicted values
    fig.add_trace(go.Scatter(x=data.index[train_size:], y=y_test, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=data.index[train_size:], y=lr_predictions, mode='lines', name='Linear Regression Predictions'))

    # Update layout
    fig.update_layout(
        title='Actual vs. Predicted Prices (Linear Regression)',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Display the interactive chart using Streamlit
    st.plotly_chart(fig)


def normal_stock_recommendations_using_moving_averages(ticker, start_date, end_date):
   
    rcParams['figure.figsize'] = 20, 10

    # Download historical data for Google stock
    data = yf.download(ticker, start_date, end_date)

    # Moving Average and other calculations
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for close price, 50-day MA, and 200-day MA
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{ticker} Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50 Day MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], mode='lines', name='200 Day MA'))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Prices with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Display the interactive chart using Streamlit
    st.plotly_chart(fig)        

    # Preprocessing for Linear Regression and k-Nearest Neighbors
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data = add_datepart(data, 'Date')
    data.drop('Elapsed', axis=1, inplace=True)  # Remove Elapsed column

    # Scaling data to fit in graph
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.drop(['Close'], axis=1))

    # Train-test split
    train_size = int(len(data) * 0.8)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    # Separate features and target variable
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]

    # Initialize SimpleImputer to handle missing values
    imputer = SimpleImputer(strategy='mean')

    # Fit and transform the imputer on the training data
    X_train_imputed = imputer.fit_transform(X_train)

    # Transform the testing data using the fitted imputer
    X_test_imputed = imputer.transform(X_test)

    # Initialize and fit Linear Regression model on the imputed data
    lr_model = LinearRegression()
    lr_model.fit(X_train_imputed, y_train)
    lr_score = lr_model.score(X_test_imputed, y_test)
    lr_predictions = lr_model.predict(X_test_imputed)
    lr_printing_score = f"Linear Regression Score: {lr_score}"
    st.write(lr_printing_score)

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for actual and predicted values
    fig.add_trace(go.Scatter(x=data.index[train_size:], y=y_test, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=data.index[train_size:], y=lr_predictions, mode='lines', name='Linear Regression Predictions'))

    # Update layout
    fig.update_layout(
        title='Actual vs. Predicted Prices (Linear Regression)',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Display the interactive chart using Streamlit
    st.plotly_chart(fig)
