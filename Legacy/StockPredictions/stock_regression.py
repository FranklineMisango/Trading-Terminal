import yfinance as yf
import streamlit as st
import datetime as dt
from sklearn.model_selection import train_test_split
from langchain_core.tools import tool
from sklearn.linear_model import LinearRegression


 
@tool
def tool_stock_regression(start_date : dt.date, end_date : dt.date, ticker: str):
    '''This tool allows you to predict stock prices using Linear Regression'''
    # Set parameters for stock data
    stock = ticker

    # Download historical data for the specified stock
    data = yf.download(stock, start_date, end_date)

    
    # Prepare the features (X) and target (y)
    X = data.drop(['Close'], axis=1)
    y = data['Adj Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Initialize and train a Linear Regression model
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # Output the intercept of the model
    intercept = regression_model.intercept_
    st.write(f"The intercept for our model is: {intercept}")

    # Evaluate the model's performance
    score = regression_model.score(X_test, y_test)
    st.write(f"The score for our model is: {score}")

    # Predict the next day's closing price using the latest data
    latest_data = data.tail(1).drop(['Close'], axis=1)
    next_day_price = regression_model.predict(latest_data)[0]
    st.write(f"The predicted price for the next trading day is: {next_day_price}") 


def normal_stock_regression(start_date, end_date, ticker):
    # Set parameters for stock data
    stock = ticker

    # Download historical data for the specified stock
    data = yf.download(stock, start_date, end_date)

    
    # Prepare the features (X) and target (y)
    X = data.drop(['Close'], axis=1)
    y = data['Adj Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Initialize and train a Linear Regression model
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # Output the intercept of the model
    intercept = regression_model.intercept_
    st.write(f"The intercept for our model is: {intercept}")

    # Evaluate the model's performance
    score = regression_model.score(X_test, y_test)
    st.write(f"The score for our model is: {score}")

    # Predict the next day's closing price using the latest data
    latest_data = data.tail(1).drop(['Close'], axis=1)
    next_day_price = regression_model.predict(latest_data)[0]
    st.write(f"The predicted price for the next trading day is: {next_day_price}")