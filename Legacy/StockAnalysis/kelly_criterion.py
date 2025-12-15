import datetime as dt
import yfinance as yf
import numpy as np
from langchain_core.tools import tool
import streamlit as st

@tool
def tool_kelly_criterion(symbol:str, start_date:dt.time, end_date:dt.time):
    '''This function calculates the Kelly Criterion for a given stock.'''
    # Define stock symbol and time frame for analysis
    symbol 

    # Download stock data using yfinance package
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate daily returns and drop rows with missing data
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()
    stock_data.dropna(inplace=True)

    # Display the first few rows of the data for verification
    st.write(stock_data.head())

    # Calculate Kelly Criterion
    # Extract positive (wins) and negative (losses) returns
    wins = stock_data['Returns'][stock_data['Returns'] > 0]
    losses = stock_data['Returns'][stock_data['Returns'] <= 0]

    # Calculate win ratio and win-loss ratio
    win_ratio = len(wins) / len(stock_data['Returns'])
    win_loss_ratio = np.mean(wins) / np.abs(np.mean(losses))

    # Apply Kelly Criterion formula
    kelly_criterion = win_ratio - ((1 - win_ratio) / win_loss_ratio)

    # Print the Kelly Criterion percentage
    st.write('Kelly Criterion: {:.3f}%'.format(kelly_criterion * 100))


def norm_kelly_criterion(symbol:str, start_date:dt.time, end_date:dt.time):
    '''This function calculates the Kelly Criterion for a given stock.'''
    # Define stock symbol and time frame for analysis
    symbol 

    # Download stock data using yfinance package
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate daily returns and drop rows with missing data
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()
    stock_data.dropna(inplace=True)

    # Display the first few rows of the data for verification
    st.write(stock_data.head())

    # Calculate Kelly Criterion
    # Extract positive (wins) and negative (losses) returns
    wins = stock_data['Returns'][stock_data['Returns'] > 0]
    losses = stock_data['Returns'][stock_data['Returns'] <= 0]

    # Calculate win ratio and win-loss ratio
    win_ratio = len(wins) / len(stock_data['Returns'])
    win_loss_ratio = np.mean(wins) / np.abs(np.mean(losses))

    # Apply Kelly Criterion formula
    kelly_criterion = win_ratio - ((1 - win_ratio) / win_loss_ratio)

    # Print the Kelly Criterion percentage
    st.write('Kelly Criterion: {:.3f}%'.format(kelly_criterion * 100))