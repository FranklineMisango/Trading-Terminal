import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from pandas_datareader._utils import RemoteDataError
import streamlit as st
from socket import gaierror
import tickers as ti
from langchain_core.tools import tool


@tool
def tool_capm_analysis(ticker : str, start_date : dt.time, end_date : dt.time):
    '''This function calculates the expected return of a stock using the Capital Asset Pricing Model (CAPM).'''
    def get_stock_data(ticker, start, end):
        return pdr.DataReader(ticker, start_date, end_date)

    # Calculates expected return using CAPM.                
    def calculate_expected_return(stock, index, risk_free_return):
        # Check if the index is DateTimeIndex, if not, convert it
        if not isinstance(stock.index, pd.DatetimeIndex):
            stock.index = pd.to_datetime(stock.index)
        if not isinstance(index.index, pd.DatetimeIndex):
            index.index = pd.to_datetime(index.index)
        
        # Resample to monthly data
        return_stock = stock.resample('M').last()['Adj Close']
        return_index = index.resample('M').last()['Adj Close']

        # Create DataFrame with returns
        df = pd.DataFrame({'stock_close': return_stock, 'index_close': return_index})
        df[['stock_return', 'index_return']] = np.log(df / df.shift(1))
        df = df.dropna()

        # Check if df contains non-empty vectors
        if len(df['index_return']) == 0 or len(df['stock_return']) == 0:
            raise ValueError("Empty vectors found in DataFrame df")

        # Calculate beta and alpha
        beta, alpha = np.polyfit(df['index_return'], df['stock_return'], deg=1)
        
        # Calculate expected return
        expected_return = risk_free_return + beta * (df['index_return'].mean() * 12 - risk_free_return)
        return expected_return

    # Risk-free return rate
    risk_free_return = 0.02

    # Define time period
    start = start_date
    end = end_date

    # Get all tickers in NASDAQ
    #nasdaq_tickers = ti.tickers_nasdaq()
    sp_500 =  ti.tickers_sp500()

    # Index ticker
    index_ticker = '^GSPC'

    # Fetch index data
    try:
        index_data = get_stock_data(index_ticker, start, end)
    except RemoteDataError:
        st.write("Failed to fetch index data.")
        return

    # Loop through NASDAQ tickers
    for ticker in sp_500:
        try:
            # Fetch stock data
            stock_data = get_stock_data(ticker, start, end)

            # Calculate expected return
            expected_return = calculate_expected_return(stock_data, index_data, risk_free_return)

            # Output expected return
            st.write(f'{ticker}: Expected Return: {expected_return}')

        except (RemoteDataError, gaierror):
            st.write(f"Data not available for ticker: {ticker}")


def norm_capm_analysis(ticker, start_date, end_date):
    '''This function calculates the expected return of a stock using the Capital Asset Pricing Model (CAPM).'''
    def get_stock_data(ticker, start, end):
        return pdr.DataReader(ticker, start_date, end_date)

    # Calculates expected return using CAPM.                
    def calculate_expected_return(stock, index, risk_free_return):
        # Check if the index is DateTimeIndex, if not, convert it
        if not isinstance(stock.index, pd.DatetimeIndex):
            stock.index = pd.to_datetime(stock.index)
        if not isinstance(index.index, pd.DatetimeIndex):
            index.index = pd.to_datetime(index.index)
        
        # Resample to monthly data
        return_stock = stock.resample('M').last()['Adj Close']
        return_index = index.resample('M').last()['Adj Close']

        # Create DataFrame with returns
        df = pd.DataFrame({'stock_close': return_stock, 'index_close': return_index})
        df[['stock_return', 'index_return']] = np.log(df / df.shift(1))
        df = df.dropna()

        # Check if df contains non-empty vectors
        if len(df['index_return']) == 0 or len(df['stock_return']) == 0:
            raise ValueError("Empty vectors found in DataFrame df")

        # Calculate beta and alpha
        beta, alpha = np.polyfit(df['index_return'], df['stock_return'], deg=1)
        
        # Calculate expected return
        expected_return = risk_free_return + beta * (df['index_return'].mean() * 12 - risk_free_return)
        return expected_return

    # Risk-free return rate
    risk_free_return = 0.02

    # Define time period
    start = start_date
    end = end_date

    # Get all tickers in NASDAQ
    #nasdaq_tickers = ti.tickers_nasdaq()
    sp_500 =  ti.tickers_sp500()

    # Index ticker
    index_ticker = '^GSPC'

    # Fetch index data
    try:
        index_data = get_stock_data(index_ticker, start, end)
    except RemoteDataError:
        st.write("Failed to fetch index data.")
        return

    # Loop through NASDAQ tickers
    for ticker in sp_500:
        try:
            # Fetch stock data
            stock_data = get_stock_data(ticker, start, end)

            # Calculate expected return
            expected_return = calculate_expected_return(stock_data, index_data, risk_free_return)

            # Output expected return
            st.write(f'{ticker}: Expected Return: {expected_return}')

        except (RemoteDataError, gaierror):
            st.write(f"Data not available for ticker: {ticker}")