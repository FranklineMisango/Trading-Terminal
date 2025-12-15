 # Load list of S&P 500 tickers from tickers module
import tickers as ti
import yfinance as yf
import pandas as pd
import streamlit as st
import ta_functions as ta
import datetime as dt
from langchain_core.tools import tool

@tool
def tool_rsi_tickers(tool_start_date : dt.date, tool_end_date : dt.date):
    '''This tool allows you to find overbought and oversold tickers using RSI'''
    tickers = ti.tickers_sp500()

    # Initialize lists for overbought and oversold tickers
    oversold_tickers = []
    overbought_tickers = []

    # Retrieve adjusted close prices for the tickers
    sp500_data = yf.download(tickers, tool_start_date, tool_end_date)['Adj Close']

    # Analyze each ticker for RSI
    for ticker in tickers:
        try:
            # Create a new DataFrame for the ticker
            data = sp500_data[[ticker]].copy()

            # Calculate the RSI for the ticker
            data["rsi"] = ta.RSI(data[ticker], timeperiod=14)

            # Calculate the mean of the last 14 RSI values
            mean_rsi = data["rsi"].tail(14).mean()

            # Print the RSI value
            st.write(f'{ticker} has an RSI value of {round(mean_rsi, 2)}')

            # Classify the ticker based on its RSI value
            if mean_rsi <= 30:
                oversold_tickers.append(ticker)
            elif mean_rsi >= 70:
                overbought_tickers.append(ticker)

        except Exception as e:
            print(f'Error processing {ticker}: {e}')

    # Output the lists of oversold and overbought tickers
    st.write(f'Oversold tickers: {oversold_tickers}')
    st.write(f'Overbought tickers: {overbought_tickers}')


def normal_rsi_tickers(start_date, end_date):
     # Load list of S&P 500 tickers from tickers module
    tickers = ti.tickers_sp500()

    # Initialize lists for overbought and oversold tickers
    oversold_tickers = []
    overbought_tickers = []

    # Retrieve adjusted close prices for the tickers
    sp500_data = yf.download(tickers, start_date, end_date)['Adj Close']

    # Analyze each ticker for RSI
    for ticker in tickers:
        try:
            # Create a new DataFrame for the ticker
            data = sp500_data[[ticker]].copy()

            # Calculate the RSI for the ticker
            data["rsi"] = ta.RSI(data[ticker], timeperiod=14)

            # Calculate the mean of the last 14 RSI values
            mean_rsi = data["rsi"].tail(14).mean()

            # Print the RSI value
            st.write(f'{ticker} has an RSI value of {round(mean_rsi, 2)}')

            # Classify the ticker based on its RSI value
            if mean_rsi <= 30:
                oversold_tickers.append(ticker)
            elif mean_rsi >= 70:
                overbought_tickers.append(ticker)

        except Exception as e:
            print(f'Error processing {ticker}: {e}')

    # Output the lists of oversold and overbought tickers
    st.write(f'Oversold tickers: {oversold_tickers}')
    st.write(f'Overbought tickers: {overbought_tickers}')