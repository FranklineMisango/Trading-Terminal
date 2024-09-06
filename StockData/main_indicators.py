import streamlit as st
import pandas as pd
import yfinance as yf
import ta_functions as ta
from langchain_core.tools import tool
import datetime as dt

@tool
def tool_main_indicators(ticker : str ,start_date : dt.time, end_date : dt.time):
    '''This tool allows you to analyze stock data using technical indicators'''
    symbol = ticker

    # Convert string dates to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Download stock data from Yahoo Finance
    data = yf.download(symbol, start=start, end=end)

    # Display Adjusted Close Price
    st.header(f"Adjusted Close Price\n {symbol}")
    st.line_chart(data["Adj Close"])

    # Calculate and display SMA and EMA
    data["SMA"] = ta.SMA(data["Adj Close"], timeperiod=20)
    data["EMA"] = ta.EMA(data["Adj Close"], timeperiod=20)
    st.header(f"Simple Moving Average vs. Exponential Moving Average\n {symbol}")
    st.line_chart(data[["Adj Close", "SMA", "EMA"]])

    # Calculate and display Bollinger Bands
    data["upper_band"], data["middle_band"], data["lower_band"] = ta.BBANDS(data["Adj Close"], timeperiod=20)
    st.header(f"Bollinger Bands\n {symbol}")
    st.line_chart(data[["Adj Close", "upper_band", "middle_band", "lower_band"]])

    # Calculate and display MACD
    data["macd"], data["macdsignal"], data["macdhist"] = ta.MACD(data["Adj Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    st.header(f"Moving Average Convergence Divergence\n {symbol}")
    st.line_chart(data[["macd", "macdsignal"]])

    # Calculate and display CCI
    data["CCI"] = ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=14)
    st.header(f"Commodity Channel Index\n {symbol}")
    st.line_chart(data["CCI"])

    # Calculate and display RSI
    data["RSI"] = ta.RSI(data["Adj Close"], timeperiod=14)
    st.header(f"Relative Strength Index\n {symbol}")
    st.line_chart(data["RSI"])

    # Calculate and display OBV
    data["OBV"] = ta.OBV(data["Adj Close"], data["Volume"]) / 10**6
    st.header(f"On Balance Volume\n {symbol}")
    st.line_chart(data["OBV"])


def norm_main_indicators(ticker, start_date, end_date):
    '''This tool allows you to analyze stock data using technical indicators'''
    symbol = ticker

    # Convert string dates to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Download stock data from Yahoo Finance
    data = yf.download(symbol, start=start, end=end)

    # Display Adjusted Close Price
    st.header(f"Adjusted Close Price\n {symbol}")
    st.line_chart(data["Adj Close"])

    # Calculate and display SMA and EMA
    data["SMA"] = ta.SMA(data["Adj Close"], timeperiod=20)
    data["EMA"] = ta.EMA(data["Adj Close"], timeperiod=20)
    st.header(f"Simple Moving Average vs. Exponential Moving Average\n {symbol}")
    st.line_chart(data[["Adj Close", "SMA", "EMA"]])

    # Calculate and display Bollinger Bands
    data["upper_band"], data["middle_band"], data["lower_band"] = ta.BBANDS(data["Adj Close"], timeperiod=20)
    st.header(f"Bollinger Bands\n {symbol}")
    st.line_chart(data[["Adj Close", "upper_band", "middle_band", "lower_band"]])

    # Calculate and display MACD
    data["macd"], data["macdsignal"], data["macdhist"] = ta.MACD(data["Adj Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    st.header(f"Moving Average Convergence Divergence\n {symbol}")
    st.line_chart(data[["macd", "macdsignal"]])

    # Calculate and display CCI
    data["CCI"] = ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=14)
    st.header(f"Commodity Channel Index\n {symbol}")
    st.line_chart(data["CCI"])

    # Calculate and display RSI
    data["RSI"] = ta.RSI(data["Adj Close"], timeperiod=14)
    st.header(f"Relative Strength Index\n {symbol}")
    st.line_chart(data["RSI"])

    # Calculate and display OBV
    data["OBV"] = ta.OBV(data["Adj Close"], data["Volume"]) / 10**6
    st.header(f"On Balance Volume\n {symbol}")
    st.line_chart(data["OBV"])
