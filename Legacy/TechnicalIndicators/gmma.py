import datetime as dt
import yfinance as yf
import streamlit as st
from langchain_core.tools import tool
from plotly import graph_objs as go
import pandas as pd
import matplotlib.dates as mdates
import numpy as np


@tool
def tool_gmma(start_date: dt.time, end_date: dt.time, ticker: str):
    ''' This tool plots the candlestick chart of a stock along with the Guppy Multiple Moving Averages (GMMA) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Short-term for EMA
    df["EMA3"] = df["Adj Close"].ewm(span=3, adjust=False).mean()
    df["EMA5"] = df["Adj Close"].ewm(span=5, adjust=False).mean()
    df["EMA8"] = df["Adj Close"].ewm(span=8, adjust=False).mean()
    df["EMA10"] = df["Adj Close"].ewm(span=10, adjust=False).mean()
    df["EMA12"] = df["Adj Close"].ewm(span=12, adjust=False).mean()
    df["EMA15"] = df["Adj Close"].ewm(span=15, adjust=False).mean()

    # Long-term for EMA
    df["EMA30"] = df["Adj Close"].ewm(span=30, adjust=False).mean()
    df["EMA35"] = df["Adj Close"].ewm(span=35, adjust=False).mean()
    df["EMA40"] = df["Adj Close"].ewm(span=40, adjust=False).mean()
    df["EMA45"] = df["Adj Close"].ewm(span=45, adjust=False).mean()
    df["EMA50"] = df["Adj Close"].ewm(span=50, adjust=False).mean()
    df["EMA60"] = df["Adj Close"].ewm(span=60, adjust=False).mean()

    EMA_Short = df[["EMA3", "EMA5", "EMA8", "EMA10", "EMA12", "EMA15"]]
    EMA_Long = df[["EMA30", "EMA35", "EMA40", "EMA45", "EMA50", "EMA60"]]

    # Short-term for SMA
    df["SMA3"] = df["Adj Close"].rolling(window=3).mean()
    df["SMA5"] = df["Adj Close"].rolling(window=5).mean()
    df["SMA8"] = df["Adj Close"].rolling(window=8).mean()
    df["SMA10"] = df["Adj Close"].rolling(window=10).mean()
    df["SMA12"] = df["Adj Close"].rolling(window=12).mean()
    df["SMA15"] = df["Adj Close"].rolling(window=15).mean()

    # Long-term for SMA
    df["SMA30"] = df["Adj Close"].rolling(window=30).mean()
    df["SMA35"] = df["Adj Close"].rolling(window=35).mean()
    df["SMA40"] = df["Adj Close"].rolling(window=40).mean()
    df["SMA45"] = df["Adj Close"].rolling(window=45).mean()
    df["SMA50"] = df["Adj Close"].rolling(window=50).mean()
    df["SMA60"] = df["Adj Close"].rolling(window=60).mean()

    SMA_Short = df[["SMA3", "SMA5", "SMA8", "SMA10", "SMA12", "SMA15"]]
    SMA_Long = df[["SMA30", "SMA35", "SMA40", "SMA45", "SMA50", "SMA60"]]

    # Plot EMA
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    for col in EMA_Short.columns:
        fig_ema.add_trace(go.Scatter(x=df.index, y=EMA_Short[col], mode='lines', name=col, line=dict(color='blue')))
    for col in EMA_Long.columns:
        fig_ema.add_trace(go.Scatter(x=df.index, y=EMA_Long[col], mode='lines', name=col, line=dict(color='orange')))
    fig_ema.update_layout(title="Guppy Multiple Moving Averages of EMA", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_ema)

    # Plot SMA
    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    for col in SMA_Short.columns:
        fig_sma.add_trace(go.Scatter(x=df.index, y=SMA_Short[col], mode='lines', name=col, line=dict(color='blue')))
    for col in SMA_Long.columns:
        fig_sma.add_trace(go.Scatter(x=df.index, y=SMA_Long[col], mode='lines', name=col, line=dict(color='orange')))
    fig_sma.update_layout(title="Guppy Multiple Moving Averages of SMA", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_sma)

    st.warning("Untick the volume to view the candlesticks and the movement lines")

    # Candlestick with GMMA
    fig_candlestick = go.Figure()

    # Candlestick
    fig_candlestick.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))

    # Plot EMA on Candlestick
    for col in EMA_Short.columns:
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=EMA_Short[col], mode='lines', name=col, line=dict(color='orange')))
    for col in EMA_Long.columns:
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=EMA_Long[col], mode='lines', name=col, line=dict(color='blue')))

    # Volume
    fig_candlestick.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=np.where(df['Open'] < df['Close'], 'green', 'red')))

    fig_candlestick.update_layout(title="Stock Closing Price", xaxis_title="Date", yaxis_title="Price",
                                yaxis2=dict(title="Volume", overlaying='y', side='right', tickformat=',.0f'))  # Set tick format to not show in millions
    st.plotly_chart(fig_candlestick)


def norm_gmma(start_date, end_date, ticker):\

    ''' This tool plots the candlestick chart of a stock along with the Guppy Multiple Moving Averages (GMMA) of the stock's closing price.'''
    
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Short-term for EMA
    df["EMA3"] = df["Adj Close"].ewm(span=3, adjust=False).mean()
    df["EMA5"] = df["Adj Close"].ewm(span=5, adjust=False).mean()
    df["EMA8"] = df["Adj Close"].ewm(span=8, adjust=False).mean()
    df["EMA10"] = df["Adj Close"].ewm(span=10, adjust=False).mean()
    df["EMA12"] = df["Adj Close"].ewm(span=12, adjust=False).mean()
    df["EMA15"] = df["Adj Close"].ewm(span=15, adjust=False).mean()

    # Long-term for EMA
    df["EMA30"] = df["Adj Close"].ewm(span=30, adjust=False).mean()
    df["EMA35"] = df["Adj Close"].ewm(span=35, adjust=False).mean()
    df["EMA40"] = df["Adj Close"].ewm(span=40, adjust=False).mean()
    df["EMA45"] = df["Adj Close"].ewm(span=45, adjust=False).mean()
    df["EMA50"] = df["Adj Close"].ewm(span=50, adjust=False).mean()
    df["EMA60"] = df["Adj Close"].ewm(span=60, adjust=False).mean()

    EMA_Short = df[["EMA3", "EMA5", "EMA8", "EMA10", "EMA12", "EMA15"]]
    EMA_Long = df[["EMA30", "EMA35", "EMA40", "EMA45", "EMA50", "EMA60"]]

    # Short-term for SMA
    df["SMA3"] = df["Adj Close"].rolling(window=3).mean()
    df["SMA5"] = df["Adj Close"].rolling(window=5).mean()
    df["SMA8"] = df["Adj Close"].rolling(window=8).mean()
    df["SMA10"] = df["Adj Close"].rolling(window=10).mean()
    df["SMA12"] = df["Adj Close"].rolling(window=12).mean()
    df["SMA15"] = df["Adj Close"].rolling(window=15).mean()

    # Long-term for SMA
    df["SMA30"] = df["Adj Close"].rolling(window=30).mean()
    df["SMA35"] = df["Adj Close"].rolling(window=35).mean()
    df["SMA40"] = df["Adj Close"].rolling(window=40).mean()
    df["SMA45"] = df["Adj Close"].rolling(window=45).mean()
    df["SMA50"] = df["Adj Close"].rolling(window=50).mean()
    df["SMA60"] = df["Adj Close"].rolling(window=60).mean()

    SMA_Short = df[["SMA3", "SMA5", "SMA8", "SMA10", "SMA12", "SMA15"]]
    SMA_Long = df[["SMA30", "SMA35", "SMA40", "SMA45", "SMA50", "SMA60"]]

    # Plot EMA
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    for col in EMA_Short.columns:
        fig_ema.add_trace(go.Scatter(x=df.index, y=EMA_Short[col], mode='lines', name=col, line=dict(color='blue')))
    for col in EMA_Long.columns:
        fig_ema.add_trace(go.Scatter(x=df.index, y=EMA_Long[col], mode='lines', name=col, line=dict(color='orange')))
    fig_ema.update_layout(title="Guppy Multiple Moving Averages of EMA", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_ema)

    # Plot SMA
    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    for col in SMA_Short.columns:
        fig_sma.add_trace(go.Scatter(x=df.index, y=SMA_Short[col], mode='lines', name=col, line=dict(color='blue')))
    for col in SMA_Long.columns:
        fig_sma.add_trace(go.Scatter(x=df.index, y=SMA_Long[col], mode='lines', name=col, line=dict(color='orange')))
    fig_sma.update_layout(title="Guppy Multiple Moving Averages of SMA", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_sma)

    st.warning("Untick the volume to view the candlesticks and the movement lines")

    # Candlestick with GMMA
    fig_candlestick = go.Figure()

    # Candlestick
    fig_candlestick.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))

    # Plot EMA on Candlestick
    for col in EMA_Short.columns:
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=EMA_Short[col], mode='lines', name=col, line=dict(color='orange')))
    for col in EMA_Long.columns:
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=EMA_Long[col], mode='lines', name=col, line=dict(color='blue')))

    # Volume
    fig_candlestick.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=np.where(df['Open'] < df['Close'], 'green', 'red')))

    fig_candlestick.update_layout(title="Stock Closing Price", xaxis_title="Date", yaxis_title="Price",
                                yaxis2=dict(title="Volume", overlaying='y', side='right', tickformat=',.0f'))  # Set tick format to not show in millions
    st.plotly_chart(fig_candlestick)