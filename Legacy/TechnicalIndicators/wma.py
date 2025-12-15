import streamlit as st
import datetime as dt
import yfinance as yf
from langchain_core.tools import tool
from plotly import graph_objs as go
import pandas as pd
import plotly.express as px
import matplotlib.dates as mdates
import numpy as np


@tool
def tool_wma(ticker:str, start_date:dt.time, end_date:dt.time):
    '''Tool for Weighted Moving Average (WMA)'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    def WMA(data, n):
        ws = np.zeros(data.shape[0])
        t_sum = sum(range(1, n + 1))
        for i in range(n - 1, data.shape[0]):
            ws[i] = sum(data[i - n + 1 : i + 1] * np.linspace(1, n, n)) / t_sum
        return ws

    df["WMA"] = WMA(df["Adj Close"], 5)

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index[4:], y=df["WMA"][4:], mode='lines', name='WMA'))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='#0079a3', opacity=0.4))

    fig.update_layout(title=f'Stock {symbol} Closing Price', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    # Candlestick with WMA
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index[4:], y=df["WMA"][4:], mode='lines', name='WMA'))

    fig.update_layout(title=f'Stock {symbol} Candlestick Chart with WMA', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

def norm_wma(ticker:str, start_date:dt.time, end_date:dt.time):
    '''Tool for Weighted Moving Average (WMA)'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    def WMA(data, n):
        ws = np.zeros(data.shape[0])
        t_sum = sum(range(1, n + 1))
        for i in range(n - 1, data.shape[0]):
            ws[i] = sum(data[i - n + 1 : i + 1] * np.linspace(1, n, n)) / t_sum
        return ws

    df["WMA"] = WMA(df["Adj Close"], 5)

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index[4:], y=df["WMA"][4:], mode='lines', name='WMA'))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='#0079a3', opacity=0.4))

    fig.update_layout(title=f'Stock {symbol} Closing Price', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    # Candlestick with WMA
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index[4:], y=df["WMA"][4:], mode='lines', name='WMA'))

    fig.update_layout(title=f'Stock {symbol} Candlestick Chart with WMA', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)