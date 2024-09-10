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
def tool_wsma(ticker:str, start_date:dt.time, end_date:dt.time):
    '''Tool for Wilder's Smoothing Moving Average (WSMA)'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    def WSMA(df, column="Adj Close", n=14):
        ema = df[column].ewm(span=n, min_periods=n - 1).mean()
        K = 1 / n
        wsma = df[column] * K + ema * (1 - K)
        return wsma

    df["WSMA"] = WSMA(df, column="Adj Close", n=14)
    df = df.dropna()

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df["WSMA"], mode='lines', name="WSMA"))
    fig.update_layout(title="Wilder's Smoothing Moving Average for Stock",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick with WSMA
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index, y=df["WSMA"], mode='lines', name="WSMA"))

    fig.update_layout(title=f"Stock {symbol} Candlestick Chart with WSMA",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)       

def norm_wsma(ticker:str, start_date:dt.time, end_date:dt.time):
    '''Tool for Wilder's Smoothing Moving Average (WSMA)'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    def WSMA(df, column="Adj Close", n=14):
        ema = df[column].ewm(span=n, min_periods=n - 1).mean()
        K = 1 / n
        wsma = df[column] * K + ema * (1 - K)
        return wsma

    df["WSMA"] = WSMA(df, column="Adj Close", n=14)
    df = df.dropna()

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df["WSMA"], mode='lines', name="WSMA"))
    fig.update_layout(title="Wilder's Smoothing Moving Average for Stock",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick with WSMA
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index, y=df["WSMA"], mode='lines', name="WSMA"))

    fig.update_layout(title=f"Stock {symbol} Candlestick Chart with WSMA",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig) 