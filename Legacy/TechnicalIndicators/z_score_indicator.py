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
def tool_z_score(ticker:str, start_date:dt.time, end_date:dt.time):
    '''This tool is for calculating the zscore'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Read data
    df = yf.download(symbol, start, end)

    from scipy.stats import zscore

    df["z_score"] = zscore(df["Adj Close"])

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    # Z-Score Chart
    fig.add_trace(go.Scatter(x=df.index, y=df["z_score"], mode='lines', name="Z-Score"))
    fig.update_layout(title="Z-Score for " + symbol,
                    xaxis_title="Date",
                    yaxis_title="Z-Score",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick with Z-Score
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index, y=df["z_score"], mode='lines', name="Z-Score"))
    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with Z-Score",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    
    st.plotly_chart(fig)


def norm_z_score(ticker,str, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Read data
    df = yf.download(symbol, start, end)

    from scipy.stats import zscore

    df["z_score"] = zscore(df["Adj Close"])

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    # Z-Score Chart
    fig.add_trace(go.Scatter(x=df.index, y=df["z_score"], mode='lines', name="Z-Score"))
    fig.update_layout(title="Z-Score for " + symbol,
                    xaxis_title="Date",
                    yaxis_title="Z-Score",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick with Z-Score
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index, y=df["z_score"], mode='lines', name="Z-Score"))
    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with Z-Score",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    
    st.plotly_chart(fig)