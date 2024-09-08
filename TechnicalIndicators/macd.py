import datetime as dt
import yfinance as yf
import streamlit as st
from langchain_core.tools import tool
from plotly import graph_objs as go
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
from ta import add_all_ta_features
import ta


@tool
def tool_macd(start_date:dt.time, end_date:dt.time, ticker:str):
    ''' This tool plots the candlestick chart of a stock along with the Moving Average Convergence Divergence (MACD) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)
    # Read data
    df = yf.download(symbol, start, end)

    df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(
        df["Adj Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df = df.dropna()
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")

    # Line Chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_hline(y=df["Adj Close"].mean(), line_dash="dash", line_color="red", name='Mean')

    # Volume
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=df['Volume'].map(lambda x: 'green' if x > 0 else 'red')))
    # MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"], mode='lines', name='MACD'))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["macdsignal"], mode='lines', name='Signal'))
    fig_macd.add_trace(go.Bar(x=df.index, y=df["macdhist"], name='Histogram', marker_color=df['macdhist'].map(lambda x: 'green' if x > 0 else 'red')))

    st.plotly_chart(fig_line)
    st.plotly_chart(fig_volume)
    st.plotly_chart(fig_macd)


#normal

def norm_macd(start_date, end_date, ticker):
    ''' This tool plots the candlestick chart of a stock along with the Moving Average Convergence Divergence (MACD) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)
    # Read data
    df = yf.download(symbol, start, end)

    df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(
        df["Adj Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df = df.dropna()
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")

    # Line Chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_hline(y=df["Adj Close"].mean(), line_dash="dash", line_color="red", name='Mean')

    # Volume
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=df['Volume'].map(lambda x: 'green' if x > 0 else 'red')))
    # MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"], mode='lines', name='MACD'))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["macdsignal"], mode='lines', name='Signal'))
    fig_macd.add_trace(go.Bar(x=df.index, y=df["macdhist"], name='Histogram', marker_color=df['macdhist'].map(lambda x: 'green' if x > 0 else 'red')))

    st.plotly_chart(fig_line)
    st.plotly_chart(fig_volume)
    st.plotly_chart(fig_macd)