
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
def tool_vwap(ticker:str, start_date:dt.time, end_date:dt.time):
    '''Tool for Volume Weighted Average Price (VWAP)'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    def VWAP(df):
        return (df["Adj Close"] * df["Volume"]).sum() / df["Volume"].sum()

    n = 14
    df["VWAP"] = pd.concat(
        [
            (pd.Series(VWAP(df.iloc[i : i + n]), index=[df.index[i + n]]))
            for i in range(len(df) - n)
        ]
    )

    vwap_series = pd.concat([(pd.Series(VWAP(df.iloc[i : i + n]), index=[df.index[i + n]])) for i in range(len(df) - n)])
    vwap_series = vwap_series.dropna()


    # Simple Line Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.add_trace(go.Scatter(x=vwap_series.index, y=vwap_series, mode='lines', name='VWAP'))
    fig1.update_layout(title="Volume Weighted Average Price for Stock", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig1)

    # Candlestick with VWAP
    df.loc[:, "VolumePositive"] = df["Open"] < df["Adj Close"]

    df = df.dropna()
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].apply(mdates.date2num)

    fig2 = go.Figure()

    fig2.add_trace(go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Adj Close'], name='Candlestick'))

    fig2.add_trace(go.Scatter(x=df['Date'], y=df["VWAP"], mode='lines', name='VWAP'))

    fig2.update_layout(title=f"Stock {symbol} Closing Price with VWAP", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2)


def norm_vwap(ticker:str, start_date:dt.time, end_date:dt.time):
    '''Tool for Volume Weighted Average Price (VWAP)'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    def VWAP(df):
        return (df["Adj Close"] * df["Volume"]).sum() / df["Volume"].sum()

    n = 14
    df["VWAP"] = pd.concat(
        [
            (pd.Series(VWAP(df.iloc[i : i + n]), index=[df.index[i + n]]))
            for i in range(len(df) - n)
        ]
    )

    vwap_series = pd.concat([(pd.Series(VWAP(df.iloc[i : i + n]), index=[df.index[i + n]])) for i in range(len(df) - n)])
    vwap_series = vwap_series.dropna()


    # Simple Line Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.add_trace(go.Scatter(x=vwap_series.index, y=vwap_series, mode='lines', name='VWAP'))
    fig1.update_layout(title="Volume Weighted Average Price for Stock", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig1)

    # Candlestick with VWAP
    df.loc[:, "VolumePositive"] = df["Open"] < df["Adj Close"]

    df = df.dropna()
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].apply(mdates.date2num)

    fig2 = go.Figure()

    fig2.add_trace(go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Adj Close'], name='Candlestick'))

    fig2.add_trace(go.Scatter(x=df['Date'], y=df["VWAP"], mode='lines', name='VWAP'))

    fig2.update_layout(title=f"Stock {symbol} Closing Price with VWAP", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2)

