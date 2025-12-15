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
def tool_pvi(start_date: dt.time, end_date: dt.time, ticker: str):
    ''' This tool plots the candlestick chart of a stock along with the Positive Volume Index (PVI) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date  
    # Read data
    df = yf.download(symbol, start, end)

    returns = df["Adj Close"].pct_change()
    vol_increase = df["Volume"].shift(1) < df["Volume"]
    pvi = pd.Series(data=np.nan, index=df["Adj Close"].index, dtype="float64")

    pvi.iloc[0] = 1000
    for i in range(1, len(pvi)):
        if vol_increase.iloc[i]:
            pvi.iloc[i] = pvi.iloc[i - 1] * (1.0 + returns.iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]

    pvi = pvi.replace([np.inf, -np.inf], np.nan).fillna(1000)

    df["PVI"] = pd.Series(pvi)

    # Line Chart with PVI
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.add_trace(go.Scatter(x=df.index, y=df["PVI"], mode='lines', name='Positive Volume Index'))

    fig1.update_layout(title="Adj Close and Positive Volume Index (PVI) Over Time",
                    xaxis_title="Date",
                    yaxis_title="Price/PVI")

    st.plotly_chart(fig1)

    # Candlestick with PVI
    dfc = df.copy()
    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
    dfc = dfc.reset_index()
    dfc["Date"] = mdates.date2num(dfc["Date"].tolist())

    fig2 = go.Figure()

    # Candlestick chart
    fig2.add_trace(go.Candlestick(x=dfc['Date'],
                    open=dfc['Open'],
                    high=dfc['High'],
                    low=dfc['Low'],
                    close=dfc['Adj Close'], name='Candlestick'))

    # Volume bars
    fig2.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], marker_color=dfc.VolumePositive.map({True: "green", False: "red"}), name='Volume'))

    fig2.add_trace(go.Scatter(x=df.index, y=df["PVI"], mode='lines', name='Positive Volume Index', line=dict(color='green')))

    fig2.update_layout(title="Candlestick Chart with Positive Volume Index (PVI)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False)

    st.plotly_chart(fig2)


def norm_pvi(start_date, end_date, ticker):
    ''' This tool plots the candlestick chart of a stock along with the Positive Volume Index (PVI) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date  
    # Read data
    df = yf.download(symbol, start, end)

    returns = df["Adj Close"].pct_change()
    vol_increase = df["Volume"].shift(1) < df["Volume"]
    pvi = pd.Series(data=np.nan, index=df["Adj Close"].index, dtype="float64")

    pvi.iloc[0] = 1000
    for i in range(1, len(pvi)):
        if vol_increase.iloc[i]:
            pvi.iloc[i] = pvi.iloc[i - 1] * (1.0 + returns.iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]

    pvi = pvi.replace([np.inf, -np.inf], np.nan).fillna(1000)

    df["PVI"] = pd.Series(pvi)

    # Line Chart with PVI
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.add_trace(go.Scatter(x=df.index, y=df["PVI"], mode='lines', name='Positive Volume Index'))

    fig1.update_layout(title="Adj Close and Positive Volume Index (PVI) Over Time",
                    xaxis_title="Date",
                    yaxis_title="Price/PVI")

    st.plotly_chart(fig1)

    # Candlestick with PVI
    dfc = df.copy()
    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
    dfc = dfc.reset_index()
    dfc["Date"] = mdates.date2num(dfc["Date"].tolist())

    fig2 = go.Figure()

    # Candlestick chart
    fig2.add_trace(go.Candlestick(x=dfc['Date'],
                    open=dfc['Open'],
                    high=dfc['High'],
                    low=dfc['Low'],
                    close=dfc['Adj Close'], name='Candlestick'))

    # Volume bars
    fig2.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], marker_color=dfc.VolumePositive.map({True: "green", False: "red"}), name='Volume'))

    fig2.add_trace(go.Scatter(x=df.index, y=df["PVI"], mode='lines', name='Positive Volume Index', line=dict(color='green')))

    fig2.update_layout(title="Candlestick Chart with Positive Volume Index (PVI)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False)

    st.plotly_chart(fig2)