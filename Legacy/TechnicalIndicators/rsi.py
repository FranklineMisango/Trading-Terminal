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
def tool_rsi(ticker:str, start_date:dt.time, end_date:dt.time):
    '''Tool for Relative Strength Index (RSI)'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    n = 14  # Number of period
    change = df["Adj Close"].diff(1)
    df["Gain"] = change.mask(change < 0, 0)
    df["Loss"] = abs(change.mask(change > 0, 0))
    df["AVG_Gain"] = df.Gain.rolling(n).mean()
    df["AVG_Loss"] = df.Loss.rolling(n).mean()
    df["RS"] = df["AVG_Gain"] / df["AVG_Loss"]
    df["RSI"] = 100 - (100 / (1 + df["RS"]))

    # Create RSI plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))

    fig1.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='Relative Strength Index'))

    fig1.update_layout(title=symbol + " Closing Price and Relative Strength Index (RSI)",
                    xaxis_title="Date",
                    yaxis_title="Price/RSI")

    st.plotly_chart(fig1)

    # Candlestick with RSI
    dfc = df.copy()
    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
    dfc = dfc.reset_index()
    dfc["Date"] = pd.to_datetime(dfc["Date"])
    dfc["Date"] = dfc["Date"].apply(mdates.date2num)

    fig2 = go.Figure()

    # Candlestick chart
    fig2.add_trace(go.Candlestick(x=dfc['Date'],
                    open=dfc['Open'],
                    high=dfc['High'],
                    low=dfc['Low'],
                    close=dfc['Adj Close'], name='Candlestick'))

    # Volume bars
    fig2.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], marker_color=dfc.VolumePositive.map({True: "green", False: "red"}), name='Volume'))

    fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='Relative Strength Index', line=dict(color='blue')))

    fig2.update_layout(title=symbol + " Candlestick Chart with Relative Strength Index (RSI)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False)

    st.plotly_chart(fig2)




def norm_rsi(ticker, start_date, end_date):
    '''Tool for Relative Strength Index (RSI)'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    n = 14  # Number of period
    change = df["Adj Close"].diff(1)
    df["Gain"] = change.mask(change < 0, 0)
    df["Loss"] = abs(change.mask(change > 0, 0))
    df["AVG_Gain"] = df.Gain.rolling(n).mean()
    df["AVG_Loss"] = df.Loss.rolling(n).mean()
    df["RS"] = df["AVG_Gain"] / df["AVG_Loss"]
    df["RSI"] = 100 - (100 / (1 + df["RS"]))

    # Create RSI plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))

    fig1.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='Relative Strength Index'))

    fig1.update_layout(title=symbol + " Closing Price and Relative Strength Index (RSI)",
                    xaxis_title="Date",
                    yaxis_title="Price/RSI")

    st.plotly_chart(fig1)

    # Candlestick with RSI
    dfc = df.copy()
    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
    dfc = dfc.reset_index()
    dfc["Date"] = pd.to_datetime(dfc["Date"])
    dfc["Date"] = dfc["Date"].apply(mdates.date2num)

    fig2 = go.Figure()

    # Candlestick chart
    fig2.add_trace(go.Candlestick(x=dfc['Date'],
                    open=dfc['Open'],
                    high=dfc['High'],
                    low=dfc['Low'],
                    close=dfc['Adj Close'], name='Candlestick'))

    # Volume bars
    fig2.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], marker_color=dfc.VolumePositive.map({True: "green", False: "red"}), name='Volume'))

    fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='Relative Strength Index', line=dict(color='blue')))

    fig2.update_layout(title=symbol + " Candlestick Chart with Relative Strength Index (RSI)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False)

    st.plotly_chart(fig2)