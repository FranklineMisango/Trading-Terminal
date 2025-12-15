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
def tool_pvt(start_date: dt.time, end_date: dt.time, ticker: str):
    ''' This tool plots the candlestick chart of a stock along with the Price Volume Trend (PVT) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    df = yf.download(symbol, start, end)

    # Calculate Price Volume Trend (PVT)
    df["Momentum_1D"] = (df["Adj Close"] - df["Adj Close"].shift(1)).fillna(0)
    df["PVT"] = (df["Momentum_1D"] / df["Adj Close"].shift(1)) * df["Volume"]
    df["PVT"] = df["PVT"] - df["PVT"].shift(1)
    df["PVT"] = df["PVT"].fillna(0)

    # Line Chart with PVT
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.add_trace(go.Scatter(x=df.index, y=df["PVT"], mode='lines', name='Price Volume Trend'))

    fig1.update_layout(title="Adj Close and Price Volume Trend (PVT) Over Time",
                    xaxis_title="Date",
                    yaxis_title="Price/PVT")

    st.plotly_chart(fig1)

    # Candlestick with PVT
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

    fig2.add_trace(go.Scatter(x=df.index, y=df["PVT"], mode='lines', name='Price Volume Trend', line=dict(color='blue')))

    fig2.update_layout(title="Candlestick Chart with Price Volume Trend (PVT)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False)

    st.plotly_chart(fig2)


def norm_pvt(start_date, end_date , ticker):
    ''' This tool plots the candlestick chart of a stock along with the Price Volume Trend (PVT) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    df = yf.download(symbol, start, end)

    # Calculate Price Volume Trend (PVT)
    df["Momentum_1D"] = (df["Adj Close"] - df["Adj Close"].shift(1)).fillna(0)
    df["PVT"] = (df["Momentum_1D"] / df["Adj Close"].shift(1)) * df["Volume"]
    df["PVT"] = df["PVT"] - df["PVT"].shift(1)
    df["PVT"] = df["PVT"].fillna(0)

    # Line Chart with PVT
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.add_trace(go.Scatter(x=df.index, y=df["PVT"], mode='lines', name='Price Volume Trend'))

    fig1.update_layout(title="Adj Close and Price Volume Trend (PVT) Over Time",
                    xaxis_title="Date",
                    yaxis_title="Price/PVT")

    st.plotly_chart(fig1)

    # Candlestick with PVT
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

    fig2.add_trace(go.Scatter(x=df.index, y=df["PVT"], mode='lines', name='Price Volume Trend', line=dict(color='blue')))

    fig2.update_layout(title="Candlestick Chart with Price Volume Trend (PVT)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False)

    st.plotly_chart(fig2)