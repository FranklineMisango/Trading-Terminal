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
def tool_roi(ticker : str, start_date : dt.time, end_date:dt.time):
    ''' This tool plots the candlestick chart of a stock along with the Return on Investment (ROI) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Return on Investment (ROI)
    df["ROI"] = ((df["Adj Close"] - df["Adj Close"].shift(1)) / df["Adj Close"].shift(1) * 100)

    # Line Chart with ROI
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.add_trace(go.Scatter(x=df.index, y=df["ROI"], mode='lines', name='Return on Investment'))

    fig1.update_layout(title="Adj Close and Return on Investment (ROI) Over Time",
                    xaxis_title="Date",
                    yaxis_title="Price/ROI")

    st.plotly_chart(fig1)

    # Candlestick with ROI
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

    fig2.add_trace(go.Scatter(x=df.index, y=df["ROI"], mode='lines', name='Return on Investment', line=dict(color='red')))

    fig2.update_layout(title="Candlestick Chart with Return on Investment (ROI)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False)

    st.plotly_chart(fig2)


def norm_roi(ticker : str, start_date : dt.time, end_date:dt.time):
    ''' This tool plots the candlestick chart of a stock along with the Return on Investment (ROI) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Return on Investment (ROI)
    df["ROI"] = ((df["Adj Close"] - df["Adj Close"].shift(1)) / df["Adj Close"].shift(1) * 100)

    # Line Chart with ROI
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.add_trace(go.Scatter(x=df.index, y=df["ROI"], mode='lines', name='Return on Investment'))

    fig1.update_layout(title="Adj Close and Return on Investment (ROI) Over Time",
                    xaxis_title="Date",
                    yaxis_title="Price/ROI")

    st.plotly_chart(fig1)

    # Candlestick with ROI
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

    fig2.add_trace(go.Scatter(x=df.index, y=df["ROI"], mode='lines', name='Return on Investment', line=dict(color='red')))

    fig2.update_layout(title="Candlestick Chart with Return on Investment (ROI)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False)

    st.plotly_chart(fig2)
