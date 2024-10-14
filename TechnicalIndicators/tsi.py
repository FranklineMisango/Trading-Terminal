import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_tsi(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This program allows you to view the True Strength Index (TSI) of a ticker over time. '''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate True Strength Index (TSI)
    df["PC"] = df["Adj Close"] - df["Adj Close"].shift()
    df["EMA_FS"] = df["PC"].ewm(span=25, min_periods=25, adjust=True).mean()
    df["EMA_SS"] = df["EMA_FS"].ewm(span=13, min_periods=13, adjust=True).mean()
    df["Absolute_PC"] = abs(df["Adj Close"] - df["Adj Close"].shift())
    df["Absolute_FS"] = df["Absolute_PC"].ewm(span=25, min_periods=25).mean()
    df["Absolute_SS"] = df["Absolute_FS"].ewm(span=13, min_periods=13).mean()
    df["TSI"] = 100 * df["EMA_SS"] / df["Absolute_SS"]
    df = df.drop(["PC", "EMA_FS", "EMA_SS", "Absolute_PC", "Absolute_FS", "Absolute_SS"], axis=1)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df["TSI"], mode='lines', name='True Strength Index'))
    fig.update_layout(title="Stock " + symbol + " True Strength Index",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with True Strength Index
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=df["TSI"], mode='lines', name='True Strength Index'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with True Strength Index",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)

def norm_tsi(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate True Strength Index (TSI)
    df["PC"] = df["Adj Close"] - df["Adj Close"].shift()
    df["EMA_FS"] = df["PC"].ewm(span=25, min_periods=25, adjust=True).mean()
    df["EMA_SS"] = df["EMA_FS"].ewm(span=13, min_periods=13, adjust=True).mean()
    df["Absolute_PC"] = abs(df["Adj Close"] - df["Adj Close"].shift())
    df["Absolute_FS"] = df["Absolute_PC"].ewm(span=25, min_periods=25).mean()
    df["Absolute_SS"] = df["Absolute_FS"].ewm(span=13, min_periods=13).mean()
    df["TSI"] = 100 * df["EMA_SS"] / df["Absolute_SS"]
    df = df.drop(["PC", "EMA_FS", "EMA_SS", "Absolute_PC", "Absolute_FS", "Absolute_SS"], axis=1)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df["TSI"], mode='lines', name='True Strength Index'))
    fig.update_layout(title="Stock " + symbol + " True Strength Index",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with True Strength Index
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=df["TSI"], mode='lines', name='True Strength Index'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with True Strength Index",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)
