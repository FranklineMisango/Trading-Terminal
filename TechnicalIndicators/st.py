import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_st(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This program allows you to view the Super Trend(ST) of a ticker over time. '''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    n = 7  # Number of periods
    df["H-L"] = abs(df["High"] - df["Low"])
    df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(n).mean()

    df["Upper Basic"] = (df["High"] + df["Low"]) / 2 + (2 * df["ATR"])
    df["Lower Basic"] = (df["High"] + df["Low"]) / 2 - (2 * df["ATR"])

    df["Upper Band"] = df["Upper Basic"]
    df["Lower Band"] = df["Lower Basic"]

    for i in range(n, len(df)):
        if df["Close"][i - 1] <= df["Upper Band"][i - 1]:
            df["Upper Band"][i] = min(df["Upper Basic"][i], df["Upper Band"][i - 1])
        else:
            df["Upper Band"][i] = df["Upper Basic"][i]

    for i in range(n, len(df)):
        if df["Close"][i - 1] >= df["Lower Band"][i - 1]:
            df["Lower Band"][i] = max(df["Lower Basic"][i], df["Lower Band"][i - 1])
        else:
            df["Lower Band"][i] = df["Lower Basic"][i]

    df["SuperTrend"] = 0.00
    for i in range(n, len(df)):
        if df["Close"][i] <= df["Upper Band"][i]:
            df["SuperTrend"][i] = df["Upper Band"][i]
        elif df["Close"][i] > df["Upper Band"][i]:
            df["SuperTrend"][i] = df["Lower Band"][i]

    # Candlestick Chart with Super Trend
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with Super Trend",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["SuperTrend"], mode='lines', name='SuperTrend'))
    st.plotly_chart(fig)

def norm_st(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    n = 7  # Number of periods
    df["H-L"] = abs(df["High"] - df["Low"])
    df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(n).mean()

    df["Upper Basic"] = (df["High"] + df["Low"]) / 2 + (2 * df["ATR"])
    df["Lower Basic"] = (df["High"] + df["Low"]) / 2 - (2 * df["ATR"])

    df["Upper Band"] = df["Upper Basic"]
    df["Lower Band"] = df["Lower Basic"]

    for i in range(n, len(df)):
        if df["Close"][i - 1] <= df["Upper Band"][i - 1]:
            df["Upper Band"][i] = min(df["Upper Basic"][i], df["Upper Band"][i - 1])
        else:
            df["Upper Band"][i] = df["Upper Basic"][i]

    for i in range(n, len(df)):
        if df["Close"][i - 1] >= df["Lower Band"][i - 1]:
            df["Lower Band"][i] = max(df["Lower Basic"][i], df["Lower Band"][i - 1])
        else:
            df["Lower Band"][i] = df["Lower Basic"][i]

    df["SuperTrend"] = 0.00
    for i in range(n, len(df)):
        if df["Close"][i] <= df["Upper Band"][i]:
            df["SuperTrend"][i] = df["Upper Band"][i]
        elif df["Close"][i] > df["Upper Band"][i]:
            df["SuperTrend"][i] = df["Lower Band"][i]

    # Candlestick Chart with Super Trend
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with Super Trend",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["SuperTrend"], mode='lines', name='SuperTrend'))
    st.plotly_chart(fig)

