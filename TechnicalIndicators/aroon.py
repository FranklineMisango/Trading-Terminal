import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool
import numpy as np

@tool
def tool_aroon(ticker: str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots the Aroon Indicator of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    df = yf.download(symbol, start, end)

    n = 25
    high_max = lambda xs: np.argmax(xs[::-1])
    low_min = lambda xs: np.argmin(xs[::-1])

    df["Days since last High"] = (
        df["High"]
        .rolling(center=False, min_periods=0, window=n)
        .apply(func=high_max)
        .astype(int)
    )

    df["Days since last Low"] = (
        df["Low"]
        .rolling(center=False, min_periods=0, window=n)
        .apply(func=low_min)
        .astype(int)
    )

    df["Aroon_Up"] = ((25 - df["Days since last High"]) / 25) * 100
    df["Aroon_Down"] = ((25 - df["Days since last Low"]) / 25) * 100

    df = df.drop(["Days since last High", "Days since last Low"], axis=1)

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Up"], mode='lines', name='Aroon UP', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Down"], mode='lines', name='Aroon DOWN', line=dict(color='red')))
    fig.update_layout(yaxis2=dict(title="Aroon"))

    st.plotly_chart(fig)

    # Candlestick Chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Up"], mode='lines', name='Aroon UP', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Down"], mode='lines', name='Aroon DOWN', line=dict(color='red')))
    fig.update_layout(yaxis2=dict(title="Aroon"))

    st.plotly_chart(fig)


def norm_aroon(ticker,start_date,end_date):
    symbol = ticker
    start = start_date
    end = end_date

    df = yf.download(symbol, start, end)

    n = 25
    high_max = lambda xs: np.argmax(xs[::-1])
    low_min = lambda xs: np.argmin(xs[::-1])

    df["Days since last High"] = (
        df["High"]
        .rolling(center=False, min_periods=0, window=n)
        .apply(func=high_max)
        .astype(int)
    )

    df["Days since last Low"] = (
        df["Low"]
        .rolling(center=False, min_periods=0, window=n)
        .apply(func=low_min)
        .astype(int)
    )

    df["Aroon_Up"] = ((25 - df["Days since last High"]) / 25) * 100
    df["Aroon_Down"] = ((25 - df["Days since last Low"]) / 25) * 100

    df = df.drop(["Days since last High", "Days since last Low"], axis=1)

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Up"], mode='lines', name='Aroon UP', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Down"], mode='lines', name='Aroon DOWN', line=dict(color='red')))
    fig.update_layout(yaxis2=dict(title="Aroon"))

    st.plotly_chart(fig)

    # Candlestick Chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Up"], mode='lines', name='Aroon UP', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Down"], mode='lines', name='Aroon DOWN', line=dict(color='red')))
    fig.update_layout(yaxis2=dict(title="Aroon"))

    st.plotly_chart(fig)