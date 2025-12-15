import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool
import numpy as np

@tool
def tool_aroon_oscillator(ticker: str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots the Aroon Oscillator of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
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

    df["Aroon_Oscillator"] = df["Aroon_Up"] - df["Aroon_Down"]

    df = df.drop(
        ["Days since last High", "Days since last Low", "Aroon_Up", "Aroon_Down"], axis=1
    )

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Oscillator"], mode='lines', name='Aroon Oscillator', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig)

    # Bar Chart
    fig = go.Figure()
    df["Positive"] = df["Aroon_Oscillator"] > 0
    fig.add_trace(go.Bar(x=df.index, y=df["Aroon_Oscillator"], marker_color=df.Positive.map({True: "green", False: "red"})))
    fig.add_shape(type="line", x0=df.index[0], y0=0, x1=df.index[-1], y1=0, line=dict(color="red", width=1, dash="dash"))

    fig.update_layout(title="Aroon Oscillator",
                    xaxis_title="Date",
                    yaxis_title="Aroon Oscillator",
                    legend=dict(x=0, y=1, traceorder="normal"))

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

    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Oscillator"], mode='lines', name='Aroon Oscillator', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig)

def norm_aroon_oscillator(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
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

    df["Aroon_Oscillator"] = df["Aroon_Up"] - df["Aroon_Down"]

    df = df.drop(
        ["Days since last High", "Days since last Low", "Aroon_Up", "Aroon_Down"], axis=1
    )

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Oscillator"], mode='lines', name='Aroon Oscillator', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig)

    # Bar Chart
    fig = go.Figure()
    df["Positive"] = df["Aroon_Oscillator"] > 0
    fig.add_trace(go.Bar(x=df.index, y=df["Aroon_Oscillator"], marker_color=df.Positive.map({True: "green", False: "red"})))
    fig.add_shape(type="line", x0=df.index[0], y0=0, x1=df.index[-1], y1=0, line=dict(color="red", width=1, dash="dash"))

    fig.update_layout(title="Aroon Oscillator",
                    xaxis_title="Date",
                    yaxis_title="Aroon Oscillator",
                    legend=dict(x=0, y=1, traceorder="normal"))

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

    fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Oscillator"], mode='lines', name='Aroon Oscillator', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig)