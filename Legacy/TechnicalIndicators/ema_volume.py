import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_ema_volume(start_date: dt.time, end_date: dt.time, ticker: str):
    ''' This tool plots the candlestick chart of a stock along with the Exponential Moving Average (EMA) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 15
    df["EMA"] = (
        df["Adj Close"].ewm(ignore_na=False, span=n, min_periods=n, adjust=True).mean()
    )

    df["EMA"] = df["Adj Close"].ewm(span=n, adjust=False).mean()  # Recalculate EMA
    df["VolumePositive"] = df["Open"] < df["Adj Close"]

    # Create traces
    trace_candlestick = go.Candlestick(x=df.index,
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name='Candlestick')

    trace_volume = go.Bar(x=df.index, y=df['Volume'],
                        marker=dict(color=df['VolumePositive'].map({True: 'green', False: 'red'})),
                        name='Volume')

    trace_ema = go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA', line=dict(color='green'))

    # Create layout
    layout = go.Layout(title="Stock " + symbol + " Closing Price",
                    xaxis=dict(title="Date", type='date', tickformat='%d-%m-%Y'),
                    yaxis=dict(title="Price"),
                    yaxis2=dict(title="Volume", overlaying='y', side='right'))

    # Create figure
    fig = go.Figure(data=[trace_candlestick, trace_ema, trace_volume], layout=layout)

    # Display Plotly figure in Streamlit
    st.plotly_chart(fig)


def norm_ema_volume(start_date: dt.time, end_date: dt.time, ticker: str):
    ''' This tool plots the candlestick chart of a stock along with the Exponential Moving Average (EMA) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 15
    df["EMA"] = (
        df["Adj Close"].ewm(ignore_na=False, span=n, min_periods=n, adjust=True).mean()
    )

    df["EMA"] = df["Adj Close"].ewm(span=n, adjust=False).mean()  # Recalculate EMA
    df["VolumePositive"] = df["Open"] < df["Adj Close"]

    # Create traces
    trace_candlestick = go.Candlestick(x=df.index,
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name='Candlestick')

    trace_volume = go.Bar(x=df.index, y=df['Volume'],
                        marker=dict(color=df['VolumePositive'].map({True: 'green', False: 'red'})),
                        name='Volume')

    trace_ema = go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA', line=dict(color='green'))

    # Create layout
    layout = go.Layout(title="Stock " + symbol + " Closing Price",
                    xaxis=dict(title="Date", type='date', tickformat='%d-%m-%Y'),
                    yaxis=dict(title="Price"),
                    yaxis2=dict(title="Volume", overlaying='y', side='right'))

    # Create figure
    fig = go.Figure(data=[trace_candlestick, trace_ema, trace_volume], layout=layout)

    # Display Plotly figure in Streamlit
    st.plotly_chart(fig)