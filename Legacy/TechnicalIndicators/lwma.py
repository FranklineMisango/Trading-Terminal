import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_lwma(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Linearly Weighted Moving Average (LWMA) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Linear Weighted Moving Average (LWMA)
    def linear_weight_moving_average(close, n):
        lwma = [np.nan] * n
        for i in range(n, len(close)):
            lwma.append(
                (close[i - n : i] * (np.arange(n) + 1)).sum() / (np.arange(n) + 1).sum()
            )
        return lwma

    period = 14
    df["LWMA"] = linear_weight_moving_average(df["Adj Close"], period)

    # Plot Linear Weighted Moving Average (LWMA)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['LWMA'], mode='lines', name='LWMA'))
    fig.update_layout(title=f"Linear Weighted Moving Average (LWMA) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_lwma(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Linear Weighted Moving Average (LWMA)
    def linear_weight_moving_average(close, n):
        lwma = [np.nan] * n
        for i in range(n, len(close)):
            lwma.append(
                (close[i - n : i] * (np.arange(n) + 1)).sum() / (np.arange(n) + 1).sum()
            )
        return lwma

    period = 14
    df["LWMA"] = linear_weight_moving_average(df["Adj Close"], period)

    # Plot Linear Weighted Moving Average (LWMA)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['LWMA'], mode='lines', name='LWMA'))
    fig.update_layout(title=f"Linear Weighted Moving Average (LWMA) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)