import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_hma(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to view the Hull Moving Average (HMA) of a ticker over time. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Compute Hull Moving Average
    period = 20
    df['WMA'] = df['Adj Close'].rolling(window=period).mean()
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    df['Weighted_MA'] = df['Adj Close'].rolling(window=half_period).mean() * 2 - df['Adj Close'].rolling(window=period).mean()
    df['HMA'] = df['Weighted_MA'].rolling(window=sqrt_period).mean()

    # Plot Hull Moving Average with Candlestick graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], mode='lines', name='Hull Moving Average'))
    fig.update_layout(title=f"Hull Moving Average (HMA) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_hma(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Compute Hull Moving Average
    period = 20
    df['WMA'] = df['Adj Close'].rolling(window=period).mean()
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    df['Weighted_MA'] = df['Adj Close'].rolling(window=half_period).mean() * 2 - df['Adj Close'].rolling(window=period).mean()
    df['HMA'] = df['Weighted_MA'].rolling(window=sqrt_period).mean()

    # Plot Hull Moving Average with Candlestick graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], mode='lines', name='Hull Moving Average'))
    fig.update_layout(title=f"Hull Moving Average (HMA) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)