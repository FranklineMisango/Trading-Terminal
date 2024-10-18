mport datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_vwma(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to view the Volume Weighted Moving Average (VWMA) of a ticker over time. '''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Volume Weighted Moving Average (VWMA)
    df["Volume_x_Close"] = df["Volume"] * df["Close"]
    df["VWMA"] = df["Volume_x_Close"].rolling(window=14).sum() / df["Volume"].rolling(window=14).sum()

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df["VWMA"], mode='lines', name='Volume Weighted Moving Average (VWMA)'))
    fig.update_layout(title="Stock " + symbol + " Volume Weighted Moving Average (VWMA)",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with Volume Weighted Moving Average (VWMA)
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=df["VWMA"], mode='lines', name='Volume Weighted Moving Average (VWMA)'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Volume Weighted Moving Average (VWMA)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)

def norm_vwma(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Volume Weighted Moving Average (VWMA)
    df["Volume_x_Close"] = df["Volume"] * df["Close"]
    df["VWMA"] = df["Volume_x_Close"].rolling(window=14).sum() / df["Volume"].rolling(window=14).sum()

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df["VWMA"], mode='lines', name='Volume Weighted Moving Average (VWMA)'))
    fig.update_layout(title="Stock " + symbol + " Volume Weighted Moving Average (VWMA)",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with Volume Weighted Moving Average (VWMA)
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=df["VWMA"], mode='lines', name='Volume Weighted Moving Average (VWMA)'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Volume Weighted Moving Average (VWMA)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)