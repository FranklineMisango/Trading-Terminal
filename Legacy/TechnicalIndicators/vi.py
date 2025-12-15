import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_vi(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to view the Variance Indicator(VI) of a ticker over time. '''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Variance Indicator
    n = 14
    df["Variance"] = df["Adj Close"].rolling(n).var()

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Variance"], mode='lines', name='Variance Indicator'))
    fig.update_layout(title="Stock " + symbol + " Variance Indicator",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with Variance Indicator
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=df["Variance"], mode='lines', name='Variance Indicator'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Variance Indicator",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)

def norm_vi(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Variance Indicator
    n = 14
    df["Variance"] = df["Adj Close"].rolling(n).var()

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Variance"], mode='lines', name='Variance Indicator'))
    fig.update_layout(title="Stock " + symbol + " Variance Indicator",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with Variance Indicator
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=df["Variance"], mode='lines', name='Variance Indicator'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Variance Indicator",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)
