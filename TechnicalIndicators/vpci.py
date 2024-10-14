import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_vpci(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This program allows you to view the Volume Price Confirmation Indicator(VPCI) of a ticker over time. '''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Volume Price Confirmation Indicator
    short_term = 5
    long_term = 20
    vwma_lt = ((df["Adj Close"] * df["Volume"]) + (df["Adj Close"].shift(1) * df["Volume"].shift(1)) + (df["Adj Close"].shift(2) * df["Volume"].shift(2))) / (df["Volume"].rolling(long_term).sum())
    vwma_st = ((df["Adj Close"] * df["Volume"]) + (df["Adj Close"].shift(1) * df["Volume"].shift(1)) + (df["Adj Close"].shift(2) * df["Volume"].shift(2))) / (df["Volume"].rolling(short_term).sum())
    vpc = vwma_lt - df["Adj Close"].rolling(long_term).mean()
    vpr = vwma_st / df["Adj Close"].rolling(short_term).mean()
    vm = df["Adj Close"].rolling(short_term).mean() / df["Adj Close"].rolling(long_term).mean()
    vpci = vpc * vpr * vm

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=vpci, mode='lines', name='Volume Price Confirmation Indicator'))
    fig.update_layout(title="Stock " + symbol + " Volume Price Confirmation Indicator",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with Volume Price Confirmation Indicator
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=vpci, mode='lines', name='Volume Price Confirmation Indicator'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Volume Price Confirmation Indicator",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)

def norm_vpci(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Volume Price Confirmation Indicator
    short_term = 5
    long_term = 20
    vwma_lt = ((df["Adj Close"] * df["Volume"]) + (df["Adj Close"].shift(1) * df["Volume"].shift(1)) + (df["Adj Close"].shift(2) * df["Volume"].shift(2))) / (df["Volume"].rolling(long_term).sum())
    vwma_st = ((df["Adj Close"] * df["Volume"]) + (df["Adj Close"].shift(1) * df["Volume"].shift(1)) + (df["Adj Close"].shift(2) * df["Volume"].shift(2))) / (df["Volume"].rolling(short_term).sum())
    vpc = vwma_lt - df["Adj Close"].rolling(long_term).mean()
    vpr = vwma_st / df["Adj Close"].rolling(short_term).mean()
    vm = df["Adj Close"].rolling(short_term).mean() / df["Adj Close"].rolling(long_term).mean()
    vpci = vpc * vpr * vm

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=vpci, mode='lines', name='Volume Price Confirmation Indicator'))
    fig.update_layout(title="Stock " + symbol + " Volume Price Confirmation Indicator",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with Volume Price Confirmation Indicator
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=vpci, mode='lines', name='Volume Price Confirmation Indicator'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Volume Price Confirmation Indicator",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)
