import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_sma(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This program allows you to visualize Smoothed Moving Average(SMA) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    n = 7
    dataset["SMMA"] = dataset["Adj Close"].ewm(alpha=1 / float(n)).mean()

    # Plot Smoothed Moving Average
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["SMMA"], mode='lines', name='Smoothed Moving Average', line=dict(color='red'))])
    fig.update_layout(title=f"{symbol} Smoothed Moving Average",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

def norm_sma(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    n = 7
    dataset["SMMA"] = dataset["Adj Close"].ewm(alpha=1 / float(n)).mean()

    # Plot Smoothed Moving Average
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["SMMA"], mode='lines', name='Smoothed Moving Average', line=dict(color='red'))])
    fig.update_layout(title=f"{symbol} Smoothed Moving Average",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)