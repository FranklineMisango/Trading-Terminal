import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_sdv(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This program allows you to visualize Standard Deviation Volatility(SDV) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    dataset["STD"] = dataset["Adj Close"].rolling(10).std()

    # Plot Standard Deviation Volatility
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["STD"], mode='lines', name='Standard Deviation Volatility', line=dict(color='red'))])
    fig.update_layout(title=f"{symbol} Standard Deviation Volatility",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

def norm_sdv(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    dataset["STD"] = dataset["Adj Close"].rolling(10).std()

    # Plot Standard Deviation Volatility
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["STD"], mode='lines', name='Standard Deviation Volatility', line=dict(color='red'))])
    fig.update_layout(title=f"{symbol} Standard Deviation Volatility",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

    