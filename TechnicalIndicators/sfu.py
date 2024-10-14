import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_sfu(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Stochastic Full(SFU) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Stochastic Full
    low_min = dataset['Low'].rolling(window=14).min()
    high_max = dataset['High'].rolling(window=14).max()
    dataset['%K'] = 100 * (dataset['Close'] - low_min) / (high_max - low_min)
    dataset['%D'] = dataset['%K'].rolling(window=3).mean()
    dataset['%D_full'] = dataset['%D'].rolling(window=3).mean()

    # Plot Stochastic Full
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["%K"], mode='lines', name='Stochastic Full %K', line=dict(color='red')),
                        go.Scatter(x=dataset.index, y=dataset["%D_full"], mode='lines', name='Stochastic Full %D', line=dict(color='blue'))])
    fig.update_layout(title=f"{symbol} Stochastic Full",
                    xaxis_title="Date",
                    yaxis_title="Stochastic Full")
    st.plotly_chart(fig)

def norm_sfu(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Stochastic Full
    low_min = dataset['Low'].rolling(window=14).min()
    high_max = dataset['High'].rolling(window=14).max()
    dataset['%K'] = 100 * (dataset['Close'] - low_min) / (high_max - low_min)
    dataset['%D'] = dataset['%K'].rolling(window=3).mean()
    dataset['%D_full'] = dataset['%D'].rolling(window=3).mean()

    # Plot Stochastic Full
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["%K"], mode='lines', name='Stochastic Full %K', line=dict(color='red')),
                        go.Scatter(x=dataset.index, y=dataset["%D_full"], mode='lines', name='Stochastic Full %D', line=dict(color='blue'))])
    fig.update_layout(title=f"{symbol} Stochastic Full",
                    xaxis_title="Date",
                    yaxis_title="Stochastic Full")
    st.plotly_chart(fig)