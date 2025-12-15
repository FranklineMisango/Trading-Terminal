import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_pc(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Price Channels(PC) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Price Channels
    rolling_high = dataset['High'].rolling(window=20).max()
    rolling_low = dataset['Low'].rolling(window=20).min()
    midline = (rolling_high + rolling_low) / 2

    # Plot Price Channels
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=rolling_high, mode='lines', name='Upper Channel'),
                        go.Scatter(x=dataset.index, y=rolling_low, mode='lines', name='Lower Channel'),
                        go.Scatter(x=dataset.index, y=midline, mode='lines', name='Midline')])
    fig.update_layout(title=f"{symbol} Price Channels",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig)

def norm_pc(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Price Channels
    rolling_high = dataset['High'].rolling(window=20).max()
    rolling_low = dataset['Low'].rolling(window=20).min()
    midline = (rolling_high + rolling_low) / 2

    # Plot Price Channels
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=rolling_high, mode='lines', name='Upper Channel'),
                        go.Scatter(x=dataset.index, y=rolling_low, mode='lines', name='Lower Channel'),
                        go.Scatter(x=dataset.index, y=midline, mode='lines', name='Midline')])
    fig.update_layout(title=f"{symbol} Price Channels",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig)