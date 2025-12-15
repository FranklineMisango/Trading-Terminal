import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_pr(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Price Relative for a selected ticker. '''
    symbol = ticker
    benchmark = benchmark_ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)
    benchmark_data = yf.download(benchmark, start, end)

    # Calculate Price Relative
    price_relative = dataset['Adj Close'] / benchmark_data['Adj Close']

    # Plot Price Relative
    fig = go.Figure(data=[go.Scatter(x=price_relative.index, y=price_relative, mode='lines', name='Price Relative')])
    fig.update_layout(title=f"{symbol} Price Relative to {benchmark}",
                    xaxis_title='Date',
                    yaxis_title='Price Relative')
    
    st.plotly_chart(fig)

def norm_pr(ticker, start_date, end_date):
    symbol = ticker
    benchmark = benchmark_ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)
    benchmark_data = yf.download(benchmark, start, end)

    # Calculate Price Relative
    price_relative = dataset['Adj Close'] / benchmark_data['Adj Close']

    # Plot Price Relative
    fig = go.Figure(data=[go.Scatter(x=price_relative.index, y=price_relative, mode='lines', name='Price Relative')])
    fig.update_layout(title=f"{symbol} Price Relative to {benchmark}",
                    xaxis_title='Date',
                    yaxis_title='Price Relative')
    
    st.plotly_chart(fig)
    