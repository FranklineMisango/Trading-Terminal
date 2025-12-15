import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_gri(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to view the Geometric Return Indicator(GRI) of a ticker over time '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Compute Geometric Return Indicator
    n = 10
    df["Geometric_Return"] = pd.Series(df["Adj Close"]).rolling(n).apply(gmean)

    # Plot Geometric Return Indicator with Candlestick graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Geometric_Return'], mode='lines', name='Geometric Return Indicator'))
    fig.update_layout(title=f"Geometric Return Indicator for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_gri(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Compute Geometric Return Indicator
    n = 10
    df["Geometric_Return"] = pd.Series(df["Adj Close"]).rolling(n).apply(gmean)

    # Plot Geometric Return Indicator with Candlestick graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Geometric_Return'], mode='lines', name='Geometric Return Indicator'))
    fig.update_layout(title=f"Geometric Return Indicator for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)