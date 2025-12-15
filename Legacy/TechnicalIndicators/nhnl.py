import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_nhnl(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize New Highs/New Lows(NHNL) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df['52wHigh'] = df['Adj Close'].rolling(window=252).max()
    df['52wLow'] = df['Adj Close'].rolling(window=252).min()

    # Plot Line Chart with New Highs/New Lows
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["52wHigh"], mode='lines', name='52 Weeks High', line=dict(color='green')))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["52wLow"], mode='lines', name='52 Weeks Low', line=dict(color='red')))
    fig_line.update_layout(title=f"Stock {symbol} New Highs/New Lows",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)

def norm_nhnl(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df['52wHigh'] = df['Adj Close'].rolling(window=252).max()
    df['52wLow'] = df['Adj Close'].rolling(window=252).min()

    # Plot Line Chart with New Highs/New Lows
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["52wHigh"], mode='lines', name='52 Weeks High', line=dict(color='green')))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["52wLow"], mode='lines', name='52 Weeks Low', line=dict(color='red')))
    fig_line.update_layout(title=f"Stock {symbol} New Highs/New Lows",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)

    