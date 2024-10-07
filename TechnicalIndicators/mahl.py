import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_mahl(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Moving Average High/Low(MAHL) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 14  # number of periods
    df["MA_High"] = df["High"].rolling(n).mean()
    df["MA_Low"] = df["Low"].rolling(n).mean()

    # Plot Line Chart with Moving Average High/Low
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA_High"], mode='lines', name='Moving Average of High'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA_Low"], mode='lines', name='Moving Average of Low'))
    fig_line.update_layout(title=f"Stock {symbol} Moving Average High/Low",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)

def norm_mahl(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 14  # number of periods
    df["MA_High"] = df["High"].rolling(n).mean()
    df["MA_Low"] = df["Low"].rolling(n).mean()

    # Plot Line Chart with Moving Average High/Low
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA_High"], mode='lines', name='Moving Average of High'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA_Low"], mode='lines', name='Moving Average of Low'))
    fig_line.update_layout(title=f"Stock {symbol} Moving Average High/Low",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)
