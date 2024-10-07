import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_mar(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Moving Average Ribbon(MAR) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["MA10"] = df["Adj Close"].rolling(10).mean()
    df["MA20"] = df["Adj Close"].rolling(20).mean()
    df["MA30"] = df["Adj Close"].rolling(30).mean()
    df["MA40"] = df["Adj Close"].rolling(40).mean()
    df["MA50"] = df["Adj Close"].rolling(50).mean()
    df["MA60"] = df["Adj Close"].rolling(60).mean()

    # Plot Line Chart with Moving Average Ribbon
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA10"], mode='lines', name='MA10'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode='lines', name='MA20'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA30"], mode='lines', name='MA30'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA40"], mode='lines', name='MA40'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode='lines', name='MA50'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA60"], mode='lines', name='MA60'))
    fig_line.update_layout(title=f"Stock {symbol} Moving Average Ribbon",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)

def norm_mar(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["MA10"] = df["Adj Close"].rolling(10).mean()
    df["MA20"] = df["Adj Close"].rolling(20).mean()
    df["MA30"] = df["Adj Close"].rolling(30).mean()
    df["MA40"] = df["Adj Close"].rolling(40).mean()
    df["MA50"] = df["Adj Close"].rolling(50).mean()
    df["MA60"] = df["Adj Close"].rolling(60).mean()

    # Plot Line Chart with Moving Average Ribbon
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA10"], mode='lines', name='MA10'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode='lines', name='MA20'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA30"], mode='lines', name='MA30'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA40"], mode='lines', name='MA40'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode='lines', name='MA50'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["MA60"], mode='lines', name='MA60'))
    fig_line.update_layout(title=f"Stock {symbol} Moving Average Ribbon",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)