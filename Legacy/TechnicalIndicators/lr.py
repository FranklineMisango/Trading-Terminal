import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_lr(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Linear Regression(LR) for a selected ticker '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Linear Regression
    avg = df["Adj Close"].mean()
    df["Linear_Regression"] = avg - (df["Adj Close"].mean() - df["Adj Close"]) / 20

    # Plot Linear Regression
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Regression'], mode='lines', name='Linear Regression'))
    fig.update_layout(title=f"Linear Regression for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_lr(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Linear Regression
    avg = df["Adj Close"].mean()
    df["Linear_Regression"] = avg - (df["Adj Close"].mean() - df["Adj Close"]) / 20

    # Plot Linear Regression
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Regression'], mode='lines', name='Linear Regression'))
    fig.update_layout(title=f"Linear Regression for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)