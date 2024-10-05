import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_fi(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to view the Force Index of a ticker over time. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 13
    df["FI_1"] = (df["Adj Close"] - df["Adj Close"].shift()) * df["Volume"]
    df["FI_13"] = df["FI_1"].ewm(ignore_na=False, span=n, min_periods=n, adjust=True).mean()

    # Plot Force Index
    fig = go.Figure(data=[
        go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Adj Close'],
                    name='Candlestick'),
        go.Scatter(x=df.index,
                y=df['FI_13'],
                mode='lines',
                name='Force Index')
    ])
    fig.update_layout(title=f"Force Index for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_fi(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 13
    df["FI_1"] = (df["Adj Close"] - df["Adj Close"].shift()) * df["Volume"]
    df["FI_13"] = df["FI_1"].ewm(ignore_na=False, span=n, min_periods=n, adjust=True).mean()

    # Plot Force Index
    fig = go.Figure(data=[
        go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Adj Close'],
                    name='Candlestick'),
        go.Scatter(x=df.index,
                y=df['FI_13'],
                mode='lines',
                name='Force Index')
    ])
    fig.update_layout(title=f"Force Index for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)
