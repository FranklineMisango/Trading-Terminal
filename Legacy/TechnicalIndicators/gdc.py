import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_gdc(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to view the Golden/Death Cross(GDC) of a ticker over time '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Compute Golden/Death Cross
    df["MA_50"] = df["Adj Close"].rolling(center=False, window=50).mean()
    df["MA_200"] = df["Adj Close"].rolling(center=False, window=200).mean()
    df["diff"] = df["MA_50"] - df["MA_200"]

    # Plot Golden/Death Cross with Candlestick graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], mode='lines', name='MA_50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], mode='lines', name='MA_200'))
    fig.update_layout(title=f"Golden/Death Cross for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_gdc(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Compute Golden/Death Cross
    df["MA_50"] = df["Adj Close"].rolling(center=False, window=50).mean()
    df["MA_200"] = df["Adj Close"].rolling(center=False, window=200).mean()
    df["diff"] = df["MA_50"] - df["MA_200"]

    # Plot Golden/Death Cross with Candlestick graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], mode='lines', name='MA_50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], mode='lines', name='MA_200'))
    fig.update_layout(title=f"Golden/Death Cross for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)