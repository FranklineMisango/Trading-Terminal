import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_hml(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to view the High Minus Low(HML) of a ticker over time. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Compute High Minus Low
    df["H-L"] = df["High"] - df["Low"]

    # Plot High Minus Low with Candlestick graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['H-L'], mode='lines', name='High Minus Low'))
    fig.update_layout(title=f"High Minus Low for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_hml(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Compute High Minus Low
    df["H-L"] = df["High"] - df["Low"]

    # Plot High Minus Low with Candlestick graph
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['H-L'], mode='lines', name='High Minus Low'))
    fig.update_layout(title=f"High Minus Low for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

