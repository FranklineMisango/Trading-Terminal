import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_kc(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Keltner Channels(KC) for a selected ticker '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Keltner Channels
    n = 20
    df["EMA"] = ta.EMA(df["Adj Close"], timeperiod=n)
    df["ATR"] = ta.ATR(df["High"], df["Low"], df["Adj Close"], timeperiod=10)
    df["Upper Line"] = df["EMA"] + 2 * df["ATR"]
    df["Lower Line"] = df["EMA"] - 2 * df["ATR"]

    # Plot Keltner Channels
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode='lines', name='EMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper Line'], mode='lines', name='Upper Line'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower Line'], mode='lines', name='Lower Line'))
    fig.update_layout(title=f"Keltner Channels for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_kc(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Keltner Channels
    n = 20
    df["EMA"] = ta.EMA(df["Adj Close"], timeperiod=n)
    df["ATR"] = ta.ATR(df["High"], df["Low"], df["Adj Close"], timeperiod=10)
    df["Upper Line"] = df["EMA"] + 2 * df["ATR"]
    df["Lower Line"] = df["EMA"] - 2 * df["ATR"]

    # Plot Keltner Channels
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode='lines', name='EMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper Line'], mode='lines', name='Upper Line'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower Line'], mode='lines', name='Lower Line'))
    fig.update_layout(title=f"Keltner Channels for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)