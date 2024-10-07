import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool
import ta_functions as ta


@tool

def tool_dema(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Double Exponential Moving Average (DEMA) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["EMA"] = ta.EMA(df["Adj Close"], timeperiod=5)
    df["EMA_S"] = ta.EMA(df["EMA"], timeperiod=5)
    df["DEMA"] = (2 * df["EMA"]) - df["EMA_S"]

    # Plot DEMA
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'],
                                        name='Candlestick'),
                        go.Scatter(x=df.index,
                                    y=df['DEMA'],
                                    mode='lines',
                                    name='DEMA')])
    fig.update_layout(title=f"Double Exponential Moving Average (DEMA) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_dema(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["EMA"] = ta.EMA(df["Adj Close"], timeperiod=5)
    df["EMA_S"] = ta.EMA(df["EMA"], timeperiod=5)
    df["DEMA"] = (2 * df["EMA"]) - df["EMA_S"]

    # Plot DEMA
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'],
                                        name='Candlestick'),
                        go.Scatter(x=df.index,
                                    y=df['DEMA'],
                                    mode='lines',
                                    name='DEMA')])
    fig.update_layout(title=f"Double Exponential Moving Average (DEMA) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)