import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_dmi(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Dynamic Momentum Index (DMI) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["sd"] = df["Adj Close"].rolling(5).std()
    df["asd"] = df["sd"].rolling(10).mean()
    df["DMI"] = 14 / (df["sd"] / df["asd"])
    df = df.drop(["sd", "asd"], axis=1)

    # Plot DMI
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'],
                                        name='Candlestick'),
                        go.Scatter(x=df.index,
                                    y=df['DMI'],
                                    mode='lines',
                                    name='Dynamic Momentum Index')])
    fig.update_layout(title=f"Dynamic Momentum Index (DMI) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)  

def norm_dmi(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["sd"] = df["Adj Close"].rolling(5).std()
    df["asd"] = df["sd"].rolling(10).mean()
    df["DMI"] = 14 / (df["sd"] / df["asd"])
    df = df.drop(["sd", "asd"], axis=1)

    # Plot DMI
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'],
                                        name='Candlestick'),
                        go.Scatter(x=df.index,
                                    y=df['DMI'],
                                    mode='lines',
                                    name='Dynamic Momentum Index')])
    fig.update_layout(title=f"Dynamic Momentum Index (DMI) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)  