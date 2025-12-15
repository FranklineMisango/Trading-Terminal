import streamlit as st
import datetime as dt
import yfinance as yf
from langchain_core.tools import tool
from plotly import graph_objs as go
import pandas as pd
import plotly.express as px
import matplotlib.dates as mdates
import numpy as np
import ta_functions as ta


@tool
def tool_rsi_bollinger_bands(ticker:str, start_date:dt.time, end_date:dt.time):
    '''Tool for Relative Strength Index (RSI) and Bollinger Bands'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Simple Line Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.update_layout(title=symbol + " Closing Price", xaxis_title="Date", yaxis_title="Price")

    st.plotly_chart(fig1)

    # RSI
    rsi = ta.RSI(df["Adj Close"], timeperiod=14)
    rsi = rsi.dropna()

    # Bollinger Bands
    df["20 Day MA"] = df["Adj Close"].rolling(window=20).mean()
    df["20 Day STD"] = df["Adj Close"].rolling(window=20).std()
    df["Upper Band"] = df["20 Day MA"] + (df["20 Day STD"] * 2)
    df["Lower Band"] = df["20 Day MA"] - (df["20 Day STD"] * 2)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig2.add_trace(go.Scatter(x=df.index, y=df["20 Day MA"], mode='lines', name='20 Day MA'))
    fig2.add_trace(go.Scatter(x=df.index, y=df["Upper Band"], mode='lines', name='Upper Band'))
    fig2.add_trace(go.Scatter(x=df.index, y=df["Lower Band"], mode='lines', name='Lower Band'))
    fig2.update_layout(title=f"30 Day Bollinger Band for {symbol.upper()}", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2)

    dfc = df.reset_index()
    dfc["Date"] = pd.to_datetime(dfc["Date"])

    # Candlestick with Bollinger Bands
    fig3 = go.Figure()
    fig3.add_trace(go.Candlestick(x=dfc['Date'],
                    open=dfc['Open'],
                    high=dfc['High'],
                    low=dfc['Low'],
                    close=dfc['Adj Close'], name='Candlestick'))

    fig3.add_trace(go.Scatter(x=dfc['Date'], y=df["20 Day MA"], mode='lines', name='20 Day MA'))
    fig3.add_trace(go.Scatter(x=dfc['Date'], y=df["Upper Band"], mode='lines', name='Upper Band'))
    fig3.add_trace(go.Scatter(x=dfc['Date'], y=df["Lower Band"], mode='lines', name='Lower Band'))

    fig3.update_layout(title=f"{symbol.upper()} Candlestick Chart with Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig3)

    # Combine RSI and Bollinger Bands
    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(x=df.index, y=df["20 Day MA"], mode='lines', name='20 Day MA'))
    fig4.add_trace(go.Scatter(x=df.index, y=df["Upper Band"], mode='lines', name='Upper Band'))
    fig4.add_trace(go.Scatter(x=df.index, y=df["Lower Band"], mode='lines', name='Lower Band'))

    fig4.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI'))

    fig4.update_layout(title=f"Bollinger Bands & RSI for {symbol.upper()}", xaxis_title="Date", yaxis_title="Price/RSI")
    st.plotly_chart(fig4)

def norm_rsi_bollinger_bands(ticker:str, start_date:dt.time, end_date:dt.time):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Simple Line Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig1.update_layout(title=symbol + " Closing Price", xaxis_title="Date", yaxis_title="Price")

    st.plotly_chart(fig1)

    # RSI
    rsi = ta.RSI(df["Adj Close"], timeperiod=14)
    rsi = rsi.dropna()

    # Bollinger Bands
    df["20 Day MA"] = df["Adj Close"].rolling(window=20).mean()
    df["20 Day STD"] = df["Adj Close"].rolling(window=20).std()
    df["Upper Band"] = df["20 Day MA"] + (df["20 Day STD"] * 2)
    df["Lower Band"] = df["20 Day MA"] - (df["20 Day STD"] * 2)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig2.add_trace(go.Scatter(x=df.index, y=df["20 Day MA"], mode='lines', name='20 Day MA'))
    fig2.add_trace(go.Scatter(x=df.index, y=df["Upper Band"], mode='lines', name='Upper Band'))
    fig2.add_trace(go.Scatter(x=df.index, y=df["Lower Band"], mode='lines', name='Lower Band'))
    fig2.update_layout(title=f"30 Day Bollinger Band for {symbol.upper()}", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2)

    dfc = df.reset_index()
    dfc["Date"] = pd.to_datetime(dfc["Date"])

    # Candlestick with Bollinger Bands
    fig3 = go.Figure()
    fig3.add_trace(go.Candlestick(x=dfc['Date'],
                    open=dfc['Open'],
                    high=dfc['High'],
                    low=dfc['Low'],
                    close=dfc['Adj Close'], name='Candlestick'))

    fig3.add_trace(go.Scatter(x=dfc['Date'], y=df["20 Day MA"], mode='lines', name='20 Day MA'))
    fig3.add_trace(go.Scatter(x=dfc['Date'], y=df["Upper Band"], mode='lines', name='Upper Band'))
    fig3.add_trace(go.Scatter(x=dfc['Date'], y=df["Lower Band"], mode='lines', name='Lower Band'))

    fig3.update_layout(title=f"{symbol.upper()} Candlestick Chart with Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig3)

    # Combine RSI and Bollinger Bands
    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(x=df.index, y=df["20 Day MA"], mode='lines', name='20 Day MA'))
    fig4.add_trace(go.Scatter(x=df.index, y=df["Upper Band"], mode='lines', name='Upper Band'))
    fig4.add_trace(go.Scatter(x=df.index, y=df["Lower Band"], mode='lines', name='Lower Band'))

    fig4.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI'))

    fig4.update_layout(title=f"Bollinger Bands & RSI for {symbol.upper()}", xaxis_title="Date", yaxis_title="Price/RSI")
    st.plotly_chart(fig4)