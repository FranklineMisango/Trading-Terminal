import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_dpo(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Detrended Price Oscillator (DPO) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 15
    df["DPO"] = (df["Adj Close"].shift(int((0.5 * n) + 1)) - df["Adj Close"].rolling(n).mean())

    # Plot DPO
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'],
                                        name='Candlestick'),
                        go.Scatter(x=df.index,
                                    y=df['DPO'],
                                    mode='lines',
                                    name='DPO')])
    fig.update_layout(title=f"Detrended Price Oscillator (DPO) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
st.plotly_chart(fig) 

def norm_dpo(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 15
    df["DPO"] = (df["Adj Close"].shift(int((0.5 * n) + 1)) - df["Adj Close"].rolling(n).mean())

    # Plot DPO
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'],
                                        name='Candlestick'),
                        go.Scatter(x=df.index,
                                    y=df['DPO'],
                                    mode='lines',
                                    name='DPO')])
    fig.update_layout(title=f"Detrended Price Oscillator (DPO) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
st.plotly_chart(fig) 