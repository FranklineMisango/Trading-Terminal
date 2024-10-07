import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool
import pandas as pd

@tool
def tool_dc(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Donchian Channel (DC) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["Upper_Channel_Line"] = pd.Series.rolling(df["High"], window=20).max()
    df["Lower_Channel_Line"] = pd.Series.rolling(df["Low"], window=20).min()
    df["Middle_Channel_Line"] = (df["Upper_Channel_Line"] + df["Lower_Channel_Line"]) / 2
    df = df.dropna()

    # Plot Donchian Channel
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'],
                                        name='Candlestick'),
                        go.Scatter(x=df.index,
                                    y=df['Upper_Channel_Line'],
                                    mode='lines',
                                    name='Upper Channel Line'),
                        go.Scatter(x=df.index,
                                    y=df['Lower_Channel_Line'],
                                    mode='lines',
                                    name='Lower Channel Line'),
                        go.Scatter(x=df.index,
                                    y=df['Middle_Channel_Line'],
                                    mode='lines',
                                    name='Middle Channel Line')])
    fig.update_layout(title=f"Donchian Channel for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_dc(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["Upper_Channel_Line"] = pd.Series.rolling(df["High"], window=20).max()
    df["Lower_Channel_Line"] = pd.Series.rolling(df["Low"], window=20).min()
    df["Middle_Channel_Line"] = (df["Upper_Channel_Line"] + df["Lower_Channel_Line"]) / 2
    df = df.dropna()

    # Plot Donchian Channel
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'],
                                        name='Candlestick'),
                        go.Scatter(x=df.index,
                                    y=df['Upper_Channel_Line'],
                                    mode='lines',
                                    name='Upper Channel Line'),
                        go.Scatter(x=df.index,
                                    y=df['Lower_Channel_Line'],
                                    mode='lines',
                                    name='Lower Channel Line'),
                        go.Scatter(x=df.index,
                                    y=df['Middle_Channel_Line'],
                                    mode='lines',
                                    name='Middle Channel Line')])
    fig.update_layout(title=f"Donchian Channel for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

