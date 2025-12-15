import streamlit as st
import datetime as dt
import yfinance as yf
from langchain_core.tools import tool
from plotly import graph_objs as go
import pandas as pd
import plotly.express as px

@tool
def tool_ma_high_low(start_date: dt.time, end_date: dt.time, ticker: str):
    ''' This tool plots the candlestick chart of a stock along with the Moving Average of High and Low for the stock.'''
    symbol = ticker
    start = start_date
    end = end_date
    
    # Read data
    df = yf.download(symbol, start, end)

    df["MA_High"] = df["High"].rolling(10).mean()
    df["MA_Low"] = df["Low"].rolling(10).mean()
    df = df.dropna()

    # Moving Average Line Chart
    fig1 = px.line(df, x=df.index, y=["Adj Close", "MA_High", "MA_Low"], title="Moving Average of High and Low for Stock")
    fig1.update_xaxes(title_text="Date")
    fig1.update_yaxes(title_text="Price")
    st.plotly_chart(fig1)

    # Candlestick with Moving Averages High and Low
    fig2 = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'])])

    fig2.add_trace(go.Scatter(x=df.index, y=df['MA_High'], mode='lines', name='MA High'))
    fig2.add_trace(go.Scatter(x=df.index, y=df['MA_Low'], mode='lines', name='MA Low'))

    fig2.update_layout(title="Candlestick with Moving Averages High and Low",
                    xaxis_title="Date",
                    yaxis_title="Price")

    st.plotly_chart(fig2)

def norm_ma_high_low(start_date: dt.time, end_date: dt.time, ticker: str):
    symbol = ticker
    start = start_date
    end = end_date
    
    # Read data
    df = yf.download(symbol, start, end)

    df["MA_High"] = df["High"].rolling(10).mean()
    df["MA_Low"] = df["Low"].rolling(10).mean()
    df = df.dropna()

    # Moving Average Line Chart
    fig1 = px.line(df, x=df.index, y=["Adj Close", "MA_High", "MA_Low"], title="Moving Average of High and Low for Stock")
    fig1.update_xaxes(title_text="Date")
    fig1.update_yaxes(title_text="Price")
    st.plotly_chart(fig1)

    # Candlestick with Moving Averages High and Low
    fig2 = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Adj Close'])])

    fig2.add_trace(go.Scatter(x=df.index, y=df['MA_High'], mode='lines', name='MA High'))
    fig2.add_trace(go.Scatter(x=df.index, y=df['MA_Low'], mode='lines', name='MA Low'))

    fig2.update_layout(title="Candlestick with Moving Averages High and Low",
                    xaxis_title="Date",
                    yaxis_title="Price")

    st.plotly_chart(fig2)