import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_cpr(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Central Pivot Range (CPR) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate CPR
    df["Pivot"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3.0
    df["BC"] = (df["High"] + df["Low"]) / 2.0
    df["TC"] = (df["Pivot"] - df["BC"]) + df["Pivot"]

    # Plot candlestick with CPR
    candlestick = go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Candlestick')

    pivot = go.Scatter(x=df.index, y=df["Pivot"], mode='lines', name='Pivot')
    bc = go.Scatter(x=df.index, y=df["BC"], mode='lines', name='BC')
    tc = go.Scatter(x=df.index, y=df["TC"], mode='lines', name='TC')

    data = [candlestick, pivot, bc, tc]

    layout = go.Layout(title=f'Stock {symbol} Closing Price with Central Pivot Range (CPR)',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Price'),
                    showlegend=True)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)

def norm_cpr(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate CPR
    df["Pivot"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3.0
    df["BC"] = (df["High"] + df["Low"]) / 2.0
    df["TC"] = (df["Pivot"] - df["BC"]) + df["Pivot"]

    # Plot candlestick with CPR
    candlestick = go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Candlestick')

    pivot = go.Scatter(x=df.index, y=df["Pivot"], mode='lines', name='Pivot')
    bc = go.Scatter(x=df.index, y=df["BC"], mode='lines', name='BC')
    tc = go.Scatter(x=df.index, y=df["TC"], mode='lines', name='TC')

    data = [candlestick, pivot, bc, tc]

    layout = go.Layout(title=f'Stock {symbol} Closing Price with Central Pivot Range (CPR)',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Price'),
                    showlegend=True)

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)
    