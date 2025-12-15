import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_cci(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Commodity Channel Index (CCI) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate CCI
    n = 20
    df["TP"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3
    df["SMA_TP"] = df["TP"].rolling(n).mean()
    df["SMA_STD"] = df["TP"].rolling(n).std()
    df["CCI"] = (df["TP"] - df["SMA_TP"]) / (0.015 * df["SMA_STD"])
    df = df.drop(["TP", "SMA_TP", "SMA_STD"], axis=1)

    # Plot CCI
    cci_chart = go.Scatter(x=df.index, y=df["CCI"], mode='lines', name='Commodity Channel Index (CCI)')

    layout = go.Layout(title=f'Commodity Channel Index (CCI) for Stock {symbol}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='CCI'),
                    showlegend=True)

    fig = go.Figure(data=[cci_chart], layout=layout)
    st.plotly_chart(fig)  

def norm_cci(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate CCI
    n = 20
    df["TP"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3
    df["SMA_TP"] = df["TP"].rolling(n).mean()
    df["SMA_STD"] = df["TP"].rolling(n).std()
    df["CCI"] = (df["TP"] - df["SMA_TP"]) / (0.015 * df["SMA_STD"])
    df = df.drop(["TP", "SMA_TP", "SMA_STD"], axis=1)

    # Plot CCI
    cci_chart = go.Scatter(x=df.index, y=df["CCI"], mode='lines', name='Commodity Channel Index (CCI)')

    layout = go.Layout(title=f'Commodity Channel Index (CCI) for Stock {symbol}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='CCI'),
                    showlegend=True)

    fig = go.Figure(data=[cci_chart], layout=layout)
    st.plotly_chart(fig)  