import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_cmf(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Chaikin Money Flow (CMF) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate CMF
    n = 20
    df["MF_Multiplier"] = (2 * df["Adj Close"] - df["Low"] - df["High"]) / (df["High"] - df["Low"])
    df["MF_Volume"] = df["MF_Multiplier"] * df["Volume"]
    df["CMF"] = df["MF_Volume"].rolling(n).sum() / df["Volume"].rolling(n).sum()
    df = df.drop(["MF_Multiplier", "MF_Volume"], axis=1)

    # Plot CMF
    cmf_chart = go.Scatter(x=df.index, y=df["CMF"], mode='lines', name='Chaikin Money Flow')

    layout = go.Layout(title=f'Chaikin Money Flow for Stock {symbol}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='CMF'),
                    showlegend=True)

    fig = go.Figure(data=[cmf_chart], layout=layout)
    st.plotly_chart(fig)  

def norm_cmf(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate CMF
    n = 20
    df["MF_Multiplier"] = (2 * df["Adj Close"] - df["Low"] - df["High"]) / (df["High"] - df["Low"])
    df["MF_Volume"] = df["MF_Multiplier"] * df["Volume"]
    df["CMF"] = df["MF_Volume"].rolling(n).sum() / df["Volume"].rolling(n).sum()
    df = df.drop(["MF_Multiplier", "MF_Volume"], axis=1)

    # Plot CMF
    cmf_chart = go.Scatter(x=df.index, y=df["CMF"], mode='lines', name='Chaikin Money Flow')

    layout = go.Layout(title=f'Chaikin Money Flow for Stock {symbol}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='CMF'),
                    showlegend=True)

    fig = go.Figure(data=[cmf_chart], layout=layout)
    st.plotly_chart(fig) 