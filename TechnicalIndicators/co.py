import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_co(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Chaikin Oscillator (CO) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Chaikin Oscillator
    df["MF_Multiplier"] = (2 * df["Adj Close"] - df["Low"] - df["High"]) / (df["High"] - df["Low"])
    df["MF_Volume"] = df["MF_Multiplier"] * df["Volume"]
    df["ADL"] = df["MF_Volume"].cumsum()
    df["ADL_3_EMA"] = df["ADL"].ewm(ignore_na=False, span=3, min_periods=2, adjust=True).mean()
    df["ADL_10_EMA"] = df["ADL"].ewm(ignore_na=False, span=10, min_periods=9, adjust=True).mean()
    df["Chaikin_Oscillator"] = df["ADL_3_EMA"] - df["ADL_10_EMA"]
    df = df.drop(["MF_Multiplier", "MF_Volume", "ADL", "ADL_3_EMA", "ADL_10_EMA"], axis=1)

    # Plot Chaikin Oscillator
    co_chart = go.Scatter(x=df.index, y=df["Chaikin_Oscillator"], mode='lines', name='Chaikin Oscillator')

    layout = go.Layout(title=f'Chaikin Oscillator for Stock {symbol}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Chaikin Oscillator'),
                    showlegend=True)

    fig = go.Figure(data=[co_chart], layout=layout)
    st.plotly_chart(fig)  

def norm_co(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Chaikin Oscillator
    df["MF_Multiplier"] = (2 * df["Adj Close"] - df["Low"] - df["High"]) / (df["High"] - df["Low"])
    df["MF_Volume"] = df["MF_Multiplier"] * df["Volume"]
    df["ADL"] = df["MF_Volume"].cumsum()
    df["ADL_3_EMA"] = df["ADL"].ewm(ignore_na=False, span=3, min_periods=2, adjust=True).mean()
    df["ADL_10_EMA"] = df["ADL"].ewm(ignore_na=False, span=10, min_periods=9, adjust=True).mean()
    df["Chaikin_Oscillator"] = df["ADL_3_EMA"] - df["ADL_10_EMA"]
    df = df.drop(["MF_Multiplier", "MF_Volume", "ADL", "ADL_3_EMA", "ADL_10_EMA"], axis=1)

    # Plot Chaikin Oscillator
    co_chart = go.Scatter(x=df.index, y=df["Chaikin_Oscillator"], mode='lines', name='Chaikin Oscillator')

    layout = go.Layout(title=f'Chaikin Oscillator for Stock {symbol}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Chaikin Oscillator'),
                    showlegend=True)

    fig = go.Figure(data=[co_chart], layout=layout)
    st.plotly_chart(fig)  