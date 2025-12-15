import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_rvi(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Relative Volatility Index(RVI) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Relative Volatility Index (RVI)
    n = 14  # Number of period
    change = dataset["Adj Close"].diff(1)
    gain = change.mask(change < 0, 0)
    loss = abs(change.mask(change > 0, 0))
    avg_gain = gain.rolling(n).std()
    avg_loss = loss.rolling(n).std()
    RS = avg_gain / avg_loss
    RVI = 100 - (100 / (1 + RS))

    # Plot RVI
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=RVI.index, y=RVI, mode='lines', name='Relative Volatility Index', line=dict(color='blue')))
    fig.add_shape(type="line", x0=RVI.index[0], y0=60, x1=RVI.index[-1], y1=60, line=dict(color="red", width=1, dash="dash"), name="Overbought")
    fig.add_shape(type="line", x0=RVI.index[0], y0=40, x1=RVI.index[-1], y1=40, line=dict(color="green", width=1, dash="dash"), name="Oversold")
    fig.update_layout(title=f"{symbol} Relative Volatility Index",
                    xaxis_title="Date",
                    yaxis_title="RVI")
    st.plotly_chart(fig)

def norm_rvi(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Relative Volatility Index (RVI)
    n = 14  # Number of period
    change = dataset["Adj Close"].diff(1)
    gain = change.mask(change < 0, 0)
    loss = abs(change.mask(change > 0, 0))
    avg_gain = gain.rolling(n).std()
    avg_loss = loss.rolling(n).std()
    RS = avg_gain / avg_loss
    RVI = 100 - (100 / (1 + RS))

    # Plot RVI
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=RVI.index, y=RVI, mode='lines', name='Relative Volatility Index', line=dict(color='blue')))
    fig.add_shape(type="line", x0=RVI.index[0], y0=60, x1=RVI.index[-1], y1=60, line=dict(color="red", width=1, dash="dash"), name="Overbought")
    fig.add_shape(type="line", x0=RVI.index[0], y0=40, x1=RVI.index[-1], y1=40, line=dict(color="green", width=1, dash="dash"), name="Oversold")
    fig.update_layout(title=f"{symbol} Relative Volatility Index",
                    xaxis_title="Date",
                    yaxis_title="RVI")
    st.plotly_chart(fig)

