import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_car(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Candle Absolute Returns (CAR) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)
    df["Absolute_Return"] = (
        100 * (df["Adj Close"] - df["Adj Close"].shift(1)) / df["Adj Close"].shift(1)
    )

    # Plot the closing price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name="Closing Price"))
    fig.update_layout(title="Stock " + symbol + " Closing Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Plot the Absolute Return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Absolute_Return"], mode="lines", name="Absolute Return", line=dict(color="red")))
    fig.update_layout(title="Absolute Return", xaxis_title="Date", yaxis_title="Absolute Return")
    
    st.plotly_chart(fig)  

def norm_car(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)
    df["Absolute_Return"] = (
        100 * (df["Adj Close"] - df["Adj Close"].shift(1)) / df["Adj Close"].shift(1)
    )

    # Plot the closing price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name="Closing Price"))
    fig.update_layout(title="Stock " + symbol + " Closing Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

    # Plot the Absolute Return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Absolute_Return"], mode="lines", name="Absolute Return", line=dict(color="red")))
    fig.update_layout(title="Absolute Return", xaxis_title="Date", yaxis_title="Absolute Return")
    
    st.plotly_chart(fig)
