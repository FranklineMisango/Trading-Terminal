import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_apo( ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots the Absolute Price Oscillator (APO) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    df["HL"] = (df["High"] + df["Low"]) / 2
    df["HLC"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3
    df["HLCC"] = (df["High"] + df["Low"] + df["Adj Close"] + df["Adj Close"]) / 4
    df["OHLC"] = (df["Open"] + df["High"] + df["Low"] + df["Adj Close"]) / 4

    df["Long_Cycle"] = df["Adj Close"].rolling(20).mean()
    df["Short_Cycle"] = df["Adj Close"].rolling(5).mean()
    df["APO"] = df["Long_Cycle"] - df["Short_Cycle"]

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    # Absolute Price Oscillator (APO) Chart
    fig.add_trace(go.Scatter(x=df.index, y=df["APO"], mode='lines', name="Absolute Price Oscillator", line=dict(color='green')))
    fig.update_layout(title="Absolute Price Oscillator (APO) for " + symbol,
                    xaxis_title="Date",
                    yaxis_title="APO",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick with Absolute Price Oscillator (APO)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index, y=df["APO"], mode='lines', name="Absolute Price Oscillator", line=dict(color='green')))
    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with APO",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

def norm_apo(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    df["HL"] = (df["High"] + df["Low"]) / 2
    df["HLC"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3
    df["HLCC"] = (df["High"] + df["Low"] + df["Adj Close"] + df["Adj Close"]) / 4
    df["OHLC"] = (df["Open"] + df["High"] + df["Low"] + df["Adj Close"]) / 4

    df["Long_Cycle"] = df["Adj Close"].rolling(20).mean()
    df["Short_Cycle"] = df["Adj Close"].rolling(5).mean()
    df["APO"] = df["Long_Cycle"] - df["Short_Cycle"]

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    # Absolute Price Oscillator (APO) Chart
    fig.add_trace(go.Scatter(x=df.index, y=df["APO"], mode='lines', name="Absolute Price Oscillator", line=dict(color='green')))
    fig.update_layout(title="Absolute Price Oscillator (APO) for " + symbol,
                    xaxis_title="Date",
                    yaxis_title="APO",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick with Absolute Price Oscillator (APO)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index, y=df["APO"], mode='lines', name="Absolute Price Oscillator", line=dict(color='green')))
    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with APO",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)