import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_adl(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots the Accumulation Distribution Line (ADL) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)
    
    df["MF Multiplier"] = (2 * df["Adj Close"] - df["Low"] - df["High"]) / (
        df["High"] - df["Low"]
    )
    df["MF Volume"] = df["MF Multiplier"] * df["Volume"]
    df["ADL"] = df["MF Volume"].cumsum()
    df = df.drop(["MF Multiplier", "MF Volume"], axis=1)

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["ADL"], mode='lines', name='Accumulation Distribution Line'))
    fig.update_layout(yaxis2=dict(title="Accumulation Distribution Line"))

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color='rgba(0, 0, 255, 0.4)'))
    fig.update_layout(yaxis3=dict(title="Volume"))

    st.plotly_chart(fig)

    # Candlestick Chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["ADL"], mode='lines', name='Accumulation Distribution Line'))
    fig.update_layout(yaxis2=dict(title="Accumulation Distribution Line"))

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color='rgba(0, 0, 255, 0.4)'))
    fig.update_layout(yaxis3=dict(title="Volume"))

    st.plotly_chart(fig)


def norm_adl(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)
    
    df["MF Multiplier"] = (2 * df["Adj Close"] - df["Low"] - df["High"]) / (
        df["High"] - df["Low"]
    )
    df["MF Volume"] = df["MF Multiplier"] * df["Volume"]
    df["ADL"] = df["MF Volume"].cumsum()
    df = df.drop(["MF Multiplier", "MF Volume"], axis=1)

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["ADL"], mode='lines', name='Accumulation Distribution Line'))
    fig.update_layout(yaxis2=dict(title="Accumulation Distribution Line"))

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color='rgba(0, 0, 255, 0.4)'))
    fig.update_layout(yaxis3=dict(title="Volume"))

    st.plotly_chart(fig)

    # Candlestick Chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["ADL"], mode='lines', name='Accumulation Distribution Line'))
    fig.update_layout(yaxis2=dict(title="Accumulation Distribution Line"))

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color='rgba(0, 0, 255, 0.4)'))
    fig.update_layout(yaxis3=dict(title="Volume"))

    st.plotly_chart(fig)
