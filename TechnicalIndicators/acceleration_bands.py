import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_accleration_bands(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots the Acceleration Bands of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)


    n = 7
    UBB = df["High"] * (1 + 4 * (df["High"] - df["Low"]) / (df["High"] + df["Low"]))
    df["Upper_Band"] = UBB.rolling(n, center=False).mean()
    df["Middle_Band"] = df["Adj Close"].rolling(n).mean()
    LBB = df["Low"] * (1 - 4 * (df["High"] - df["Low"]) / (df["High"] + df["Low"]))
    df["Lower_Band"] = LBB.rolling(n, center=False).mean()

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper_Band"], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Middle_Band"], mode='lines', name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower_Band"], mode='lines', name='Lower Band'))
    fig.update_layout(title="Stock Closing Price of " + str(n) + "-Day Acceleration Bands",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index, y=df["Upper_Band"], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Middle_Band"], mode='lines', name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower_Band"], mode='lines', name='Lower Band'))
    fig.update_layout(title="Stock " + symbol + " Closing Price with Acceleration Bands",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    st.plotly_chart(fig)

def norm_accleration_bands(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)


    n = 7
    UBB = df["High"] * (1 + 4 * (df["High"] - df["Low"]) / (df["High"] + df["Low"]))
    df["Upper_Band"] = UBB.rolling(n, center=False).mean()
    df["Middle_Band"] = df["Adj Close"].rolling(n).mean()
    LBB = df["Low"] * (1 - 4 * (df["High"] - df["Low"]) / (df["High"] + df["Low"]))
    df["Lower_Band"] = LBB.rolling(n, center=False).mean()

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper_Band"], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Middle_Band"], mode='lines', name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower_Band"], mode='lines', name='Lower Band'))
    fig.update_layout(title="Stock Closing Price of " + str(n) + "-Day Acceleration Bands",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.add_trace(go.Scatter(x=df.index, y=df["Upper_Band"], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Middle_Band"], mode='lines', name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower_Band"], mode='lines', name='Lower Band'))
    fig.update_layout(title="Stock " + symbol + " Closing Price with Acceleration Bands",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    st.plotly_chart(fig)