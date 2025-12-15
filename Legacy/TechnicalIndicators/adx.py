import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool

@tool
def tool_adx(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This tool is for ADX of a stock'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Simple Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                     xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    st.plotly_chart(fig)

    # ADX calculation
    adx = ta.ADX(df["High"], df["Low"], df["Adj Close"], timeperiod=14)
    adx = adx.dropna()

    # Line Chart with ADX
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=adx.index, y=adx, mode='lines', name='ADX'))
    fig.add_shape(type="line", x0=adx.index[0], y0=20, x1=adx.index[-1], y1=20, line=dict(color="red", width=1, dash="dash"))
    fig.add_shape(type="line", x0=adx.index[0], y0=50, x1=adx.index[-1], y1=50, line=dict(color="red", width=1, dash="dash"))
    st.plotly_chart(fig)

    # Candlestick Chart with ADX
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with ADX",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=adx.index, y=adx, mode='lines', name='ADX'))
    fig.add_shape(type="line", x0=adx.index[0], y0=20, x1=adx.index[-1], y1=20, line=dict(color="red", width=1, dash="dash"))
    fig.add_shape(type="line", x0=adx.index[0], y0=50, x1=adx.index[-1], y1=50, line=dict(color="red", width=1, dash="dash"))
    st.plotly_chart(fig)


def norm_adx(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Simple Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                     xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    st.plotly_chart(fig)

    # ADX calculation
    adx = ta.ADX(df["High"], df["Low"], df["Adj Close"], timeperiod=14)
    adx = adx.dropna()

    # Line Chart with ADX
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=adx.index, y=adx, mode='lines', name='ADX'))
    fig.add_shape(type="line", x0=adx.index[0], y0=20, x1=adx.index[-1], y1=20, line=dict(color="red", width=1, dash="dash"))
    fig.add_shape(type="line", x0=adx.index[0], y0=50, x1=adx.index[-1], y1=50, line=dict(color="red", width=1, dash="dash"))
    st.plotly_chart(fig)

    # Candlestick Chart with ADX
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with ADX",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=adx.index, y=adx, mode='lines', name='ADX'))
    fig.add_shape(type="line", x0=adx.index[0], y0=20, x1=adx.index[-1], y1=20, line=dict(color="red", width=1, dash="dash"))
    fig.add_shape(type="line", x0=adx.index[0], y0=50, x1=adx.index[-1], y1=50, line=dict(color="red", width=1, dash="dash"))
    st.plotly_chart(fig)