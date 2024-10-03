import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_atr(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This tool is for ATR of a stock'''
   symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    n = 14
    df["HL"] = df["High"] - df["Low"]
    df["HC"] = abs(df["High"] - df["Adj Close"].shift())
    df["LC"] = abs(df["Low"] - df["Adj Close"].shift())
    df["TR"] = df[["HL", "HC", "LC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(n).mean()
    df = df.drop(["HL", "HC", "LC", "TR"], axis=1)

    # Simple Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # ATR Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], mode='lines', name='ATR'))
    fig.add_shape(type="line", x0=df.index[0], y0=1, x1=df.index[-1], y1=1, line=dict(color="black", width=1, dash="dash"))
    fig.update_layout(title="Average True Range (ATR)",
                    xaxis_title="Date",
                    yaxis_title="ATR",
                    legend=dict(x=0, y=1, traceorder="normal"))
    
    st.plotly_chart(fig)

    # Candlestick Chart with ATR
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with ATR",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], mode='lines', name='ATR'))
    fig.add_shape(type="line", x0=df.index[0], y0=1, x1=df.index[-1], y1=1, line=dict(color="black", width=1, dash="dash"))
    
    st.plotly_chart(fig) 

def norm_atr(ticker, start_date, end_date):
    '''This tool is for ATR of a stock'''
   symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    n = 14
    df["HL"] = df["High"] - df["Low"]
    df["HC"] = abs(df["High"] - df["Adj Close"].shift())
    df["LC"] = abs(df["Low"] - df["Adj Close"].shift())
    df["TR"] = df[["HL", "HC", "LC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(n).mean()
    df = df.drop(["HL", "HC", "LC", "TR"], axis=1)

    # Simple Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # ATR Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], mode='lines', name='ATR'))
    fig.add_shape(type="line", x0=df.index[0], y0=1, x1=df.index[-1], y1=1, line=dict(color="black", width=1, dash="dash"))
    fig.update_layout(title="Average True Range (ATR)",
                    xaxis_title="Date",
                    yaxis_title="ATR",
                    legend=dict(x=0, y=1, traceorder="normal"))
    
    st.plotly_chart(fig)

    # Candlestick Chart with ATR
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))

    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with ATR",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))

    fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], mode='lines', name='ATR'))
    fig.add_shape(type="line", x0=df.index[0], y0=1, x1=df.index[-1], y1=1, line=dict(color="black", width=1, dash="dash"))
    
    st.plotly_chart(fig) 
