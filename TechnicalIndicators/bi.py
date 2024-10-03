import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_bi(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Beta Indicator (BI) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    market = "^GSPC"
    # Read data
    df = yf.download(symbol, start, end)
    mk = yf.download(market, start, end)

    df["Returns"] = df["Adj Close"].pct_change().dropna()
    mk["Returns"] = mk["Adj Close"].pct_change().dropna()

    n = 5
    covar = df["Returns"].rolling(n).cov(mk["Returns"])
    variance = mk["Returns"].rolling(n).var()
    df["Beta"] = covar / variance

    # Stock Closing Price Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

    # Beta Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Beta"], mode='lines', name='Beta', line=dict(color='red')))
    fig.update_layout(title="Beta",
                    xaxis_title="Date",
                    yaxis_title="Beta")
    st.plotly_chart(fig)

    # Candlestick Chart with Beta
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))
    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with Beta",
                    xaxis_title="Date",
                    yaxis_title="Price")
    fig.add_trace(go.Scatter(x=df.index, y=df["Beta"], mode='lines', name='Beta', line=dict(color='red')))
    st.plotly_chart(fig)   

def norm_bi(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    market = "^GSPC"
    # Read data
    df = yf.download(symbol, start, end)
    mk = yf.download(market, start, end)

    df["Returns"] = df["Adj Close"].pct_change().dropna()
    mk["Returns"] = mk["Adj Close"].pct_change().dropna()

    n = 5
    covar = df["Returns"].rolling(n).cov(mk["Returns"])
    variance = mk["Returns"].rolling(n).var()
    df["Beta"] = covar / variance

    # Stock Closing Price Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

    # Beta Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Beta"], mode='lines', name='Beta', line=dict(color='red')))
    fig.update_layout(title="Beta",
                    xaxis_title="Date",
                    yaxis_title="Beta")
    st.plotly_chart(fig)

    # Candlestick Chart with Beta
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))
    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with Beta",
                    xaxis_title="Date",
                    yaxis_title="Price")
    fig.add_trace(go.Scatter(x=df.index, y=df["Beta"], mode='lines', name='Beta', line=dict(color='red')))
    st.plotly_chart(fig)   
