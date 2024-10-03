import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_bbw(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Bollinger Bandwidth (BBW) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 20
    MA = pd.Series(df["Adj Close"].rolling(n).mean())
    STD = pd.Series(df["Adj Close"].rolling(n).std())
    bb1 = MA + 2 * STD
    df["Upper Bollinger Band"] = pd.Series(bb1)
    bb2 = MA - 2 * STD
    df["Lower Bollinger Band"] = pd.Series(bb2)
    df["SMA"] = df["Adj Close"].rolling(n).mean()

    df["BBWidth"] = ((df["Upper Bollinger Band"] - df["Lower Bollinger Band"]) / df["SMA"] * 100)

    # Bollinger Bands Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper Bollinger Band"], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower Bollinger Band"], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"].rolling(20).mean(), mode='lines', name='Mean Average', line=dict(color='orange', dash='dash')))
    fig.update_layout(title=f"{symbol} Bollinger Bands",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

    # BB Width Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["BBWidth"], mode='lines', name='BB Width', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df.index, y=df["BBWidth"].rolling(20).mean(), mode='lines', name='200 Moving Average', line=dict(color='darkblue')))
    fig.update_layout(title="Bollinger Bands Width",
                    xaxis_title="Date",
                    yaxis_title="BB Width")
    st.plotly_chart(fig)

def norm_bbw(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 20
    MA = pd.Series(df["Adj Close"].rolling(n).mean())
    STD = pd.Series(df["Adj Close"].rolling(n).std())
    bb1 = MA + 2 * STD
    df["Upper Bollinger Band"] = pd.Series(bb1)
    bb2 = MA - 2 * STD
    df["Lower Bollinger Band"] = pd.Series(bb2)
    df["SMA"] = df["Adj Close"].rolling(n).mean()

    df["BBWidth"] = ((df["Upper Bollinger Band"] - df["Lower Bollinger Band"]) / df["SMA"] * 100)

    # Bollinger Bands Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper Bollinger Band"], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower Bollinger Band"], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"].rolling(20).mean(), mode='lines', name='Mean Average', line=dict(color='orange', dash='dash')))
    fig.update_layout(title=f"{symbol} Bollinger Bands",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

    # BB Width Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["BBWidth"], mode='lines', name='BB Width', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df.index, y=df["BBWidth"].rolling(20).mean(), mode='lines', name='200 Moving Average', line=dict(color='darkblue')))
    fig.update_layout(title="Bollinger Bands Width",
                    xaxis_title="Date",
                    yaxis_title="BB Width")
    st.plotly_chart(fig)

