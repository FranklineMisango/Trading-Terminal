import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool
import pandas as pd

@tool
def tool_bb(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Bollinger Bands (BB) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)
    df["VolumePositive"] = df["Open"] < df["Adj Close"]

    n = 20
    MA = pd.Series(df["Adj Close"].rolling(n).mean())
    STD = pd.Series(df["Adj Close"].rolling(n).std())
    bb1 = MA + 2 * STD
    df["Upper Bollinger Band"] = pd.Series(bb1)
    bb2 = MA - 2 * STD
    df["Lower Bollinger Band"] = pd.Series(bb2)

    # Bollinger Bands Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper Bollinger Band"], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower Bollinger Band"], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
    fig.update_layout(title=f"{symbol} Bollinger Bands",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

    # Candlestick Chart with Bollinger Bands
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick')])
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper Bollinger Band"], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower Bollinger Band"], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
    fig.update_layout(title="Candlestick Chart with Bollinger Bands",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

    # Volume Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name='Volume', marker_color=df.VolumePositive.map({True: "green", False: "red"})))
    fig.update_layout(title="Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume")
    st.plotly_chart(fig) 


def norm_bb(ticker, start_date, end_date):
        symbol = ticker
        start = start_date
        end = end_date

        # Read data
        df = yf.download(symbol, start, end)
        df["VolumePositive"] = df["Open"] < df["Adj Close"]

        n = 20
        MA = pd.Series(df["Adj Close"].rolling(n).mean())
        STD = pd.Series(df["Adj Close"].rolling(n).std())
        bb1 = MA + 2 * STD
        df["Upper Bollinger Band"] = pd.Series(bb1)
        bb2 = MA - 2 * STD
        df["Lower Bollinger Band"] = pd.Series(bb2)

        # Bollinger Bands Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
        fig.add_trace(go.Scatter(x=df.index, y=df["Upper Bollinger Band"], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower Bollinger Band"], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
        fig.update_layout(title=f"{symbol} Bollinger Bands",
                        xaxis_title="Date",
                        yaxis_title="Price")
        st.plotly_chart(fig)

        # Candlestick Chart with Bollinger Bands
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'], name='Candlestick')])
        fig.add_trace(go.Scatter(x=df.index, y=df["Upper Bollinger Band"], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower Bollinger Band"], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
        fig.update_layout(title="Candlestick Chart with Bollinger Bands",
                        xaxis_title="Date",
                        yaxis_title="Price")
        st.plotly_chart(fig)

        # Volume Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name='Volume', marker_color=df.VolumePositive.map({True: "green", False: "red"})))
        fig.update_layout(title="Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume")
        st.plotly_chart(fig)  