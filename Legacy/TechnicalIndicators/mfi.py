import streamlit as st
import datetime as dt
import yfinance as yf
from langchain_core.tools import tool
from plotly import graph_objs as go
import pandas as pd



@tool
def tool_mfi(start_date: dt.time, end_date: dt.time, ticker: str):
    ''' This tool plots the candlestick chart of a stock along with the Money Flow Index (MFI) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Money Flow Index (MFI)
    def calculate_mfi(high, low, close, volume, period=14):
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = (raw_money_flow * (close > close.shift(1))).rolling(window=period).sum()
        negative_flow = (raw_money_flow * (close < close.shift(1))).rolling(window=period).sum()
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    df["MFI"] = calculate_mfi(df["High"], df["Low"], df["Close"], df["Volume"])    

    # Plot interactive chart for closing price
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig_close.update_layout(
        title="Interactive Chart for Closing Price",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        legend=dict(x=0, y=1),
    )

    # Find min and max price
    min_price = df["Close"].min()
    max_price = df["Close"].max()

    # Plot interactive chart for MFI
    fig_mfi = go.Figure()
    fig_mfi.add_trace(go.Scatter(x=df.index, y=df["MFI"], mode="lines", name="MFI"))
    fig_mfi.update_layout(
        title="Interactive Chart for Money Flow Index (MFI)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="MFI"),
        legend=dict(x=0, y=1),
        shapes=[
            dict(
                type="line",
                x0=df.index[0],
                y0=min_price,
                x1=df.index[-1],
                y1=min_price,
                line=dict(color="blue", width=1, dash="dash"),
            ),
            dict(
                type="line",
                x0=df.index[0],
                y0=max_price,
                x1=df.index[-1],
                y1=max_price,
                line=dict(color="red", width=1, dash="dash"),
            ),
        ],
    )
    
    # Plot interactive chart
    fig_main = go.Figure()

    # Add closing price trace
    fig_main.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

    # Add MFI trace
    fig_main.add_trace(go.Scatter(x=df.index, y=df["MFI"], mode="lines", name="MFI"))

    # Update layout
    fig_main.update_layout(
        title="Interactive Chart with Money Flow Index (MFI)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="MFI", overlaying="y", side="right"),
        legend=dict(x=0, y=1),
    )

    # Show interactive chart
    st.plotly_chart(fig_close)
    st.plotly_chart(fig_mfi)
    st.plotly_chart(fig_main) 


#User function 
def norm_mfi(start_date, end_date, ticker):
    ''' This tool plots the candlestick chart of a stock along with the Money Flow Index (MFI) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Money Flow Index (MFI)
    def calculate_mfi(high, low, close, volume, period=14):
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = (raw_money_flow * (close > close.shift(1))).rolling(window=period).sum()
        negative_flow = (raw_money_flow * (close < close.shift(1))).rolling(window=period).sum()
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    df["MFI"] = calculate_mfi(df["High"], df["Low"], df["Close"], df["Volume"])    

    # Plot interactive chart for closing price
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig_close.update_layout(
        title="Interactive Chart for Closing Price",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        legend=dict(x=0, y=1),
    )

    # Find min and max price
    min_price = df["Close"].min()
    max_price = df["Close"].max()

    # Plot interactive chart for MFI
    fig_mfi = go.Figure()
    fig_mfi.add_trace(go.Scatter(x=df.index, y=df["MFI"], mode="lines", name="MFI"))
    fig_mfi.update_layout(
        title="Interactive Chart for Money Flow Index (MFI)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="MFI"),
        legend=dict(x=0, y=1),
        shapes=[
            dict(
                type="line",
                x0=df.index[0],
                y0=min_price,
                x1=df.index[-1],
                y1=min_price,
                line=dict(color="blue", width=1, dash="dash"),
            ),
            dict(
                type="line",
                x0=df.index[0],
                y0=max_price,
                x1=df.index[-1],
                y1=max_price,
                line=dict(color="red", width=1, dash="dash"),
            ),
        ],
    )
    
    # Plot interactive chart
    fig_main = go.Figure()

    # Add closing price trace
    fig_main.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

    # Add MFI trace
    fig_main.add_trace(go.Scatter(x=df.index, y=df["MFI"], mode="lines", name="MFI"))

    # Update layout
    fig_main.update_layout(
        title="Interactive Chart with Money Flow Index (MFI)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="MFI", overlaying="y", side="right"),
        legend=dict(x=0, y=1),
    )

    # Show interactive chart
    st.plotly_chart(fig_close)
    st.plotly_chart(fig_mfi)
    st.plotly_chart(fig_main) 