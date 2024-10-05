import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_evm(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots Ease of Movement (EVM) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)
    # Create a function for Ease of Movement
    def EVM(data, ndays):
        dm = ((data["High"] + data["Low"]) / 2) - (
            (data["High"].shift(1) + data["Low"].shift(1)) / 2
        )
        br = (data["Volume"] / 100000000) / ((data["High"] - data["Low"]))
        EVM = dm / br
        EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name="EVM")
        data = data.join(EVM_MA)
        return data

    # Compute the 14-day Ease of Movement for stock
    n = 14
    Stock_EVM = EVM(df, n)
    EVM = Stock_EVM["EVM"]

    # Compute EVM
    df = EVM(df, 14)

    # Plot EVM
    fig = go.Figure(data=[
        go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Adj Close'],
                    name='Candlestick'),
        go.Scatter(x=df.index,
                y=df['EVM'],
                mode='lines',
                name='Ease of Movement')
    ])
    fig.update_layout(title=f"Ease of Movement (EVM) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_evm(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)
    # Create a function for Ease of Movement
    def EVM(data, ndays):
        dm = ((data["High"] + data["Low"]) / 2) - (
            (data["High"].shift(1) + data["Low"].shift(1)) / 2
        )
        br = (data["Volume"] / 100000000) / ((data["High"] - data["Low"]))
        EVM = dm / br
        EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name="EVM")
        data = data.join(EVM_MA)
        return data

    # Compute the 14-day Ease of Movement for stock
    n = 14
    Stock_EVM = EVM(df, n)
    EVM = Stock_EVM["EVM"]

    # Compute EVM
    df = EVM(df, 14)

    # Plot EVM
    fig = go.Figure(data=[
        go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Adj Close'],
                    name='Candlestick'),
        go.Scatter(x=df.index,
                y=df['EVM'],
                mode='lines',
                name='Ease of Movement')
    ])
    fig.update_layout(title=f"Ease of Movement (EVM) for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)
