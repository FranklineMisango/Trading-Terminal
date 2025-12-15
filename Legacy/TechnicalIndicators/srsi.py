import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_srsi(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This program allows you to visualize Stochastic RSI(SRSI) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate RSI
    n = 14
    change = dataset["Adj Close"].diff(1)
    gain = change.mask(change < 0, 0)
    loss = abs(change.mask(change > 0, 0))
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))

    # Calculate Stochastic RSI
    LL_RSI = RSI.rolling(14).min()
    HH_RSI = RSI.rolling(14).max()
    dataset["Stoch_RSI"] = (RSI - LL_RSI) / (HH_RSI - LL_RSI)

    # Plot Stochastic RSI
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["Stoch_RSI"], mode='lines', name='Stochastic RSI', line=dict(color='red'))])
    fig.update_layout(title=f"{symbol} Stochastic RSI",
                    xaxis_title="Date",
                    yaxis_title="Stochastic RSI")
    st.plotly_chart(fig)

def norm_srsi(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate RSI
    n = 14
    change = dataset["Adj Close"].diff(1)
    gain = change.mask(change < 0, 0)
    loss = abs(change.mask(change > 0, 0))
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))

    # Calculate Stochastic RSI
    LL_RSI = RSI.rolling(14).min()
    HH_RSI = RSI.rolling(14).max()
    dataset["Stoch_RSI"] = (RSI - LL_RSI) / (HH_RSI - LL_RSI)

    # Plot Stochastic RSI
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["Stoch_RSI"], mode='lines', name='Stochastic RSI', line=dict(color='red'))])
    fig.update_layout(title=f"{symbol} Stochastic RSI",
                    xaxis_title="Date",
                    yaxis_title="Stochastic RSI")
    st.plotly_chart(fig)