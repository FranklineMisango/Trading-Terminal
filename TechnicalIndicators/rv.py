import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_rv(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Realized Volatility(RV) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Realized Volatility
    returns = dataset["Adj Close"].pct_change().dropna()
    realized_volatility = returns.std() * np.sqrt(252)  # Annualized volatility assuming 252 trading days

    # Plot Realized Volatility
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns.index, y=returns, mode='lines', name='Daily Returns'))
    fig.add_trace(go.Scatter(x=returns.index, y=np.ones(len(returns)) * realized_volatility, mode='lines', name='Realized Volatility', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f"{symbol} Daily Returns and Realized Volatility",
                    xaxis_title="Date",
                    yaxis_title="Returns / Realized Volatility")
    st.plotly_chart(fig)

def norm_rv(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Realized Volatility
    returns = dataset["Adj Close"].pct_change().dropna()
    realized_volatility = returns.std() * np.sqrt(252)  # Annualized volatility assuming 252 trading days

    # Plot Realized Volatility
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns.index, y=returns, mode='lines', name='Daily Returns'))
    fig.add_trace(go.Scatter(x=returns.index, y=np.ones(len(returns)) * realized_volatility, mode='lines', name='Realized Volatility', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f"{symbol} Daily Returns and Realized Volatility",
                    xaxis_title="Date",
                    yaxis_title="Returns / Realized Volatility")
    st.plotly_chart(fig)

