import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_pp(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This program allows you to visualize Pivot Points(PP) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Pivot Points
    PP = (dataset["High"] + dataset["Low"] + dataset["Close"]) / 3
    R1 = 2 * PP - dataset["Low"]
    S1 = 2 * PP - dataset["High"]
    R2 = PP + dataset["High"] - dataset["Low"]
    S2 = PP - dataset["High"] + dataset["Low"]
    R3 = dataset["High"] + 2 * (PP - dataset["Low"])
    S3 = dataset["Low"] - 2 * (dataset["High"] - PP)

    # Plot Pivot Points
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=PP, mode='lines', name='Pivot Point'),
                        go.Scatter(x=dataset.index, y=R1, mode='lines', name='R1'),
                        go.Scatter(x=dataset.index, y=S1, mode='lines', name='S1'),
                        go.Scatter(x=dataset.index, y=R2, mode='lines', name='R2'),
                        go.Scatter(x=dataset.index, y=S2, mode='lines', name='S2'),
                        go.Scatter(x=dataset.index, y=R3, mode='lines', name='R3'),
                        go.Scatter(x=dataset.index, y=S3, mode='lines', name='S3')])
    fig.update_layout(title=f"{symbol} Pivot Points",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig)

def norm_pp(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    # Calculate Pivot Points
    PP = (dataset["High"] + dataset["Low"] + dataset["Close"]) / 3
    R1 = 2 * PP - dataset["Low"]
    S1 = 2 * PP - dataset["High"]
    R2 = PP + dataset["High"] - dataset["Low"]
    S2 = PP - dataset["High"] + dataset["Low"]
    R3 = dataset["High"] + 2 * (PP - dataset["Low"])
    S3 = dataset["Low"] - 2 * (dataset["High"] - PP)

    # Plot Pivot Points
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=PP, mode='lines', name='Pivot Point'),
                        go.Scatter(x=dataset.index, y=R1, mode='lines', name='R1'),
                        go.Scatter(x=dataset.index, y=S1, mode='lines', name='S1'),
                        go.Scatter(x=dataset.index, y=R2, mode='lines', name='R2'),
                        go.Scatter(x=dataset.index, y=S2, mode='lines', name='S2'),
                        go.Scatter(x=dataset.index, y=R3, mode='lines', name='R3'),
                        go.Scatter(x=dataset.index, y=S3, mode='lines', name='S3')])
    fig.update_layout(title=f"{symbol} Pivot Points",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig)
