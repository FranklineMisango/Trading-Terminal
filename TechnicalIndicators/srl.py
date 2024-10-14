import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_srl(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Speed Resistance Lines(SRL) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    dataset["Middle_Line"] = dataset["Low"] + (dataset["High"] - dataset["Low"]) * 0.667
    dataset["Lower_Line"] = dataset["Low"] + (dataset["High"] - dataset["Low"]) * 0.333

    # Plot Speed Resistance Lines
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["Middle_Line"], mode='lines', name='Middle Line', line=dict(color='red')),
                        go.Scatter(x=dataset.index, y=dataset["Lower_Line"], mode='lines', name='Lower Line', line=dict(color='green'))])
    fig.update_layout(title=f"{symbol} Speed Resistance Lines",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)

def norm_srl(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    dataset = yf.download(symbol, start, end)

    dataset["Middle_Line"] = dataset["Low"] + (dataset["High"] - dataset["Low"]) * 0.667
    dataset["Lower_Line"] = dataset["Low"] + (dataset["High"] - dataset["Low"]) * 0.333

    # Plot Speed Resistance Lines
    fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                        open=dataset['Open'],
                                        high=dataset['High'],
                                        low=dataset['Low'],
                                        close=dataset['Close'],
                                        name='Candlesticks'),
                        go.Scatter(x=dataset.index, y=dataset["Middle_Line"], mode='lines', name='Middle Line', line=dict(color='red')),
                        go.Scatter(x=dataset.index, y=dataset["Lower_Line"], mode='lines', name='Lower Line', line=dict(color='green'))])
    fig.update_layout(title=f"{symbol} Speed Resistance Lines",
                    xaxis_title="Date",
                    yaxis_title="Price")
    st.plotly_chart(fig)
    