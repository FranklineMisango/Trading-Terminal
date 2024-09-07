import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from langchain_core.tools import tool


@tool
def tool_exponential_weighted_moving_averages(start_date : dt.time, end_date : dt.time, ticker : str):
    ''' This tool plots the candlestick chart of a stock along with the Exponential Moving Average (EMA) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    df = yf.download(symbol, start, end)

    n = 7
    df["EWMA"] = df["Adj Close"].ewm(ignore_na=False, min_periods=n - 1, span=n).mean()
    # Plotting Candlestick with EWMA
    fig_candlestick = go.Figure()

    # Plot candlestick
    fig_candlestick.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'],
                                            name='Candlestick'))
    # Plot volume
    volume_color = df['Open'] < df['Close']  # Color based on close > open
    volume_color = volume_color.map({True: 'green', False: 'red'}) 
    fig_candlestick.add_trace(go.Bar(x=df.index,
                                    y=df['Volume'],
                                    marker_color=volume_color,  
                                    name='Volume'))

    # Plot EWMA
    fig_candlestick.add_trace(go.Scatter(x=df.index,
                                        y=df['EWMA'],
                                        mode='lines',
                                        name='EWMA',
                                        line=dict(color='red')))

    # Update layout
    fig_candlestick.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis=dict(title="Date"),
                                yaxis=dict(title="Price"),
                                yaxis2=dict(title="Volume", overlaying='y', side='right'),
                                legend=dict(yanchor="top", y=1, xanchor="left", x=0))

    # Display Plotly figure in Streamlit
    st.plotly_chart(fig_candlestick)



def norm_exponential_weighted_moving_averages(start_date, end_date, ticker):
    ''' This tool plots the candlestick chart of a stock along with the Exponential Moving Average (EMA) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    df = yf.download(symbol, start, end)

    n = 7
    df["EWMA"] = df["Adj Close"].ewm(ignore_na=False, min_periods=n - 1, span=n).mean()
    # Plotting Candlestick with EWMA
    fig_candlestick = go.Figure()

    # Plot candlestick
    fig_candlestick.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'],
                                            name='Candlestick'))
    # Plot volume
    volume_color = df['Open'] < df['Close']  # Color based on close > open
    volume_color = volume_color.map({True: 'green', False: 'red'}) 
    fig_candlestick.add_trace(go.Bar(x=df.index,
                                    y=df['Volume'],
                                    marker_color=volume_color,  
                                    name='Volume'))

    # Plot EWMA
    fig_candlestick.add_trace(go.Scatter(x=df.index,
                                        y=df['EWMA'],
                                        mode='lines',
                                        name='EWMA',
                                        line=dict(color='red')))

    # Update layout
    fig_candlestick.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis=dict(title="Date"),
                                yaxis=dict(title="Price"),
                                yaxis2=dict(title="Volume", overlaying='y', side='right'),
                                legend=dict(yanchor="top", y=1, xanchor="left", x=0))

    # Display Plotly figure in Streamlit
    st.plotly_chart(fig_candlestick)