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
def tool_gann_lines_angles(start_date : dt.time, end_date : dt.time, ticker :str):
    ''' This tool plots the candlestick chart of a stock along with the GANN Angles of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Line Chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))

    # Add diagonal line
    x_lim = [df.index[0], df.index[-1]]
    y_lim = [df["Adj Close"].iloc[0], df["Adj Close"].iloc[-1]]
    fig_line.add_trace(go.Scatter(x=x_lim, y=y_lim, mode='lines', line=dict(color='red'), name='45 degree'))

    fig_line.update_layout(title="Stock of GANN Angles", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_line)

    # GANN Angles
    angles = [82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5]

    fig_gann = go.Figure()
    fig_gann.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))

    for angle in angles:
        angle_radians = np.radians(angle)
        y_values = np.tan(angle_radians) * np.arange(len(df))
        fig_gann.add_trace(go.Scatter(x=df.index, y=y_values, mode='markers', name=str(angle)))

    fig_gann.update_layout(title="Stock of GANN Angles", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_gann)

    # Candlestick with GANN Lines Angles
    fig_candlestick = go.Figure()

    # Candlestick
    fig_candlestick.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'],
                                            name='Candlestick'))

    # GANN Angles
    for angle in angles:
        angle_radians = np.radians(angle)
        y_values = np.tan(angle_radians) * np.arange(len(df))
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=y_values, mode='markers', name=str(angle)))

    # Volume
    fig_candlestick.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=np.where(df['Open'] < df['Close'], 'green', 'red')))

    fig_candlestick.update_layout(title="Stock Closing Price", xaxis_title="Date", yaxis_title="Price", 
                                yaxis2=dict(title="Volume", overlaying='y', side='right', tickformat=',.0f'))
    st.plotly_chart(fig_candlestick)


def norm_gann_lines_tracker(start_date, end_date,ticker):
    ''' This tool plots the candlestick chart of a stock along with the GANN Angles of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Line Chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))

    # Add diagonal line
    x_lim = [df.index[0], df.index[-1]]
    y_lim = [df["Adj Close"].iloc[0], df["Adj Close"].iloc[-1]]
    fig_line.add_trace(go.Scatter(x=x_lim, y=y_lim, mode='lines', line=dict(color='red'), name='45 degree'))

    fig_line.update_layout(title="Stock of GANN Angles", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_line)

    # GANN Angles
    angles = [82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5]

    fig_gann = go.Figure()
    fig_gann.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))

    for angle in angles:
        angle_radians = np.radians(angle)
        y_values = np.tan(angle_radians) * np.arange(len(df))
        fig_gann.add_trace(go.Scatter(x=df.index, y=y_values, mode='markers', name=str(angle)))

    fig_gann.update_layout(title="Stock of GANN Angles", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_gann)

    # Candlestick with GANN Lines Angles
    fig_candlestick = go.Figure()

    # Candlestick
    fig_candlestick.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'],
                                            name='Candlestick'))

    # GANN Angles
    for angle in angles:
        angle_radians = np.radians(angle)
        y_values = np.tan(angle_radians) * np.arange(len(df))
        fig_candlestick.add_trace(go.Scatter(x=df.index, y=y_values, mode='markers', name=str(angle)))

    # Volume
    fig_candlestick.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=np.where(df['Open'] < df['Close'], 'green', 'red')))

    fig_candlestick.update_layout(title="Stock Closing Price", xaxis_title="Date", yaxis_title="Price", 
                                yaxis2=dict(title="Volume", overlaying='y', side='right', tickformat=',.0f'))
    st.plotly_chart(fig_candlestick)