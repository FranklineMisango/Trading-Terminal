import numpy as np
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from langchain_core.tools import tool


@tool
def tool_sma_histogram(start_date:dt.time, end_date:dt.time, ticker:str):
    '''This function calculates the percentage change from a Simple Moving Average (SMA) and plots a histogram.'''
    stock = ticker
    # Fetch stock data
    df = yf.download(stock, start_date, end_date)

    # Calculate Simple Moving Average (SMA)
    sma = 50
    df['SMA' + str(sma)] = df['Adj Close'].rolling(window=sma).mean()
    
    # Calculate percentage change from SMA
    df['PC'] = ((df["Adj Close"] / df['SMA' + str(sma)]) - 1) * 100

    # Calculating statistics
    mean = df["PC"].mean()
    stdev = df["PC"].std()
    current = df["PC"].iloc[-1]
    yday = df["PC"].iloc[-2]

    # Histogram settings
    bins = np.arange(-100, 100, 1)

    # Plotting histogram
    fig = go.Figure()

    # Add histogram trace
    fig.add_trace(go.Histogram(x=df["PC"], histnorm='percent', nbinsx=len(bins), name='Count'))

    # Adding vertical lines for mean, std deviation, current and yesterday's percentage change
    for i in range(-3, 4):
        fig.add_shape(
            dict(type="line", x0=mean + i * stdev, y0=0, x1=mean + i * stdev, y1=100, line=dict(color="gray", dash="dash"),
                opacity=0.5 + abs(i)/6)
        )
    fig.add_shape(
        dict(type="line", x0=current, y0=0, x1=current, y1=100, line=dict(color="red"), name='Today')
    )
    fig.add_shape(
        dict(type="line", x0=yday, y0=0, x1=yday, y1=100, line=dict(color="blue"), name='Yesterday')
    )

    # Update layout
    fig.update_layout(
        title=f"{stock} - % From {sma} SMA Histogram since {start_date.year}",
        xaxis_title=f'Percent from {sma} SMA (bin size = 1)',
        yaxis_title='Percentage of Total',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig)


def norm_sma_histogram(start_date, end_date, ticker):
    '''This function calculates the percentage change from a Simple Moving Average (SMA) and plots a histogram.'''
    stock = ticker
    # Fetch stock data
    df = yf.download(stock, start_date, end_date)

    # Calculate Simple Moving Average (SMA)
    sma = 50
    df['SMA' + str(sma)] = df['Adj Close'].rolling(window=sma).mean()
    
    # Calculate percentage change from SMA
    df['PC'] = ((df["Adj Close"] / df['SMA' + str(sma)]) - 1) * 100

    # Calculating statistics
    mean = df["PC"].mean()
    stdev = df["PC"].std()
    current = df["PC"].iloc[-1]
    yday = df["PC"].iloc[-2]

    # Histogram settings
    bins = np.arange(-100, 100, 1)

    # Plotting histogram
    fig = go.Figure()

    # Add histogram trace
    fig.add_trace(go.Histogram(x=df["PC"], histnorm='percent', nbinsx=len(bins), name='Count'))

    # Adding vertical lines for mean, std deviation, current and yesterday's percentage change
    for i in range(-3, 4):
        fig.add_shape(
            dict(type="line", x0=mean + i * stdev, y0=0, x1=mean + i * stdev, y1=100, line=dict(color="gray", dash="dash"),
                opacity=0.5 + abs(i)/6)
        )
    fig.add_shape(
        dict(type="line", x0=current, y0=0, x1=current, y1=100, line=dict(color="red"), name='Today')
    )
    fig.add_shape(
        dict(type="line", x0=yday, y0=0, x1=yday, y1=100, line=dict(color="blue"), name='Yesterday')
    )

    # Update layout
    fig.update_layout(
        title=f"{stock} - % From {sma} SMA Histogram since {start_date.year}",
        xaxis_title=f'Percent from {sma} SMA (bin size = 1)',
        yaxis_title='Percentage of Total',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig)