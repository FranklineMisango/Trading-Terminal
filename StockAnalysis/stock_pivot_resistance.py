import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt
import pandas as pd
from langchain_core.tools import tool

@tool
def tool_plot_stock_pivot_resistance(stock_symbol :str, start_date :dt.time, end_date : dt.time):
    '''This function fetches stock data and plots resistance points based on pivot points.''' 
    df = yf.download(stock_symbol, start_date, end_date)
    # Plot high prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["High"], mode='lines', name='High'))
    
    # Initialize variables to find and store pivot points
    pivots = []
    dates = []
    counter = 0
    last_pivot = 0
    window_size = 10
    window = [0] * window_size
    date_window = [0] * window_size

    # Identify pivot points
    for i, high_price in enumerate(df["High"]):
        window = window[1:] + [high_price]
        date_window = date_window[1:] + [df.index[i]]

        current_max = max(window)
        if current_max == last_pivot:
            counter += 1
        else:
            counter = 0

        if counter == 5:
            last_pivot = current_max
            last_date = date_window[window.index(last_pivot)]
            pivots.append(last_pivot)
            dates.append(last_date)
    # Plot resistance levels for each pivot point
    for i in range(len(pivots)):
        time_delta = dt.timedelta(days=30)
        fig.add_shape(type="line",
                    x0=dates[i], y0=pivots[i],
                    x1=dates[i] + time_delta, y1=pivots[i],
                    line=dict(color="red", width=4, dash="solid")
                    )
    # Configure plot settings
    fig.update_layout(title=stock_symbol.upper() + ' Resistance Points', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)


def norm_plot_stock_pivot_resistance(stock_symbol, start_date, end_date):
    '''This function fetches stock data and plots resistance points based on pivot points.''' 
    df = yf.download(stock_symbol, start_date, end_date)
    # Plot high prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["High"], mode='lines', name='High'))
    
    # Initialize variables to find and store pivot points
    pivots = []
    dates = []
    counter = 0
    last_pivot = 0
    window_size = 10
    window = [0] * window_size
    date_window = [0] * window_size

    # Identify pivot points
    for i, high_price in enumerate(df["High"]):
        window = window[1:] + [high_price]
        date_window = date_window[1:] + [df.index[i]]

        current_max = max(window)
        if current_max == last_pivot:
            counter += 1
        else:
            counter = 0

        if counter == 5:
            last_pivot = current_max
            last_date = date_window[window.index(last_pivot)]
            pivots.append(last_pivot)
            dates.append(last_date)
    # Plot resistance levels for each pivot point
    for i in range(len(pivots)):
        time_delta = dt.timedelta(days=30)
        fig.add_shape(type="line",
                    x0=dates[i], y0=pivots[i],
                    x1=dates[i] + time_delta, y1=pivots[i],
                    line=dict(color="red", width=4, dash="solid")
                    )
    # Configure plot settings
    fig.update_layout(title=stock_symbol.upper() + ' Resistance Points', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)