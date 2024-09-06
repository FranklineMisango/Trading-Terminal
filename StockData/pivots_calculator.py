import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_pivots_calculator(stock, interval):
    ''''This tool calculates pivot points and support/resistance levels for a given stock'''
    ticker = yf.Ticker(stock)
    df = ticker.history(interval)

    # Extract data for the last trading day and remove unnecessary columns
    last_day = df.tail(1).copy().drop(columns=['Dividends', 'Stock Splits'])

    # Calculate pivot points and support/resistance levels
    # Pivot point formula: (High + Low + Close) / 3
    last_day['Pivot'] = (last_day['High'] + last_day['Low'] + last_day['Close']) / 3
    last_day['R1'] = 2 * last_day['Pivot'] - last_day['Low']  # Resistance 1
    last_day['S1'] = 2 * last_day['Pivot'] - last_day['High']  # Support 1
    last_day['R2'] = last_day['Pivot'] + (last_day['High'] - last_day['Low'])  # Resistance 2
    last_day['S2'] = last_day['Pivot'] - (last_day['High'] - last_day['Low'])  # Support 2
    last_day['R3'] = last_day['Pivot'] + 2 * (last_day['High'] - last_day['Low'])  # Resistance 3
    last_day['S3'] = last_day['Pivot'] - 2 * (last_day['High'] - last_day['Low'])  # Support 3

    # Display calculated pivot points and support/resistance levels for the last trading day
    st.write(last_day)

    # Fetch intraday data for the specified stock
    data = yf.download(tickers=stock, period="1d", interval="1m")

    # Extract 'Close' prices from the intraday data for plotting
    df = data['Close']

    # Create Plotly figure
    fig = go.Figure()

    # Plot intraday data
    fig.add_trace(go.Scatter(x=df.index, y=df.values, mode='lines', name='Price'))

    # Plot support and resistance levels
    for level, color in zip(['R1', 'S1', 'R2', 'S2', 'R3', 'S3'], ['blue', 'blue', 'green', 'green', 'red', 'red']):
        fig.add_trace(go.Scatter(x=df.index, y=[last_day[level].iloc[0]] * len(df.index),
                                mode='lines', name=level, line=dict(color=color, dash='dash')))

    # Customize layout
    fig.update_layout(title=f"{stock.upper()} - {dt.date.today()}",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    showlegend=True)

    # Display Plotly figure
    st.plotly_chart(fig)


def norm_pivots_calculator(stock, interval):
    ''''This tool calculates pivot points and support/resistance levels for a given stock'''
    ticker = yf.Ticker(stock)
    df = ticker.history(interval)

    # Extract data for the last trading day and remove unnecessary columns
    last_day = df.tail(1).copy().drop(columns=['Dividends', 'Stock Splits'])

    # Calculate pivot points and support/resistance levels
    # Pivot point formula: (High + Low + Close) / 3
    last_day['Pivot'] = (last_day['High'] + last_day['Low'] + last_day['Close']) / 3
    last_day['R1'] = 2 * last_day['Pivot'] - last_day['Low']  # Resistance 1
    last_day['S1'] = 2 * last_day['Pivot'] - last_day['High']  # Support 1
    last_day['R2'] = last_day['Pivot'] + (last_day['High'] - last_day['Low'])  # Resistance 2
    last_day['S2'] = last_day['Pivot'] - (last_day['High'] - last_day['Low'])  # Support 2
    last_day['R3'] = last_day['Pivot'] + 2 * (last_day['High'] - last_day['Low'])  # Resistance 3
    last_day['S3'] = last_day['Pivot'] - 2 * (last_day['High'] - last_day['Low'])  # Support 3

    # Display calculated pivot points and support/resistance levels for the last trading day
    st.write(last_day)

    # Fetch intraday data for the specified stock
    data = yf.download(tickers=stock, period="1d", interval="1m")

    # Extract 'Close' prices from the intraday data for plotting
    df = data['Close']

    # Create Plotly figure
    fig = go.Figure()

    # Plot intraday data
    fig.add_trace(go.Scatter(x=df.index, y=df.values, mode='lines', name='Price'))

    # Plot support and resistance levels
    for level, color in zip(['R1', 'S1', 'R2', 'S2', 'R3', 'S3'], ['blue', 'blue', 'green', 'green', 'red', 'red']):
        fig.add_trace(go.Scatter(x=df.index, y=[last_day[level].iloc[0]] * len(df.index),
                                mode='lines', name=level, line=dict(color=color, dash='dash')))

    # Customize layout
    fig.update_layout(title=f"{stock.upper()} - {dt.date.today()}",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    showlegend=True)

    # Display Plotly figure
    st.plotly_chart(fig)