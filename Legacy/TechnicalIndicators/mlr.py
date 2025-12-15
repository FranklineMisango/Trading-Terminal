import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_mlr(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Moving Linear Regression(MLR) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df['Slope'] = ta.LINEARREG_SLOPE(df['Adj Close'], timeperiod=14)

    # Plot Line Chart with Moving Linear Regression
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Slope"], mode='lines', name='Slope', line=dict(color='red')))
    fig_line.update_layout(title=f"Stock {symbol} Moving Linear Regression",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)

def norm_mlr(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df['Slope'] = ta.LINEARREG_SLOPE(df['Adj Close'], timeperiod=14)

    # Plot Line Chart with Moving Linear Regression
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Slope"], mode='lines', name='Slope', line=dict(color='red')))
    fig_line.update_layout(title=f"Stock {symbol} Moving Linear Regression",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)
