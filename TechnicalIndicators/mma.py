import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_mma(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Moving Average Envelopes (MMA) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["20SMA"] = ta.SMA(df["Adj Close"], timeperiod=20)
    df["Upper_Envelope"] = df["20SMA"] + (df["20SMA"] * 0.025)
    df["Lower_Envelope"] = df["20SMA"] - (df["20SMA"] * 0.025)

    # Plot Line Chart with Moving Average Envelopes (MMA)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Upper_Envelope"], mode='lines', name='Upper Envelope', line=dict(color='blue')))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Lower_Envelope"], mode='lines', name='Lower Envelope', line=dict(color='red')))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"].rolling(20).mean(), mode='lines', name='Moving Average', line=dict(color='orange', dash='dash')))
    fig_line.update_layout(title=f"Stock {symbol} Moving Average Envelopes (MMA)",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)

def norm_mma(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    df["20SMA"] = ta.SMA(df["Adj Close"], timeperiod=20)
    df["Upper_Envelope"] = df["20SMA"] + (df["20SMA"] * 0.025)
    df["Lower_Envelope"] = df["20SMA"] - (df["20SMA"] * 0.025)

    # Plot Line Chart with Moving Average Envelopes (MMA)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Upper_Envelope"], mode='lines', name='Upper Envelope', line=dict(color='blue')))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Lower_Envelope"], mode='lines', name='Lower Envelope', line=dict(color='red')))
    fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"].rolling(20).mean(), mode='lines', name='Moving Average', line=dict(color='orange', dash='dash')))
    fig_line.update_layout(title=f"Stock {symbol} Moving Average Envelopes (MMA)",
                    xaxis_title='Date',
                    yaxis_title='Price')
    
    st.plotly_chart(fig_line)