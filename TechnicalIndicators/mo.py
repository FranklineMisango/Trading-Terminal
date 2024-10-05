import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_mo(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize McClellan Oscillator(MO) for a selected ticker. '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate McClellan Oscillator
    ema_19 = ta.EMA(df["Advancing Issues"] - df["Declining Issues"], timeperiod=19)
    ema_39 = ta.EMA(df["Advancing Issues"] - df["Declining Issues"], timeperiod=39)
    mcclellan_oscillator = ema_19 - ema_39

    # Plot McClellan Oscillator
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=mcclellan_oscillator, mode='lines', name='McClellan Oscillator'))
    fig.update_layout(title=f"McClellan Oscillator for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Value',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_mo(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate McClellan Oscillator
    ema_19 = ta.EMA(df["Advancing Issues"] - df["Declining Issues"], timeperiod=19)
    ema_39 = ta.EMA(df["Advancing Issues"] - df["Declining Issues"], timeperiod=39)
    mcclellan_oscillator = ema_19 - ema_39

    # Plot McClellan Oscillator
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=mcclellan_oscillator, mode='lines', name='McClellan Oscillator'))
    fig.update_layout(title=f"McClellan Oscillator for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Value',
                    template='plotly_dark')
    st.plotly_chart(fig)