import datetime as dt
import yfinance as yf
import streamlit as st
from langchain_core.tools import tool
from plotly import graph_objs as go
import pandas as pd
import matplotlib.dates as mdates



@tool
def tool_ema(start_date : dt.time, end_date : dt.time, ticker:str):
    ''' This tool plots the candlestick chart of a stock along with the Exponential Moving Average (EMA) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 15
    df["EMA"] = (
        df["Adj Close"].ewm(ignore_na=False, span=n, min_periods=n, adjust=True).mean()
    )

    dfc = df.copy()
    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
    # dfc = dfc.dropna()
    dfc = dfc.reset_index()
    dfc["Date"] = mdates.date2num(dfc["Date"].tolist())
    dfc["Date"] = pd.to_datetime(dfc["Date"])  # Convert Date column to datetime
    dfc["Date"] = dfc["Date"].apply(mdates.date2num)

    # Plotting Moving Average using Plotly
    trace_adj_close = go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close')
    trace_ema = go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA')
    layout_ma = go.Layout(title="Stock Closing Price of " + str(n) + "-Day Exponential Moving Average",
                        xaxis=dict(title="Date"), yaxis=dict(title="Price"))
    fig_ma = go.Figure(data=[trace_adj_close, trace_ema], layout=layout_ma)

    # Plotting Candlestick with EMA using Plotly
    dfc = df.copy()
    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]

    trace_candlestick = go.Candlestick(x=dfc.index,
                                    open=dfc['Open'],
                                    high=dfc['High'],
                                    low=dfc['Low'],
                                    close=dfc['Close'],
                                    name='Candlestick')

    trace_ema = go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA')


    trace_volume = go.Bar(x=dfc.index, y=dfc['Volume'], marker=dict(color=dfc['VolumePositive'].map({True: 'green', False: 'red'})),
                        name='Volume')

    layout_candlestick = go.Layout(title="Stock " + symbol + " Closing Price",
                                xaxis=dict(title="Date", type='date', tickformat='%d-%m-%Y'),
                                yaxis=dict(title="Price"),
                                yaxis2=dict(title="Volume", overlaying='y', side='right'))
    fig_candlestick = go.Figure(data=[trace_candlestick, trace_ema, trace_volume], layout=layout_candlestick)


    # Display Plotly figures in Streamlit
    st.plotly_chart(fig_ma)
    st.warning("Click candlestick, EMA or Volume to tinker with the graph")
    st.plotly_chart(fig_candlestick)



def norm_ema(start_date : dt.time, end_date : dt.time, ticker:str):
    ''' This tool plots the candlestick chart of a stock along with the Exponential Moving Average (EMA) of the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    n = 15
    df["EMA"] = (
        df["Adj Close"].ewm(ignore_na=False, span=n, min_periods=n, adjust=True).mean()
    )

    dfc = df.copy()
    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
    # dfc = dfc.dropna()
    dfc = dfc.reset_index()
    dfc["Date"] = mdates.date2num(dfc["Date"].tolist())
    dfc["Date"] = pd.to_datetime(dfc["Date"])  # Convert Date column to datetime
    dfc["Date"] = dfc["Date"].apply(mdates.date2num)

    # Plotting Moving Average using Plotly
    trace_adj_close = go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close')
    trace_ema = go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA')
    layout_ma = go.Layout(title="Stock Closing Price of " + str(n) + "-Day Exponential Moving Average",
                        xaxis=dict(title="Date"), yaxis=dict(title="Price"))
    fig_ma = go.Figure(data=[trace_adj_close, trace_ema], layout=layout_ma)

    # Plotting Candlestick with EMA using Plotly
    dfc = df.copy()
    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]

    trace_candlestick = go.Candlestick(x=dfc.index,
                                    open=dfc['Open'],
                                    high=dfc['High'],
                                    low=dfc['Low'],
                                    close=dfc['Close'],
                                    name='Candlestick')

    trace_ema = go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA')


    trace_volume = go.Bar(x=dfc.index, y=dfc['Volume'], marker=dict(color=dfc['VolumePositive'].map({True: 'green', False: 'red'})),
                        name='Volume')

    layout_candlestick = go.Layout(title="Stock " + symbol + " Closing Price",
                                xaxis=dict(title="Date", type='date', tickformat='%d-%m-%Y'),
                                yaxis=dict(title="Price"),
                                yaxis2=dict(title="Volume", overlaying='y', side='right'))
    fig_candlestick = go.Figure(data=[trace_candlestick, trace_ema, trace_volume], layout=layout_candlestick)


    # Display Plotly figures in Streamlit
    st.plotly_chart(fig_ma)
    st.warning("Click candlestick, EMA or Volume to tinker with the graph")
    st.plotly_chart(fig_candlestick)                          