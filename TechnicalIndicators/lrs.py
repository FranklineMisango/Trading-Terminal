import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_lrs(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This program allows you to visualize Linear Regression Slope(LRS) for a selected ticker '''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Linear Regression Slope
    avg1 = df["Adj Close"].mean()
    avg2 = df["Adj Close"].mean()
    df["AVGS1_S1"] = avg1 - df["Adj Close"]
    df["AVGS2_S2"] = avg2 - df["Adj Close"]
    df["Average_SQ"] = df["AVGS1_S1"] ** 2
    df["AVG_AVG"] = df["AVGS1_S1"] * df["AVGS2_S2"]
    sum_sq = df["Average_SQ"].sum()
    sum_avg = df["AVG_AVG"].sum()
    slope = sum_avg / sum_sq
    intercept = avg2 - (slope * avg1)
    df["Linear_Regression"] = intercept + slope * (df["Adj Close"])
    df = df.drop(["AVGS1_S1", "AVGS2_S2", "Average_SQ", "AVG_AVG"], axis=1)

    # Plot Linear Regression Slope
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Regression'], mode='lines', name='Linear Regression Slope'))
    fig.update_layout(title=f"Linear Regression Slope for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)

def norm_lrs(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Linear Regression Slope
    avg1 = df["Adj Close"].mean()
    avg2 = df["Adj Close"].mean()
    df["AVGS1_S1"] = avg1 - df["Adj Close"]
    df["AVGS2_S2"] = avg2 - df["Adj Close"]
    df["Average_SQ"] = df["AVGS1_S1"] ** 2
    df["AVG_AVG"] = df["AVGS1_S1"] * df["AVGS2_S2"]
    sum_sq = df["Average_SQ"].sum()
    sum_avg = df["AVG_AVG"].sum()
    slope = sum_avg / sum_sq
    intercept = avg2 - (slope * avg1)
    df["Linear_Regression"] = intercept + slope * (df["Adj Close"])
    df = df.drop(["AVGS1_S1", "AVGS2_S2", "Average_SQ", "AVG_AVG"], axis=1)

    # Plot Linear Regression Slope
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Regression'], mode='lines', name='Linear Regression Slope'))
    fig.update_layout(title=f"Linear Regression Slope for {symbol}",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark')
    st.plotly_chart(fig)
