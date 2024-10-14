import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_uo(ticker:str, start_date: dt.time, end_date: dt.time):
    '''This program allows you to view the Ultimate Oscillator(UO) of a ticker over time. '''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Ultimate Oscillator
    df["Prior Close"] = df["Adj Close"].shift()
    df["BP"] = df["Adj Close"] - df[["Low", "Prior Close"]].min(axis=1)
    df["TR"] = df[["High", "Prior Close"]].max(axis=1) - df[["Low", "Prior Close"]].min(axis=1)
    df["Average7"] = df["BP"].rolling(7).sum() / df["TR"].rolling(7).sum()
    df["Average14"] = df["BP"].rolling(14).sum() / df["TR"].rolling(14).sum()
    df["Average28"] = df["BP"].rolling(28).sum() / df["TR"].rolling(28).sum()
    df["UO"] = 100 * (4 * df["Average7"] + 2 * df["Average14"] + df["Average28"]) / (4 + 2 + 1)
    df = df.drop(["Prior Close", "BP", "TR", "Average7", "Average14", "Average28"], axis=1)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df["UO"], mode='lines', name='Ultimate Oscillator'))
    fig.update_layout(title="Stock " + symbol + " Ultimate Oscillator",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with Ultimate Oscillator
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=df["UO"], mode='lines', name='Ultimate Oscillator'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Ultimate Oscillator",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)

def norm_uo(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    # Calculate Ultimate Oscillator
    df["Prior Close"] = df["Adj Close"].shift()
    df["BP"] = df["Adj Close"] - df[["Low", "Prior Close"]].min(axis=1)
    df["TR"] = df[["High", "Prior Close"]].max(axis=1) - df[["Low", "Prior Close"]].min(axis=1)
    df["Average7"] = df["BP"].rolling(7).sum() / df["TR"].rolling(7).sum()
    df["Average14"] = df["BP"].rolling(14).sum() / df["TR"].rolling(14).sum()
    df["Average28"] = df["BP"].rolling(28).sum() / df["TR"].rolling(28).sum()
    df["UO"] = 100 * (4 * df["Average7"] + 2 * df["Average14"] + df["Average28"]) / (4 + 2 + 1)
    df = df.drop(["Prior Close", "BP", "TR", "Average7", "Average14", "Average28"], axis=1)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df["UO"], mode='lines', name='Ultimate Oscillator'))
    fig.update_layout(title="Stock " + symbol + " Ultimate Oscillator",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with Ultimate Oscillator
    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'], name='Candlestick')])
    fig_candle.add_trace(go.Scatter(x=df.index, y=df["UO"], mode='lines', name='Ultimate Oscillator'))
    fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Ultimate Oscillator",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig_candle)
