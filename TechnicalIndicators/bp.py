import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_bp(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots the Balance of Power (BP) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    df["BOP"] = (df["Adj Close"] - df["Open"]) / (df["High"] - df["Low"])

    # Simple Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=[df["Adj Close"].mean()] * len(df), mode='lines', name='Mean', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Low"], mode='lines', name='Low', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df["High"], mode='lines', name='High', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name='Volume', marker_color='#0079a3', opacity=0.4))

    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # BOP Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["BOP"], name='Balance of Power', marker_color=df["BOP"].apply(lambda x: 'green' if x >= 0 else 'red')))
    fig.update_layout(title="Balance of Power",
                    xaxis_title="Date",
                    yaxis_title="BOP",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with BOP
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df["BOP"], mode='lines', name='Balance of Power', line=dict(color='black')))
    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with BOP",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig) 


    def norm_bp(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots the Balance of Power (BP) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    df["BOP"] = (df["Adj Close"] - df["Open"]) / (df["High"] - df["Low"])

    # Simple Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=[df["Adj Close"].mean()] * len(df), mode='lines', name='Mean', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Low"], mode='lines', name='Low', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df["High"], mode='lines', name='High', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name='Volume', marker_color='#0079a3', opacity=0.4))

    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # BOP Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["BOP"], name='Balance of Power', marker_color=df["BOP"].apply(lambda x: 'green' if x >= 0 else 'red')))
    fig.update_layout(title="Balance of Power",
                    xaxis_title="Date",
                    yaxis_title="BOP",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with BOP
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df["BOP"], mode='lines', name='Balance of Power', line=dict(color='black')))
    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with BOP",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig) 

    def tool_bp(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots the Balance of Power (BP) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date
    # Read data
    df = yf.download(symbol, start, end)

    df["BOP"] = (df["Adj Close"] - df["Open"]) / (df["High"] - df["Low"])

    # Simple Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
    fig.add_trace(go.Scatter(x=df.index, y=[df["Adj Close"].mean()] * len(df), mode='lines', name='Mean', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=df["Low"], mode='lines', name='Low', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df["High"], mode='lines', name='High', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name='Volume', marker_color='#0079a3', opacity=0.4))

    fig.update_layout(title="Stock " + symbol + " Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # BOP Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["BOP"], name='Balance of Power', marker_color=df["BOP"].apply(lambda x: 'green' if x >= 0 else 'red')))
    fig.update_layout(title="Balance of Power",
                    xaxis_title="Date",
                    yaxis_title="BOP",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig)

    # Candlestick Chart with BOP
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df["BOP"], mode='lines', name='Balance of Power', line=dict(color='black')))
    fig.update_layout(title="Stock " + symbol + " Candlestick Chart with BOP",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(x=0, y=1, traceorder="normal"))
    st.plotly_chart(fig) 