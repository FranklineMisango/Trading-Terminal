import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go 
from langchain_core.tools import tool


@tool
def tool_fibonacci_retracement(ticker : str, start_date : dt.time, end_date : dt.time) :
    
    '''This tool allows you to plot Fibonacci retracement levels for a stock'''

    def fetch_stock_data(ticker, start, end):
        return yf.download(ticker, start, end)

    # Calculate Fibonacci retracement levels
    def fibonacci_levels(price_min, price_max):
        diff = price_max - price_min
        return {
            '0%': price_max,
            '23.6%': price_max - 0.236 * diff,
            '38.2%': price_max - 0.382 * diff,
            '61.8%': price_max - 0.618 * diff,
            '100%': price_min
        }

    def plot_fibonacci_retracement(stock_data, fib_levels):
        # Create trace for stock close price
        trace_stock = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close', line=dict(color='black'))

        # Create traces for Fibonacci levels
        fib_traces = []
        for level, price in fib_levels.items():
            fib_trace = go.Scatter(x=stock_data.index, y=[price] * len(stock_data), mode='lines', name=f'{level} level at {price:.2f}', line=dict(color='blue', dash='dash'))
            fib_traces.append(fib_trace)

        # Combine traces
        data = [trace_stock] + fib_traces

        # Define layout
        layout = go.Layout(
            title=f'{ticker} Fibonacci Retracement',
            yaxis=dict(title='Price'),
            xaxis=dict(title='Date'),
            legend=dict(x=0, y=1, traceorder='normal')
        )

        # Create figure
        fig = go.Figure(data=data, layout=layout)

        return fig
                
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    price_min = stock_data['Close'].min()
    price_max = stock_data['Close'].max()
    fib_levels = fibonacci_levels(price_min, price_max)
    fig = plot_fibonacci_retracement(stock_data, fib_levels)
    st.plotly_chart(fig)

# normal
def norm_fibonacci_retracement(ticker, start_date, end_date) : #:Fetch stock data from Yahoo Finance
    def fetch_stock_data(ticker, start, end):
        return yf.download(ticker, start, end)

    # Calculate Fibonacci retracement levels
    def fibonacci_levels(price_min, price_max):
        diff = price_max - price_min
        return {
            '0%': price_max,
            '23.6%': price_max - 0.236 * diff,
            '38.2%': price_max - 0.382 * diff,
            '61.8%': price_max - 0.618 * diff,
            '100%': price_min
        }

    def plot_fibonacci_retracement(stock_data, fib_levels):
        # Create trace for stock close price
        trace_stock = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close', line=dict(color='black'))

        # Create traces for Fibonacci levels
        fib_traces = []
        for level, price in fib_levels.items():
            fib_trace = go.Scatter(x=stock_data.index, y=[price] * len(stock_data), mode='lines', name=f'{level} level at {price:.2f}', line=dict(color='blue', dash='dash'))
            fib_traces.append(fib_trace)

        # Combine traces
        data = [trace_stock] + fib_traces

        # Define layout
        layout = go.Layout(
            title=f'{ticker} Fibonacci Retracement',
            yaxis=dict(title='Price'),
            xaxis=dict(title='Date'),
            legend=dict(x=0, y=1, traceorder='normal')
        )

        # Create figure
        fig = go.Figure(data=data, layout=layout)

        return fig
                
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    price_min = stock_data['Close'].min()
    price_max = stock_data['Close'].max()
    fib_levels = fibonacci_levels(price_min, price_max)
    fig = plot_fibonacci_retracement(stock_data, fib_levels)
    st.plotly_chart(fig)