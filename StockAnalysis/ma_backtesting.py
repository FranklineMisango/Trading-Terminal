import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from langchain_core.tools import tool
import streamlit as st


@tool
def tool_ma_backtesting(ticker:str, start_date:dt.time, end_date:dt.time):
    '''This function implements a moving average crossover strategy for backtesting.'''
     # Configure the stock symbol, moving average windows, initial capital, and date range
    symbol = ticker
    short_window = 20
    long_window = 50
    initial_capital = 10000  # Starting capital

    # Download stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate short and long moving averages
    stock_data['Short_MA'] = stock_data['Adj Close'].rolling(window=short_window).mean()
    stock_data['Long_MA'] = stock_data['Adj Close'].rolling(window=long_window).mean()

    # Generate trading signals (1 = buy, 0 = hold/sell)
    stock_data['Signal'] = np.where(stock_data['Short_MA'] > stock_data['Long_MA'], 1, 0)
    stock_data['Positions'] = stock_data['Signal'].diff()

    # Calculate daily and cumulative portfolio returns
    stock_data['Daily P&L'] = stock_data['Adj Close'].diff() * stock_data['Signal']
    stock_data['Total P&L'] = stock_data['Daily P&L'].cumsum()
    stock_data['Positions'] *= 100  # Position size for each trade

    # Construct a portfolio to keep track of holdings and cash
    portfolio = pd.DataFrame(index=stock_data.index)
    portfolio['Holdings'] = stock_data['Positions'] * stock_data['Adj Close']       
    portfolio['Cash'] = initial_capital - portfolio['Holdings'].cumsum()
    portfolio['Total'] = portfolio['Cash'] + stock_data['Positions'].cumsum() * stock_data['Adj Close']
    portfolio['Returns'] = portfolio['Total'].pct_change()

    # Create matplotlib plot
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    stock_data[['Short_MA', 'Long_MA', 'Adj Close']].plot(ax=ax1, lw=2.)
    ax1.plot(stock_data.loc[stock_data['Positions'] == 1.0].index, stock_data['Short_MA'][stock_data['Positions'] == 1.0],'^', markersize=10, color='g', label='Buy Signal')
    ax1.plot(stock_data.loc[stock_data['Positions'] == -1.0].index, stock_data['Short_MA'][stock_data['Positions'] == -1.0],'v', markersize=10, color='r', label='Sell Signal')
    ax1.set_title(f'{symbol} Moving Average Crossover Strategy')
    ax1.set_ylabel('Price in $')
    ax1.grid()
    ax1.legend()

    # Convert matplotlib figure to Plotly figure
    plotly_fig = go.Figure()

    # Adding stock data to Plotly figure
    for column in ['Short_MA', 'Long_MA', 'Adj Close']:
        plotly_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[column], mode='lines', name=column))
        buy_signals = stock_data.loc[stock_data['Positions'] == 1.0]
        sell_signals = stock_data.loc[stock_data['Positions'] == -1.0]
        plotly_fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Short_MA'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'))
        plotly_fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Short_MA'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'))

    # Set layout
    plotly_fig.update_layout(
        title=f'{symbol} Moving Average Crossover Strategy',
        xaxis_title='Date',
        yaxis_title='Price in $',
        legend=dict(x=0, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='black')),
        height=600  # Adjust the height as needed
    )

    # Display Plotly figure using st.pyplot()
    st.plotly_chart(plotly_fig)

    # Subplot 1: Moving Average Crossover Strategy
    ax1 = fig.add_subplot(2, 1, 1)
    stock_data[['Short_MA', 'Long_MA', 'Adj Close']].plot(ax=ax1, lw=2.)
    ax1.plot(stock_data.loc[stock_data['Positions'] == 1.0].index, stock_data['Short_MA'][stock_data['Positions'] == 1.0],'^', markersize=10, color='g', label='Buy Signal')
    ax1.plot(stock_data.loc[stock_data['Positions'] == -1.0].index, stock_data['Short_MA'][stock_data['Positions'] == -1.0],'v', markersize=10, color='r', label='Sell Signal')
    ax1.set_title(f'{symbol} Moving Average Crossover Strategy')
    ax1.set_ylabel('Price in $')
    ax1.grid()
    ax1.legend()

    # Subplot 2: Portfolio Value
    ax2 = fig.add_subplot(2, 1, 2)
    portfolio['Total'].plot(ax=ax2, lw=2.)
    ax2.set_ylabel('Portfolio Value in $')
    ax2.set_xlabel('Date')
    ax2.grid()

    plotly_fig = go.Figure()
    line = ax2.get_lines()[0]  # Assuming there's only one line in the plot
    x = line.get_xdata()
    y = line.get_ydata()
    plotly_fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Portfolio Value Fluctuation in USD'))
    plotly_fig.update_layout(
        title=f'Portfolio Value in USD',
        xaxis_title='Date',
        yaxis_title=f'{ticker} Daily Returns',
        legend=dict(x=0, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'))
    )
    
    # Display Plotly figure using st.pyplot()
    st.plotly_chart(plotly_fig)


def norm_ma_backtesting(ticker, start_date, end_date):
    symbol = ticker
    short_window = 20
    long_window = 50
    initial_capital = 10000  # Starting capital

    # Download stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate short and long moving averages
    stock_data['Short_MA'] = stock_data['Adj Close'].rolling(window=short_window).mean()
    stock_data['Long_MA'] = stock_data['Adj Close'].rolling(window=long_window).mean()

    # Generate trading signals (1 = buy, 0 = hold/sell)
    stock_data['Signal'] = np.where(stock_data['Short_MA'] > stock_data['Long_MA'], 1, 0)
    stock_data['Positions'] = stock_data['Signal'].diff()

    # Calculate daily and cumulative portfolio returns
    stock_data['Daily P&L'] = stock_data['Adj Close'].diff() * stock_data['Signal']
    stock_data['Total P&L'] = stock_data['Daily P&L'].cumsum()
    stock_data['Positions'] *= 100  # Position size for each trade

    # Construct a portfolio to keep track of holdings and cash
    portfolio = pd.DataFrame(index=stock_data.index)
    portfolio['Holdings'] = stock_data['Positions'] * stock_data['Adj Close']       
    portfolio['Cash'] = initial_capital - portfolio['Holdings'].cumsum()
    portfolio['Total'] = portfolio['Cash'] + stock_data['Positions'].cumsum() * stock_data['Adj Close']
    portfolio['Returns'] = portfolio['Total'].pct_change()

    # Create matplotlib plot
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    stock_data[['Short_MA', 'Long_MA', 'Adj Close']].plot(ax=ax1, lw=2.)
    ax1.plot(stock_data.loc[stock_data['Positions'] == 1.0].index, stock_data['Short_MA'][stock_data['Positions'] == 1.0],'^', markersize=10, color='g', label='Buy Signal')
    ax1.plot(stock_data.loc[stock_data['Positions'] == -1.0].index, stock_data['Short_MA'][stock_data['Positions'] == -1.0],'v', markersize=10, color='r', label='Sell Signal')
    ax1.set_title(f'{symbol} Moving Average Crossover Strategy')
    ax1.set_ylabel('Price in $')
    ax1.grid()
    ax1.legend()

    # Convert matplotlib figure to Plotly figure
    plotly_fig = go.Figure()

    # Adding stock data to Plotly figure
    for column in ['Short_MA', 'Long_MA', 'Adj Close']:
        plotly_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[column], mode='lines', name=column))
        buy_signals = stock_data.loc[stock_data['Positions'] == 1.0]
        sell_signals = stock_data.loc[stock_data['Positions'] == -1.0]
        plotly_fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Short_MA'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'))
        plotly_fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Short_MA'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'))

    # Set layout
    plotly_fig.update_layout(
        title=f'{symbol} Moving Average Crossover Strategy',
        xaxis_title='Date',
        yaxis_title='Price in $',
        legend=dict(x=0, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='black')),
        height=600  # Adjust the height as needed
    )

    # Display Plotly figure using st.pyplot()
    st.plotly_chart(plotly_fig)

    # Subplot 1: Moving Average Crossover Strategy
    ax1 = fig.add_subplot(2, 1, 1)
    stock_data[['Short_MA', 'Long_MA', 'Adj Close']].plot(ax=ax1, lw=2.)
    ax1.plot(stock_data.loc[stock_data['Positions'] == 1.0].index, stock_data['Short_MA'][stock_data['Positions'] == 1.0],'^', markersize=10, color='g', label='Buy Signal')
    ax1.plot(stock_data.loc[stock_data['Positions'] == -1.0].index, stock_data['Short_MA'][stock_data['Positions'] == -1.0],'v', markersize=10, color='r', label='Sell Signal')
    ax1.set_title(f'{symbol} Moving Average Crossover Strategy')
    ax1.set_ylabel('Price in $')
    ax1.grid()
    ax1.legend()

    # Subplot 2: Portfolio Value
    ax2 = fig.add_subplot(2, 1, 2)
    portfolio['Total'].plot(ax=ax2, lw=2.)
    ax2.set_ylabel('Portfolio Value in $')
    ax2.set_xlabel('Date')
    ax2.grid()

    plotly_fig = go.Figure()
    line = ax2.get_lines()[0]  # Assuming there's only one line in the plot
    x = line.get_xdata()
    y = line.get_ydata()
    plotly_fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Portfolio Value Fluctuation in USD'))
    plotly_fig.update_layout(
        title=f'Portfolio Value in USD',
        xaxis_title='Date',
        yaxis_title=f'{ticker} Daily Returns',
        legend=dict(x=0, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'))
    )
    
    # Display Plotly figure using st.pyplot()
    st.plotly_chart(plotly_fig)