import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from scipy.stats import norm
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_get_stock_returns(ticker:str, start_date:dt.time, end_date:dt.time):
    '''This function retrieves historical stock data and calculates the returns.'''

    def get_stock_returns(ticker, start_date, end_date):
        stock_data = yf.download(ticker,start_date, end_date)
        stock_data = stock_data.reset_index()
        open_prices = stock_data['Open'].tolist()
        open_prices = open_prices[::253]  # Annual data, assuming 253 trading days per year
        df_returns = pd.DataFrame({'Open': open_prices})
        df_returns['Return'] = df_returns['Open'].pct_change()
        return df_returns.dropna()

    # Plots the normal distribution of returns.
    def plot_return_distribution(returns, ticker):
        # Calculate mean and standard deviation
        mean, std = np.mean(returns), np.std(returns)
        
        # Create x values
        x = np.linspace(min(returns), max(returns), 100)
        
        # Calculate probability density function values
        y = norm.pdf(x, mean, std)
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution'))
        fig.update_layout(title=f'Normal Distribution of Returns for {ticker.upper()}',
                        xaxis_title='Returns',
                        yaxis_title='Frequency')
        st.plotly_chart(fig)

    # Estimates the probability of returns falling within specified bounds.
    def estimate_return_probability(returns, lower_bound, higher_bound):
        mean, std = np.mean(returns), np.std(returns)
        prob = round(norm(mean, std).cdf(higher_bound) - norm(mean, std).cdf(lower_bound), 4)
        return prob

    stock_ticker = ticker
    higher_bound, lower_bound = 0.3, 0.2


    # Retrieve and process stock data
    df_returns = get_stock_returns(stock_ticker, start_date, end_date)
    plot_return_distribution(df_returns['Return'], stock_ticker)

    # Estimate probability
    prob = estimate_return_probability(df_returns['Return'], lower_bound, higher_bound)
    st.write(f'The probability of returns falling between {lower_bound} and {higher_bound} for {stock_ticker.upper()} is: {prob}')


def norm_get_stock_returns(ticker, start_date, end_date):
    '''This function retrieves historical stock data and calculates the returns.'''

    def get_stock_returns(ticker, start_date, end_date):
        stock_data = yf.download(ticker,start_date, end_date)
        stock_data = stock_data.reset_index()
        open_prices = stock_data['Open'].tolist()
        open_prices = open_prices[::253]  # Annual data, assuming 253 trading days per year
        df_returns = pd.DataFrame({'Open': open_prices})
        df_returns['Return'] = df_returns['Open'].pct_change()
        return df_returns.dropna()

    # Plots the normal distribution of returns.
    def plot_return_distribution(returns, ticker):
        # Calculate mean and standard deviation
        mean, std = np.mean(returns), np.std(returns)
        
        # Create x values
        x = np.linspace(min(returns), max(returns), 100)
        
        # Calculate probability density function values
        y = norm.pdf(x, mean, std)
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution'))
        fig.update_layout(title=f'Normal Distribution of Returns for {ticker.upper()}',
                        xaxis_title='Returns',
                        yaxis_title='Frequency')
        st.plotly_chart(fig)

    # Estimates the probability of returns falling within specified bounds.
    def estimate_return_probability(returns, lower_bound, higher_bound):
        mean, std = np.mean(returns), np.std(returns)
        prob = round(norm(mean, std).cdf(higher_bound) - norm(mean, std).cdf(lower_bound), 4)
        return prob

    stock_ticker = ticker
    higher_bound, lower_bound = 0.3, 0.2


    # Retrieve and process stock data
    df_returns = get_stock_returns(stock_ticker, start_date, end_date)
    plot_return_distribution(df_returns['Return'], stock_ticker)

    # Estimate probability
    prob = estimate_return_probability(df_returns['Return'], lower_bound, higher_bound)
    st.write(f'The probability of returns falling between {lower_bound} and {higher_bound} for {stock_ticker.upper()} is: {prob}')