import streamlit as st
import yfinance as yf
import numpy as np
import datetime as dt
import pandas as pd
from langchain_core.tools import tool

@tool
def tool_perfomance_risk_analysis(ticker:str, start_date:dt.time, end_date:dt.time):
    '''This function calculates the performance and risk metrics of a stock.'''
    index = '^GSPC'
    stock = ticker
    # Fetching data for the stock and S&P 500 index
    df_stock =yf.download(stock,start_date, end_date)
    df_index =yf.download(index,start_date, end_date)

    # Resampling the data to a monthly time series
    df_stock_monthly = df_stock['Adj Close'].resample('M').last()
    df_index_monthly = df_index['Adj Close'].resample('M').last()

    # Calculating monthly returns
    stock_returns = df_stock_monthly.pct_change().dropna()
    index_returns = df_index_monthly.pct_change().dropna()

    # Computing Beta, Alpha, and R-squared
    cov_matrix = np.cov(stock_returns, index_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha = np.mean(stock_returns) - beta * np.mean(index_returns)

    y_pred = alpha + beta * index_returns
    r_squared = 1 - np.sum((y_pred - stock_returns) ** 2) / np.sum((stock_returns - np.mean(stock_returns)) ** 2)

    # Calculating Volatility and Momentum
    volatility = np.std(stock_returns) * np.sqrt(12)  # Annualized volatility
    momentum = np.prod(1 + stock_returns.tail(12)) - 1  # 1-year momentum

    # Printing the results
    st.write(f'Beta: {beta:.4f}')
    st.write(f'Alpha: {alpha:.4f} (annualized)')
    st.write(f'R-squared: {r_squared:.4f}')
    st.write(f'Volatility: {volatility:.4f}')
    st.write(f'1-Year Momentum: {momentum:.4f}')

    # Calculating the average volume over the last 60 days
    average_volume = df_stock['Volume'].tail(60).mean()
    st.write(f'Average Volume (last 60 days): {average_volume:.2f}')   


def norm_perfomance_risk_analysis(ticker, start_date, end_date):
    index = '^GSPC'
    stock = ticker
    # Fetching data for the stock and S&P 500 index
    df_stock =yf.download(stock,start_date, end_date)
    df_index =yf.download(index,start_date, end_date)

    # Resampling the data to a monthly time series
    df_stock_monthly = df_stock['Adj Close'].resample('M').last()
    df_index_monthly = df_index['Adj Close'].resample('M').last()

    # Calculating monthly returns
    stock_returns = df_stock_monthly.pct_change().dropna()
    index_returns = df_index_monthly.pct_change().dropna()

    # Computing Beta, Alpha, and R-squared
    cov_matrix = np.cov(stock_returns, index_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha = np.mean(stock_returns) - beta * np.mean(index_returns)

    y_pred = alpha + beta * index_returns
    r_squared = 1 - np.sum((y_pred - stock_returns) ** 2) / np.sum((stock_returns - np.mean(stock_returns)) ** 2)

    # Calculating Volatility and Momentum
    volatility = np.std(stock_returns) * np.sqrt(12)  # Annualized volatility
    momentum = np.prod(1 + stock_returns.tail(12)) - 1  # 1-year momentum

    # Printing the results
    st.write(f'Beta: {beta:.4f}')
    st.write(f'Alpha: {alpha:.4f} (annualized)')
    st.write(f'R-squared: {r_squared:.4f}')
    st.write(f'Volatility: {volatility:.4f}')
    st.write(f'1-Year Momentum: {momentum:.4f}')

    # Calculating the average volume over the last 60 days
    average_volume = df_stock['Volume'].tail(60).mean()
    st.write(f'Average Volume (last 60 days): {average_volume:.2f}')