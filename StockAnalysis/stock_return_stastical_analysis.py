import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import numpy as np
from langchain_core.tools import tool
import datetime as dt

@tool
def tool_analyze_stock_returns(symbol:str, start : dt.time, end : dt.time):
    '''This function calculates and displays various statistics for stock returns.'''
    df = yf.download(symbol, start, end)

    # Calculate daily returns
    returns = df['Adj Close'].pct_change().dropna()
    # Calculate and print various statistics
    mean_return = np.mean(returns)
    median_return = np.median(returns)
    mode_return = stats.mode(returns)[0] #work on the mean more [0][0]
    arithmetic_mean_return = returns.mean()
    geometric_mean_return = stats.gmean(returns)
    std_deviation = returns.std()
    harmonic_mean_return = len(returns) / np.sum(1.0/returns)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    jarque_bera_results = stats.jarque_bera(returns)
    is_normal = jarque_bera_results[1] > 0.05

    st.write('Mean of returns:', mean_return)
    st.write('Median of returns:', median_return)
    st.write('Mode of returns:', mode_return)
    st.write('Arithmetic average of returns:', arithmetic_mean_return)
    st.write('Geometric mean of returns:', geometric_mean_return)
    st.write('Standard deviation of returns:', std_deviation)
    st.write('Harmonic mean of returns:', harmonic_mean_return)
    st.write('Skewness:', skewness)
    st.write('Kurtosis:', kurtosis)
    st.write("Jarque-Bera p-value:", jarque_bera_results[1])
    st.write('Are the returns normal?', is_normal)

    # Histogram of returns
    hist_fig = px.histogram(returns, nbins=30, title=f'Histogram of Returns for {symbol.upper()}')
    st.plotly_chart(hist_fig)


def norm_analyze_stock_returns(symbol, start, end):
    '''This function calculates and displays various statistics for stock returns.'''
    df = yf.download(symbol, start, end)

    # Calculate daily returns
    returns = df['Adj Close'].pct_change().dropna()
    # Calculate and print various statistics
    mean_return = np.mean(returns)
    median_return = np.median(returns)
    mode_return = stats.mode(returns)[0] #work on the mean more [0][0]
    arithmetic_mean_return = returns.mean()
    geometric_mean_return = stats.gmean(returns)
    std_deviation = returns.std()
    harmonic_mean_return = len(returns) / np.sum(1.0/returns)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    jarque_bera_results = stats.jarque_bera(returns)
    is_normal = jarque_bera_results[1] > 0.05

    st.write('Mean of returns:', mean_return)
    st.write('Median of returns:', median_return)
    st.write('Mode of returns:', mode_return)
    st.write('Arithmetic average of returns:', arithmetic_mean_return)
    st.write('Geometric mean of returns:', geometric_mean_return)
    st.write('Standard deviation of returns:', std_deviation)
    st.write('Harmonic mean of returns:', harmonic_mean_return)
    st.write('Skewness:', skewness)
    st.write('Kurtosis:', kurtosis)
    st.write("Jarque-Bera p-value:", jarque_bera_results[1])
    st.write('Are the returns normal?', is_normal)

    # Histogram of returns
    hist_fig = px.histogram(returns, nbins=30, title=f'Histogram of Returns for {symbol.upper()}')
    st.plotly_chart(hist_fig)