import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt
from langchain_core.tools import tool
from scipy import stats
import plotly.express as px
import numpy as np

@tool
def tool_calculate_var(stock, start, end):
    '''This function calculates the Value at Risk (VaR) for a stock using historical bootstrap and variance-covariance methods.'''
# Download data from Yahoo Finance
    df = yf.download(stock, start, end)
    
    # Calculate daily returns
    returns = df['Adj Close'].pct_change().dropna()

    # VaR using historical bootstrap method
    hist_fig = px.histogram(returns, nbins=40, title="Histogram of stock daily returns")
    st.plotly_chart(hist_fig)

    # VaR using variance-covariance method
    tdf, tmean, tsigma = stats.t.fit(returns)
    support = np.linspace(returns.min(), returns.max(), 100)
    pdf = stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf)
    cov_fig = go.Figure(go.Scatter(x=support, y=pdf, mode='lines', line=dict(color='red')))
    cov_fig.update_layout(title="VaR using variance-covariance method")
    st.plotly_chart(cov_fig)

    # Calculate VaR using normal distribution at 95% confidence level
    mean, sigma = returns.mean(), returns.std()
    VaR = stats.norm.ppf(0.05, mean, sigma)
    st.write("VaR using normal distribution at 95% confidence level:", VaR)

def norm_calculate_var(stock, start, end):
# Download data from Yahoo Finance
    df = yf.download(stock, start, end)
    
    # Calculate daily returns
    returns = df['Adj Close'].pct_change().dropna()

    # VaR using historical bootstrap method
    hist_fig = px.histogram(returns, nbins=40, title="Histogram of stock daily returns")
    st.plotly_chart(hist_fig)

    # VaR using variance-covariance method
    tdf, tmean, tsigma = stats.t.fit(returns)
    support = np.linspace(returns.min(), returns.max(), 100)
    pdf = stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf)
    cov_fig = go.Figure(go.Scatter(x=support, y=pdf, mode='lines', line=dict(color='red')))
    cov_fig.update_layout(title="VaR using variance-covariance method")
    st.plotly_chart(cov_fig)

    # Calculate VaR using normal distribution at 95% confidence level
    mean, sigma = returns.mean(), returns.std()
    VaR = stats.norm.ppf(0.05, mean, sigma)
    st.write("VaR using normal distribution at 95% confidence level:", VaR)