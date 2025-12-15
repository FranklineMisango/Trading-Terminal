import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib as plt
from langchain_core.tools import tool
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


@tool
def tool_ols_regression(ticker:str, start_date:dt.time, end_date:dt.time):
    '''This function performs Ordinary Least Squares (OLS) regression analysis between a stock and the S&P 500.'''
    stock = ticker
    # Fetch stock and S&P 500 data
    stock_data = yf.download(stock, start_date, end_date)['Close']
    sp500_data = yf.download('^GSPC',start_date, end_date)['Close']

    # Combine data into a single DataFrame and calculate monthly returns
    combined_data = pd.concat([stock_data, sp500_data], axis=1)
    combined_data.columns = [stock, 'S&P500']
    monthly_returns = combined_data.pct_change().dropna()

    # Define dependent and independent variables for regression
    X = monthly_returns['S&P500']  # Independent variable (S&P500 returns)
    y = monthly_returns[stock]  # Dependent variable (Stock returns)

    # Ordinary Least Squares (OLS) Regression using statsmodels
    X_sm = sm.add_constant(X)  # Adding a constant
    model = sm.OLS(y, X_sm)  # Model definition
    results = model.fit()  # Fit the model
    print(results.summary())  # Print the results summary

    # Linear Regression using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

    # Create matplotlib plot
    plt.figure(figsize=(14, 7))
    plt.scatter(X, y, alpha=0.5, label='Daily Returns')
    plt.plot(X, intercept + slope * X, color='red', label='Regression Line')
    plt.title(f'Regression Analysis: {stock} vs S&P 500')
    plt.xlabel('S&P 500 Daily Returns')
    plt.ylabel(f'{stock} Daily Returns')
    plt.legend()
    plt.grid(True)

    # Convert matplotlib figure to Plotly figure
    plotly_fig = go.Figure()
    plotly_fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Daily Returns'))
    plotly_fig.add_trace(go.Scatter(x=X, y=intercept + slope * pd.Series(X), mode='lines', name='Regression Line'))

    plotly_fig.update_layout(
        title=f'Regression Analysis: {stock} vs S&P 500',
        xaxis_title='S&P 500 Daily Returns',
        yaxis_title=f'{stock} Daily Returns',
        legend=dict(x=0, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'))
    )

    # Display Plotly figure using st.pyplot()
    st.plotly_chart(plotly_fig)

    # Calculate beta and alpha
    beta = slope
    alpha = intercept
    st.write(f'alpha (intercept) = {alpha:.4f}')
    st.write(f'beta (slope) = {beta:.4f}')


def norm_ols_regression(ticker, start_date, end_date):
    '''This function performs Ordinary Least Squares (OLS) regression analysis between a stock and the S&P 500.'''
    stock = ticker
    # Fetch stock and S&P 500 data
    stock_data = yf.download(stock, start_date, end_date)['Close']
    sp500_data = yf.download('^GSPC',start_date, end_date)['Close']

    # Combine data into a single DataFrame and calculate monthly returns
    combined_data = pd.concat([stock_data, sp500_data], axis=1)
    combined_data.columns = [stock, 'S&P500']
    monthly_returns = combined_data.pct_change().dropna()

    # Define dependent and independent variables for regression
    X = monthly_returns['S&P500']  # Independent variable (S&P500 returns)
    y = monthly_returns[stock]  # Dependent variable (Stock returns)

    # Ordinary Least Squares (OLS) Regression using statsmodels
    X_sm = sm.add_constant(X)  # Adding a constant
    model = sm.OLS(y, X_sm)  # Model definition
    results = model.fit()  # Fit the model
    print(results.summary())  # Print the results summary

    # Linear Regression using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

    # Create matplotlib plot
    plt.figure(figsize=(14, 7))
    plt.scatter(X, y, alpha=0.5, label='Daily Returns')
    plt.plot(X, intercept + slope * X, color='red', label='Regression Line')
    plt.title(f'Regression Analysis: {stock} vs S&P 500')
    plt.xlabel('S&P 500 Daily Returns')
    plt.ylabel(f'{stock} Daily Returns')
    plt.legend()
    plt.grid(True)

    # Convert matplotlib figure to Plotly figure
    plotly_fig = go.Figure()
    plotly_fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Daily Returns'))
    plotly_fig.add_trace(go.Scatter(x=X, y=intercept + slope * pd.Series(X), mode='lines', name='Regression Line'))

    plotly_fig.update_layout(
        title=f'Regression Analysis: {stock} vs S&P 500',
        xaxis_title='S&P 500 Daily Returns',
        yaxis_title=f'{stock} Daily Returns',
        legend=dict(x=0, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='black'))
    )

    # Display Plotly figure using st.pyplot()
    st.plotly_chart(plotly_fig)

    # Calculate beta and alpha
    beta = slope
    alpha = intercept
    st.write(f'alpha (intercept) = {alpha:.4f}')
    st.write(f'beta (slope) = {beta:.4f}')