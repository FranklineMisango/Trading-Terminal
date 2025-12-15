from datetime import datetime
import datetime as dt
import numpy as np
import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import tickers as ti
from langchain_core.tools import tool

@tool
# Get tickers of S&P 500 stocks
def tool_pca_analysis(start_date : dt.date, end_date : dt.date, ticker: str):
    '''This tool allows you to perform PCA analysis on S&P 500 stocks'''
    PCA_tickers =  ti.tickers_sp500()
    sp500_tickers = yf.download(PCA_tickers, start=start_date, end=end_date)
    tickers = ' '.join(PCA_tickers)

    # Set parameters and retrieve stock tickers
    num_years = 1
    start_date = datetime.date.today() - datetime.timedelta(days=365.25 * num_years)
    end_date = datetime.date.today()

    # Calculate log differences of prices for market index and stocks
    market_prices = yf.download(tickers='^GSPC', start=start_date, end=end_date)['Adj Close']
    market_log_returns = np.log(market_prices).diff()
    stock_prices = yf.download(tickers=tickers, start=start_date, end=end_date)['Adj Close']
    stock_log_returns = np.log(stock_prices).diff()

    # Check if DataFrame is empty
    # Check if DataFrame is empty
    if stock_log_returns.empty:
        st.error("No data found for selected tickers. Please try again with different dates or tickers.")
    else:
        # Plot daily returns of S&P 500 stocks
        st.write("## Daily Returns of S&P 500 Stocks")
        fig = go.Figure()
        for column in stock_log_returns.columns:
            fig.add_trace(go.Scatter(x=stock_log_returns.index, y=stock_log_returns[column], mode='lines', name=column))
        fig.update_layout(title='Daily Returns of S&P 500 Stocks', xaxis_title='Date', yaxis_title='Returns')
        st.plotly_chart(fig)

        # Plot cumulative returns of S&P 500 stocks
        st.write("## Cumulative Returns of S&P 500 Stocks")
        cumulative_returns = stock_log_returns.cumsum().apply(np.exp)
        fig = go.Figure()
        for column in cumulative_returns.columns:
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[column], mode='lines', name=column))
        fig.update_layout(title='Cumulative Returns of S&P 500 Stocks', xaxis_title='Date', yaxis_title='Cumulative Returns')
        st.plotly_chart(fig)

        # Perform PCA on stock returns
        pca = PCA(n_components=1)
        pca.fit(stock_log_returns.fillna(0))
        pc1 = pd.Series(index=stock_log_returns.columns, data=pca.components_[0])

        # Plot the first principal component
        st.write("## First Principal Component of S&P 500 Stocks")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pc1.index, y=pc1.values, mode='lines', name='First Principal Component'))
        fig.update_layout(title='First Principal Component of S&P 500 Stocks', xaxis_title='Stocks', yaxis_title='PC1')
        st.plotly_chart(fig)

        # Calculate weights for PCA portfolio and compare with market index
        weights = abs(pc1) / sum(abs(pc1))
        pca_portfolio_returns = (weights * stock_log_returns).sum(axis=1)
        combined_returns = pd.concat([pca_portfolio_returns, market_log_returns], axis=1)
        combined_returns.columns = ['PCA Portfolio', 'S&P 500']
        cumulative_combined_returns = combined_returns.cumsum().apply(np.exp)

        # Plot PCA portfolio vs S&P 500
        st.write("## PCA Portfolio vs S&P 500")
        fig = go.Figure()
        for column in cumulative_combined_returns.columns:
            fig.add_trace(go.Scatter(x=cumulative_combined_returns.index, y=cumulative_combined_returns[column], mode='lines', name=column))
        fig.update_layout(title='PCA Portfolio vs S&P 500', xaxis_title='Date', yaxis_title='Cumulative Returns')
        st.plotly_chart(fig)

        # Plot stocks with most and least significant PCA weights
        st.write("## Stocks with Most and Least Significant PCA Weights")
        fig = go.Figure(data=[
            go.Bar(x=pc1.nsmallest(10).index, y=pc1.nsmallest(10), name='Most Negative PCA Weights', marker_color='red'),
            go.Bar(x=pc1.nlargest(10).index, y=pc1.nlargest(10), name='Most Positive PCA Weights', marker_color='green')
        ])
        fig.update_layout(title='Stocks with Most and Least Significant PCA Weights', xaxis_title='Stocks', yaxis_title='PCA Weights')
        st.plotly_chart(fig)

    


def norm_pca_analysis(start_date, end_date, ticker):
    PCA_tickers = ti.tickers_sp500()
    sp500_tickers = yf.download(PCA_tickers, start=start_date, end=end_date)
    tickers = ' '.join(PCA_tickers)

    # Set parameters and retrieve stock tickers
    num_years = 1
    start_date = dt.date.today() - dt.timedelta(days=365.25 * num_years)
    end_date = dt.date.today()

    # Calculate log differences of prices for market index and stocks
    market_prices = yf.download(tickers='^GSPC', start=start_date, end=end_date)['Adj Close']
    market_log_returns = np.log(market_prices).diff()
    stock_prices = yf.download(tickers=tickers, start=start_date, end=end_date)['Adj Close']
    stock_log_returns = np.log(stock_prices).diff()

    # Ensure both DataFrames have the same type of DatetimeIndex
    market_log_returns.index = market_log_returns.index.tz_localize(None)
    stock_log_returns.index = stock_log_returns.index.tz_localize(None)

    # Check if DataFrame is empty
    if stock_log_returns.empty:
        st.error("No data found for selected tickers. Please try again with different dates or tickers.")
    else:
        # Plot daily returns of S&P 500 stocks
        st.write("## Daily Returns of S&P 500 Stocks")
        fig = go.Figure()
        for column in stock_log_returns.columns:
            fig.add_trace(go.Scatter(x=stock_log_returns.index, y=stock_log_returns[column], mode='lines', name=column))
        fig.update_layout(title='Daily Returns of S&P 500 Stocks', xaxis_title='Date', yaxis_title='Returns')
        st.plotly_chart(fig)

        # Plot cumulative returns of S&P 500 stocks
        st.write("## Cumulative Returns of S&P 500 Stocks")
        cumulative_returns = stock_log_returns.cumsum().apply(np.exp)
        fig = go.Figure()
        for column in cumulative_returns.columns:
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[column], mode='lines', name=column))
        fig.update_layout(title='Cumulative Returns of S&P 500 Stocks', xaxis_title='Date', yaxis_title='Cumulative Returns')
        st.plotly_chart(fig)


        # Perform PCA on stock returns
        pca = PCA(n_components=1)
        pca.fit(stock_log_returns.fillna(0))
        pc1 = pd.Series(index=stock_log_returns.columns, data=pca.components_[0])

        # Plot the first principal component
        st.write("## First Principal Component of S&P 500 Stocks")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pc1.index, y=pc1.values, mode='lines', name='First Principal Component'))
        fig.update_layout(title='First Principal Component of S&P 500 Stocks', xaxis_title='Stocks', yaxis_title='PC1')
        st.plotly_chart(fig)

        # Calculate weights for PCA portfolio and compare with market index
        weights = abs(pc1) / sum(abs(pc1))
        pca_portfolio_returns = (weights * stock_log_returns).sum(axis=1)
        combined_returns = pd.concat([pca_portfolio_returns, market_log_returns], axis=1)
        combined_returns.columns = ['PCA Portfolio', 'S&P 500']
        cumulative_combined_returns = combined_returns.cumsum().apply(np.exp)

        # Plot PCA portfolio vs S&P 500
        st.write("## PCA Portfolio vs S&P 500")
        fig = go.Figure()
        for column in cumulative_combined_returns.columns:
            fig.add_trace(go.Scatter(x=cumulative_combined_returns.index, y=cumulative_combined_returns[column], mode='lines', name=column))
        fig.update_layout(title='PCA Portfolio vs S&P 500', xaxis_title='Date', yaxis_title='Cumulative Returns')
        st.plotly_chart(fig)

        # Plot stocks with most and least significant PCA weights
        st.write("## Stocks with Most and Least Significant PCA Weights")
        fig = go.Figure(data=[
            go.Bar(x=pc1.nsmallest(10).index, y=pc1.nsmallest(10), name='Most Negative PCA Weights', marker_color='red'),
            go.Bar(x=pc1.nlargest(10).index, y=pc1.nlargest(10), name='Most Positive PCA Weights', marker_color='green')
        ])
        fig.update_layout(title='Stocks with Most and Least Significant PCA Weights', xaxis_title='Stocks', yaxis_title='PCA Weights')
        st.plotly_chart(fig)