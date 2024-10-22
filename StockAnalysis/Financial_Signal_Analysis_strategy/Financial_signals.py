import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime as dt
import pandas as pd
from sklearn.mixture import GaussianMixture



def main(years, start_date):
    # Constants for analysis
    index = 'SPY'  # S&P500 as the index for comparison
    num_of_years = years  # Number of years for historical data
    start = start_date

    # Download historical stock prices
    stock_data = yf.download(ticker, start=start)['Adj Close']
    # Plotting stock prices and their distribution
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data.values, mode='lines', name=f'{ticker.upper()} Price'))
    fig1.update_layout(title=f'{ticker.upper()} Price', xaxis_title='Date', yaxis_title='Price')
    fig1.show()

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=stock_data, name=f'{ticker.upper()} Price Distribution'))
    fig2.update_layout(title=f'{ticker.upper()} Price Distribution', xaxis_title='Price', yaxis_title='Frequency')
    fig2.show()

    # Calculating and plotting stock returns
    stock_returns = stock_data.apply(np.log).diff(1)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=stock_returns.index, y=stock_returns.values, mode='lines', name=f'{ticker.upper()} Returns'))
    fig3.update_layout(title=f'{ticker.upper()} Returns', xaxis_title='Date', yaxis_title='Returns')
    fig3.show()

    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=stock_returns, name=f'{ticker.upper()} Returns Distribution'))
    fig4.update_layout(title=f'{ticker.upper()} Returns Distribution', xaxis_title='Returns', yaxis_title='Frequency')
    fig4.show()

    # Rolling statistics for stock returns
    rolling_window = 22
    rolling_mean = stock_returns.rolling(rolling_window).mean()
    rolling_std = stock_returns.rolling(rolling_window).std()
    rolling_skew = stock_returns.rolling(rolling_window).skew()
    rolling_kurtosis = stock_returns.rolling(rolling_window).kurt()

    # Combining rolling statistics into a DataFrame
    signals = pd.concat([rolling_mean, rolling_std, rolling_skew, rolling_kurtosis], axis=1)
    signals.columns = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis']

    fig5 = go.Figure()
    for col in signals.columns:
        fig5.add_trace(go.Scatter(x=signals.index, y=signals[col], mode='lines', name=col))
    fig5.update_layout(title='Rolling Statistics for Stock Returns', xaxis_title='Date', yaxis_title='Value')
    fig5.show()

    # Volatility analysis for S&P500
    index_data = yf.download(index, start=start)['Adj Close']
    index_returns = index_data.apply(np.log).diff(1)
    index_volatility = index_returns.rolling(rolling_window).std()

    # Drop NaN values from index_volatility
    index_volatility.dropna(inplace=True)

    # Gaussian Mixture Model on S&P500 volatility
    gmm_labels = GaussianMixture(2).fit_predict(index_volatility.values.reshape(-1, 1))
    index_data = index_data.reindex(index_volatility.index)

    # Plotting volatility regimes
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=index_data[gmm_labels == 0].index,
                            y=index_data[gmm_labels == 0].values,
                            mode='markers',
                            marker=dict(color='blue'),
                            name='Regime 1'))
    fig6.add_trace(go.Scatter(x=index_data[gmm_labels == 1].index,
                            y=index_data[gmm_labels == 1].values,
                            mode='markers',
                            marker=dict(color='red'),
                            name='Regime 2'))
    fig6.update_layout(title=f'{index} Volatility Regimes (Gaussian Mixture)',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    showlegend=True)
    fig6.show()



if __name__ == "__main__":
    prompt = int(input("What do you want to do : \n1. Backtest a strategy \n2. Run the strategy Live : "))
    if prompt == 1:
        print("You have selected to backtest a strategy")
        tickers = []
        int_ticker = int(input("How many tickers do you want to investigate? ( > 2) : "))
        for i in range(int_ticker):
            ticker = input(f"Enter ticker {i} to investigate : ")
            tickers.append(ticker)
        print(f"The tickers captured are : {tickers}")
        portfolio = input("Enter the portfolio size in USD That you want to start with : ")
        print(f"The portfolio size captured is : {portfolio}")

        start_date = dt.strptime(input("Enter the start date for the analysis (YYYY-MM-DD) : "), "%Y-%m-%d")
        end_date = dt.strptime(input("Enter the end date for the analysis (YYYY-MM-DD) : "), "%Y-%m-%d")
        print(f"The start date captured is : {start_date}")
        print(f"The end date captured is : {end_date}")
        years = end_date.year - start_date.year
        print(f"The number of years captured is : {years}")
        main(tickers, portfolio, start_date, end_date)
    if prompt == 2: 
        print("Still in development")
        pass
