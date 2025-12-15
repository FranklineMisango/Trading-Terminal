
import tickers as ti
import yfinance as yf
import pandas as pd
import streamlit as st
import ta_functions as ta
import datetime as dt
from langchain_core.tools import tool




@tool
def tool_green_line_valuation():
    '''This tool allows you to find stocks close to their Green Line value'''
    # Initialize lists for tracking tickers and their green line values
    diff_5 = []
    diff_5_tickers = []
    
    tickers = ti.tickers_sp500()

    # Analyze each ticker
    for ticker in tickers:
        try:
            st.write(f'Analyzing {ticker}:')

            # Load historical data
            df = yf.download(ticker)
            df.index = pd.to_datetime(df.index)
            price = df['Adj Close'][-1]

            # Filter out low volume data
            df = df[df["Volume"] >= 1000]

            # Calculate monthly high
            monthly_high = df.resample('M')['High'].max()

            # Initialize variables for tracking Green Line values
            last_green_line_value = 0
            last_green_line_date = None

            # Identify Green Line values
            for date, high in monthly_high.items():
                if high > last_green_line_value:
                    last_green_line_value = high
                    last_green_line_date = date

            # Check if a green line value has been established
            if last_green_line_value == 0:
                message = f"{ticker} has not formed a green line yet"
            else:
                # Calculate the difference from the current price
                diff = (last_green_line_value - price) / price * 100
                message = f"{ticker}'s last green line value ({round(last_green_line_value, 2)}) is {round(diff, 1)}% different from its current price ({round(price, 2)})"
                if abs(diff) <= 5:
                    diff_5_tickers.append(ticker)
                    diff_5.append(diff)

            print(message)
            print('-' * 100)

        except Exception as e:
            print(f'Error processing {ticker}: {e}')

    # Create and display a DataFrame with tickers close to their green line value
    df = pd.DataFrame({'Company': diff_5_tickers, 'GLV % Difference': diff_5})
    df.sort_values(by='GLV % Difference', inplace=True, key=abs)
    st.write('Watchlist:')
    st.write(df)