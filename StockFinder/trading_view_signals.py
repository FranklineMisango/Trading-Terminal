from langchain_core.tools import tool
import pandas as pd
import yfinance as yf
import datetime as dt
from time import sleep
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


@tool
def tool_trading_view_signals(start_date : dt.date, end_date : dt.date):
    '''This tool allows you to find stocks with TradingView signals'''
    pd.set_option('display.max_rows', None)
    interval = '1m'

    # Initialize WebDriver for Chrome
    options = Options()
    options.add_argument("--headless")

    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    #driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver = webdriver.Firefox()
    # List of tickers and initialization of lists for data
    tickers = ['SCHB', 'AAPL', 'AMZN', 'TSLA', 'AMD', 'MSFT', 'NFLX']
    signals, sells, buys, neutrals, valid_tickers, prices = [], [], [], [], [], []

    # Today's date for file naming
    today = end_date

    # Process each ticker for trading signals
    for ticker in tickers:
        try:
            driver.get(f"https://s.tradingview.com/embed-widget/technical-analysis/?symbol={ticker}&interval={interval}")
            driver.refresh()
            sleep(2)  # Adjust the sleep time if necessary

            # Extract recommendation and counter values
            #recommendation = driver.find_element(By.CLASS_NAME, "speedometerSignal-pyzN--tL").text
            counters = [element.text for element in driver.find_elements(By.CLASS_NAME, "countersWrapper-uRyD4N_Y")]
            # Append data to lists
            #signals.append(recommendation)
            sells.append(float(counters[0]))
            neutrals.append(float(counters[1]))
            buys.append(float(counters[2]))
            price = yf.download(ticker)['Adj Close'][-1]
            prices.append(price)
            valid_tickers.append(ticker)

            #st.write(f"{ticker} recommendation: {recommendation}")

        except Exception as e:
            st.write(f"Error with {ticker}: {e}")

    # Close WebDriver
    driver.close()

    # Create and print DataFrame
    dataframe = pd.DataFrame({'Tickers': valid_tickers, 'Current Price': prices, 'Signals': signals, 'Buys': buys, 'Sells': sells, 'Neutrals': neutrals}).set_index('Tickers')
    dataframe.sort_values('Signals', ascending=False, inplace=True)
    #dataframe.to_csv(f'{today}_{interval}.csv')
    st.write(dataframe)


def norm_trading_view_signals(start_date, end_date):
    '''This tool allows you to find stocks with TradingView signals'''
    pd.set_option('display.max_rows', None)
    interval = '1m'

    # Initialize WebDriver for Chrome
    options = Options()
    options.add_argument("--headless")

    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    #driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver = webdriver.Firefox()
    # List of tickers and initialization of lists for data
    tickers = ['AAPL', 'AMZN', 'TSLA', 'AMD', 'MSFT', 'NFLX']
    signals, sells, buys, neutrals, valid_tickers, prices = [], [], [], [], [], []

    # Today's date for file naming
    today = end_date

    # Process each ticker for trading signals
    for ticker in tickers:
        try:
            driver.get(f"https://s.tradingview.com/embed-widget/technical-analysis/?symbol={ticker}&interval={interval}")
            driver.refresh()
            sleep(2)  # Adjust the sleep time if necessary

            # Extract recommendation and counter values
            #recommendation = driver.find_element(By.CLASS_NAME, "speedometerSignal-pyzN--tL").text
            counters = [element.text for element in driver.find_elements(By.CLASS_NAME, "countersWrapper-uRyD4N_Y")]
            # Append data to lists
            #signals.append(recommendation)
            sells.append(float(counters[0]))
            neutrals.append(float(counters[1]))
            buys.append(float(counters[2]))
            price = yf.download(ticker)['Adj Close'][-1]
            prices.append(price)
            valid_tickers.append(ticker)

            #st.write(f"{ticker} recommendation: {recommendation}")

        except Exception as e:
            st.write(f"Error with {ticker}: {e}")

    # Close WebDriver
    driver.close()

    # Create and print DataFrame
    dataframe = pd.DataFrame({'Tickers': valid_tickers, 'Current Price': prices, 'Signals': signals, 'Buys': buys, 'Sells': sells, 'Neutrals': neutrals}).set_index('Tickers')
    dataframe.sort_values('Signals', ascending=False, inplace=True)
    #dataframe.to_csv(f'{today}_{interval}.csv')
    st.write(dataframe)