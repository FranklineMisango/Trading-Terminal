#Library imports
import streamlit as st

#Page config
st.set_page_config(page_title='Trading terminal', page_icon="📈", layout="wide")  


import scipy.optimize as sco
import matplotlib.dates as mpl_dates
from scipy.stats import zscore
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt import risk_models
from pypfopt import expected_returns
from pandas.plotting import register_matplotlib_converters
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from scipy.stats import ttest_ind
from scipy.stats import norm
from ta import add_all_ta_features
from alpaca_trade_api.rest import REST, TimeFrame
import backtrader as bt
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from yahoo_earnings_calendar import YahooEarningsCalendar
from pandas_datareader._utils import RemoteDataError
from matplotlib import dates as mdates
from socket import gaierror
from mplfinance.original_flavor import candlestick_ohlc
import nltk 
nltk.download('vader_lexicon')
from bs4 import BeautifulSoup
import os
from math import sqrt
import quandl as q
from pylab import rcParams
import pylab as pl
import calendar
import yfinance as yf
import pandas as pd
import seaborn as sns
from scipy.stats import gmean
import numpy as np
import plotly.figure_factory as ff
import tickers as ti
import string
import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
from IPython.display import clear_output
import requests
from pandas_datareader import data as pdr
import sys
from websocket import create_connection
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
import ta_functions as ta
import ta as taf
import tickers as ti
from email.message import EmailMessage
import datetime as dt
import time
import statsmodels.api as sm
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm
import warnings
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pylab import rcParams
from fastai.tabular.all import add_datepart
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
import numpy as np
from mplfinance.original_flavor import candlestick_ohlc
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras import Sequential, layers
from keras._tf_keras.keras.preprocessing import sequence
from keras.src.layers import Embedding, LSTM, Dense, Dropout
from keras.src.metrics.metrics_utils import confusion_matrix
from keras.src.utils import to_categorical
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
warnings.filterwarnings("ignore")
from autoscraper import AutoScraper
from lxml import html
import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import math
import mplfinance as mpl
import subprocess



EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
API_FMPCLOUD = st.secrets["API_FMPCLOUD"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
BASE_URL = st.secrets["BASE_URL"]
API_KEY_ALPACA = st.secrets["API_KEY_ALPACA"]
SECRET_KEY_ALPACA = st.secrets["SECRET_KEY_ALPACA"]
ALPACA_CONFIG = st.secrets["ALPACA_CONFIG"]  #TODO 
KRAKEN_CONFIG =     st.secrets["KRAKEN_CONFIG"]



#Al Trading recs
from lumibot.brokers import Alpaca
from lumibot.entities import Asset, Order
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from lumibot.backtesting import YahooDataBacktesting #TODO - Move to strategies
from lumibot.backtesting import CcxtBacktesting
from lumibot.example_strategies.crypto_important_functions import ImportantFunctions
from lumibot.entities import TradingFee

##Terminal config

import time 

def main():
    #Main app
    st.title("Frankline and Co. LP Trading Terminal)
    st.sidebar.info('Welcome to my Algorithmic Trading App. Choose your options below. This application is backed over by 100 mathematically powered algorithms handpicked from the internet and modified for different Trading roles')
    @st.cache_resource
    def correlated_stocks(start_date, end_date, tickers):
        print("Inside correlated_stocks function")
        data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
        returns = np.log(data / data.shift(1))
        correlation = returns.corr()
        return correlation

    # Function to visualize correlations as a heatmap using Plotly
    def visualize_correlation_heatmap(correlation):
        print("Inside visualize_correlation_heatmap function")
        fig = ff.create_annotated_heatmap(
            z=correlation.values,
            x=correlation.columns.tolist(),
            y=correlation.index.tolist(),
            colorscale='Viridis',
            annotation_text=correlation.values.round(2),
            showscale=True
        )
        fig.update_layout(
            title='Correlation Matrix',
            xaxis=dict(title='Tickers', tickangle=90),
            yaxis=dict(title='Tickers'),
            width=1000,
            height=1000
        )
        st.plotly_chart(fig)

    option = st.sidebar.selectbox('Make a choice', ['Find stocks','Stock Data', 'Stock Analysis','Technical Indicators', 'Stock Predictions', 'Portfolio Strategies', "Algorithmic Trading"])
    if option == 'Find stocks':
        options = st.selectbox("Choose a stock finding method:", ["IDB_RS_Rating", "Correlated Stocks", "Finviz_growth_screener", "Fundamental_screener", "RSI_Stock_tickers", "Green_line Valuations", "Minervini_screener", "Pricing Alert Email", "Trading View Signals", "Twitter Screener", "Yahoo Recommendations"])
        if options == "IDB_RS_Rating":
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button('Start Analysis'):
                sp500_tickers = ti.tickers_sp500()
                sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]
                sp500_df = yf.download(sp500_tickers, start=start_date, end=end_date)
                percentage_change_df = sp500_df['Adj Close'].pct_change()
                sp500_df = pd.concat([sp500_df, percentage_change_df.add_suffix('_PercentChange')], axis=1)
                st.write(sp500_df)
        if options == "Correlated Stocks":
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            st.title('Correlation Viewer for Stocks')
            
            # Add more stocks to the portfolio
            sectors = {
                "Technology": ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'INTC', 'CSCO', 'ADBE', 'AVGO', 'PYPL'],
                "Health Care": ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'MDT', 'DHR', 'AMGN', 'LLY'],
                "Financials": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'USB'],
                "Consumer Discretionary": ['AMZN', 'TSLA', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'BKNG', 'LOW', 'CMG'],
                "Communication Services": ['GOOGL', 'META', 'DIS', 'CMCSA', 'NFLX', 'T', 'CHTR', 'DISCA', 'FOXA', 'VZ'],
                "Industrials": ['BA', 'HON', 'UNP', 'UPS', 'CAT', 'LMT', 'MMM', 'GD', 'GE', 'CSX'],
                "Consumer Staples": ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'MO', 'EL', 'KHC', 'MDLZ'],
                "Utilities": ['NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'PEG', 'XEL', 'ED'],
                "Real Estate": ['AMT', 'PLD', 'CCI', 'EQIX', 'WY', 'PSA', 'DLR', 'BXP', 'O', 'SBAC'],
                "Materials": ['LIN', 'APD', 'SHW', 'DD', 'ECL', 'DOW', 'NEM', 'PPG', 'VMC', 'FCX']
            }

            selected_sector = st.selectbox('Select Sector', list(sectors.keys()))
            if st.button("Start Analysis"):
                print("Before fetching tickers")
                tickers = sectors[selected_sector]
                print("After fetching tickers")
                corr_matrix = correlated_stocks(start_date, end_date, tickers)
                print("After computing correlation matrix")
                visualize_correlation_heatmap(corr_matrix)
        if options == "Finviz_growth_screener":
            st.warning("This segment is still under development")
            # Set display options for pandas DataFrame
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            def growth_screener():
                try:
                    frames = []
                    # Loop through different pages of FinViz growth screener
                    for i in range(1, 105, 20):
                        url = (f"https://finviz.com/screener.ashx?v=151&f=fa_epsqoq_o15,fa_epsyoy_pos,fa_epsyoy1_o25,fa_grossmargin_pos,fa_salesqoq_o25,ind_stocksonly,sh_avgvol_o300,sh_insttrans_pos,sh_price_o10,ta_perf_52w50o,ta_sma200_pa,ta_sma50_pa&ft=4&o=-marketcap&r=0{i}")
                        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        webpage = urlopen(req).read()
                        html = soup(webpage, "html.parser")

                        # Extracting stock data from the HTML
                        stocks = pd.read_html(str(html))[-2]
                        stocks = stocks.set_index('Ticker')
                        frames.append(stocks)

                    # Concatenate all DataFrame objects from different pages
                    df = pd.concat(frames)
                    df = df.drop_duplicates()
                    df = df.drop(columns=['No.'])
                    return df
                except Exception as e:
                    print(f"Error occurred: {e}")
                    return pd.DataFrame()

            # Execute the screener and store the result
            df = growth_screener()

            # Display the results
            st.write('\nGrowth Stocks Screener:')
            st.write(df)

            # Extract and print list of tickers
            tickers = df.index
            st.write('\nList of Tickers:')
            st.write(tickers)
        if options == "Fundamental_screener":
            st.success("This portion allows you to sp500 for base overview")
            if st.button("Scan"):

                # Get the API key
                demo = API_FMPCLOUD


                # Define search criteria for the stock screener
                marketcap = str(1000000000)
                url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan={marketcap}&betaMoreThan=1&volumeMoreThan=10000&sector=Technology&exchange=NASDAQ&dividendMoreThan=0&limit=1000&apikey={demo}'

                # Fetch list of companies meeting criteria
                screener = requests.get(url).json()
                st.write(screener)

                # Extract symbols of companies
                companies = [item['symbol'] for item in screener]

                # Initialize dictionary for storing financial ratios
                value_ratios = {}

                # Limit the number of companies for ratio extraction
                max_companies = 30

                # Process financial ratios for each company
                for count, company in enumerate(companies):
                    if count >= max_companies:
                        break

                    try:
                        # Fetch financial and growth ratios
                        fin_url = f'https://financialmodelingprep.com/api/v3/ratios/{company}?apikey={demo}'
                        growth_url = f'https://financialmodelingprep.com/api/v3/financial-growth/{company}?apikey={demo}'

                        fin_ratios = requests.get(fin_url).json()
                        growth_ratios = requests.get(growth_url).json()

                        # Store required ratios
                        ratios = { 'ROE': fin_ratios[0]['returnOnEquity'], 
                                'ROA': fin_ratios[0]['returnOnAssets'], 
                                # Additional ratios can be added here
                                }

                        growth = { 'Revenue_Growth': growth_ratios[0]['revenueGrowth'],
                                'NetIncome_Growth': growth_ratios[0]['netIncomeGrowth'],
                                # Additional growth metrics can be added here
                                }

                        value_ratios[company] = {**ratios, **growth}
                    except Exception as e:
                        st.write(f"Error processing {company}: {e}")

                # Convert to DataFrame and display
                df = pd.DataFrame.from_dict(value_ratios, orient='index')
                st.write(df.head())

                # Define and apply ranking criteria
                criteria = { 'ROE': 1.2, 'ROA': 1.1, 'Debt_Ratio': -1.1, # etc.
                            'Revenue_Growth': 1.25, 'NetIncome_Growth': 1.10 }

                # Normalize and rank companies
                mean_values = df.mean()
                normalized_df = df / mean_values
                normalized_df['ranking'] = sum(normalized_df[col] * weight for col, weight in criteria.items())

                # Print ranked companies
                st.write(normalized_df.sort_values(by=['ranking'], ascending=False))

        if options == "RSI_Stock_tickers":
            st.success("This program allows you to view which tickers are overbrought and which ones are over sold")
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            # Get dates for the past year

            if st.button("Check"):

                # Load list of S&P 500 tickers from tickers module
                tickers = ti.tickers_sp500()

                # Initialize lists for overbought and oversold tickers
                oversold_tickers = []
                overbought_tickers = []

                # Retrieve adjusted close prices for the tickers
                sp500_data = yf.download(tickers, start_date, end_date)['Adj Close']

                # Analyze each ticker for RSI
                for ticker in tickers:
                    try:
                        # Create a new DataFrame for the ticker
                        data = sp500_data[[ticker]].copy()

                        # Calculate the RSI for the ticker
                        data["rsi"] = ta.RSI(data[ticker], timeperiod=14)

                        # Calculate the mean of the last 14 RSI values
                        mean_rsi = data["rsi"].tail(14).mean()

                        # Print the RSI value
                        st.write(f'{ticker} has an RSI value of {round(mean_rsi, 2)}')

                        # Classify the ticker based on its RSI value
                        if mean_rsi <= 30:
                            oversold_tickers.append(ticker)
                        elif mean_rsi >= 70:
                            overbought_tickers.append(ticker)

                    except Exception as e:
                        print(f'Error processing {ticker}: {e}')

                # Output the lists of oversold and overbought tickers
                st.write(f'Oversold tickers: {oversold_tickers}')
                st.write(f'Overbought tickers: {overbought_tickers}')
                
        if options=="Green_line Valuations":
            st.success("This programme analyzes all tickers to help identify the Green Value ones")
            # Retrieve S&P 500 tickers
            tickers = ti.tickers_sp500()
            tickers = [item.replace(".", "-") for item in tickers]

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            # Get dates for the past year

            if st.button("Check"):

                # Initialize lists for tracking tickers and their green line values
                diff_5 = []
                diff_5_tickers = []

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

        if options == "Minervini_screener":
            # Setting up variables
            st.success("This program allows you to view which tickers are overbrought and which ones are over sold")
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            # Get dates for the past year

            if st.button("Check"):

                tickers = ti.tickers_sp500()
                tickers = [ticker.replace(".", "-") for ticker in tickers]
                index_name = '^GSPC'
                start_date = start_date
                end_date = end_date
                exportList = pd.DataFrame(columns=['Stock', "RS_Rating", "50 Day MA", "150 Day Ma", "200 Day MA", "52 Week Low", "52 week High"])

                # Fetching S&P 500 index data
                index_df = yf.download(index_name, start_date, end_date)
                index_df['Percent Change'] = index_df['Adj Close'].pct_change()
                index_return = index_df['Percent Change'].cumprod().iloc[-1]

                # Identifying top performing stocks
                returns_multiples = []
                for ticker in tickers:
                    # Download stock data
                    df = yf.download(ticker, start_date, end_date)
                    #df = pdr.get_data_yahoo(ticker, start_date, end_date)
                    df['Percent Change'] = df['Adj Close'].pct_change()
                    stock_return = df['Percent Change'].cumprod().iloc[-1]
                    returns_multiple = round(stock_return / index_return, 2)
                    returns_multiples.append(returns_multiple)
                    time.sleep(1)

                # Creating a DataFrame for top 30% stocks
                rs_df = pd.DataFrame({'Ticker': tickers, 'Returns_multiple': returns_multiples})
                rs_df['RS_Rating'] = rs_df['Returns_multiple'].rank(pct=True) * 100
                top_stocks = rs_df[rs_df['RS_Rating'] >= rs_df['RS_Rating'].quantile(0.70)]['Ticker']

                # Applying Minervini's criteria
                for stock in top_stocks:
                    try:
                        df = pd.read_csv(f'{stock}.csv', index_col=0)
                        df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
                        df['SMA_150'] = df['Adj Close'].rolling(window=150).mean()
                        df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
                        current_close = df['Adj Close'].iloc[-1]
                        low_52_week = df['Low'].rolling(window=260).min().iloc[-1]
                        high_52_week = df['High'].rolling(window=260).max().iloc[-1]
                        rs_rating = rs_df[rs_df['Ticker'] == stock]['RS_Rating'].iloc[0]

                        # Minervini conditions
                        conditions = [
                            current_close > df['SMA_150'].iloc[-1] > df['SMA_200'].iloc[-1],
                            df['SMA_150'].iloc[-1] > df['SMA_200'].iloc[-20],
                            current_close > df['SMA_50'].iloc[-1],
                            current_close >= 1.3 * low_52_week,
                            current_close >= 0.75 * high_52_week
                        ]

                        if all(conditions):
                            exportList = exportList.append({
                                'Stock': stock, 
                                "RS_Rating": rs_rating,
                                "50 Day MA": df['SMA_50'].iloc[-1], 
                                "150 Day Ma": df['SMA_150'].iloc[-1], 
                                "200 Day MA": df['SMA_200'].iloc[-1], 
                                "52 Week Low": low_52_week, 
                                "52 week High": high_52_week
                            }, ignore_index=True)

                    except Exception as e:
                        print(f"Could not gather data on {stock}: {e}")

                # Exporting the results
                exportList.sort_values(by='RS_Rating', ascending=False, inplace=True)
                st.write(exportList)
                #exportList.to_csv("ScreenOutput.csv")

        if options == "Pricing Alert Email":
            # Email credentials from environment variables
            st.success("This segment allows you to track the pricing of a certain stock to your email if it hits your target price")
            stock = st.text_input("Enter the ticker you want to monitor")
            if stock:
                message = (f"Ticker captured : {stock}")
                st.success(message)
            # Stock and target price settings
            target_price = st.number_input("Enter the target price")
            if target_price:
                message_two = (f"Target price captured at : {target_price}")
                st.success(message_two)
            email_address = st.text_input("Enter your Email address")
            if email_address:
                message_three = (f"Email address captured is {email_address}")
                st.success(message_three)
            # Email setup
            msg = EmailMessage()
            msg['Subject'] = f'Alert on {stock} from Frank & Co. Trading Terminal!'
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email_address # Set the recipient email address (this is mine for testing )

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            # Initialize alerted flag
            alerted = False

            if st.button("Check"):
                # Fetch stock data
                df = pdr.get_data_yahoo(stock, start_date, end_date)
                current_close = df["Adj Close"][-1]
                # Check if the target price is reached
                if current_close > target_price and not alerted:
                    alerted = True
                    message_three = f"Your Monitored {stock} has reached the alert price of {target_price}\nCurrent Price: {current_close}\nContact Our Trading Partners to Buy\n Automated Message by Trading Terminal"
                    st.write(message_three)
                    msg.set_content(message_three)

                    # Send email
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                        smtp.send_message(msg)
                        st.write("Email sent successfully, User Alerted.")
                else:
                    print("No new alerts.")

                # Wait for 60 seconds before the next check
                time.sleep(60)
        if options=="Trading View Signals":
            st.success("This program allows you to view the Trading view signals of a particular ticker")
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Simulate signals"):
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
                        price = pdr.get_data_yahoo(ticker)['Adj Close'][-1]
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
        if options == "Twittter Screener":
            # Set pandas option to display all columns
            pd.set_option('display.max_columns', None)

            # Function to scrape most active stocks from Yahoo Finance
            def scrape_most_active_stocks():
                url = 'https://finance.yahoo.com/most-active/'
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                df = pd.read_html(str(soup), attrs={'class': 'W(100%)'})[0]
                df = df.drop(columns=['52 Week High'])
                return df

            # Scrape and filter most active stocks
            movers = scrape_most_active_stocks()
            movers = movers[movers['% Change'] >= 0]

            # Scrape sentiment data from Sentdex
            def scrape_sentdex():
                res = requests.get('http://www.sentdex.com/financial-analysis/?tf=30d')
                soup = BeautifulSoup(res.text, 'html.parser')
                table = soup.find_all('tr')
                data = {'Symbol': [], 'Sentiment': [], 'Direction': [], 'Mentions': []}

                for ticker in table:
                    ticker_info = ticker.find_all('td')
                    try:
                        data['Symbol'].append(ticker_info[0].get_text())
                        data['Sentiment'].append(ticker_info[3].get_text())
                        data['Mentions'].append(ticker_info[2].get_text())
                        trend = 'up' if ticker_info[4].find('span', {"class": "glyphicon glyphicon-chevron-up"}) else 'down'
                        data['Direction'].append(trend)
                    except:
                        continue
                
                return pd.DataFrame(data)

            sentdex_data = scrape_sentdex()

            # Merge most active stocks with sentiment data
            top_stocks = movers.merge(sentdex_data, on='Symbol', how='left')
            top_stocks.drop(['Market Cap', 'PE Ratio (TTM)'], axis=1, inplace=True)

            # Function to scrape Twitter data from Trade Followers
            def scrape_twitter(url):
                res = requests.get(url)
                soup = BeautifulSoup(res.text, 'html.parser')
                stock_twitter = soup.find_all('tr')
                data = {'Symbol': [], 'Sector': [], 'Score': []}

                for stock in stock_twitter:
                    try:
                        score = stock.find_all("td", {"class": "datalistcolumn"})
                        data['Symbol'].append(score[0].get_text().replace('$', '').strip())
                        data['Sector'].append(score[2].get_text().strip())
                        data['Score'].append(score[4].get_text().strip())
                    except:
                        continue
                
                return pd.DataFrame(data).dropna().drop_duplicates(subset="Symbol").reset_index(drop=True)

            # Scrape Twitter data and merge with previous data
            twitter_data = scrape_twitter("https://www.tradefollowers.com/strength/twitter_strongest.jsp?tf=1m")
            final_list = top_stocks.merge(twitter_data, on='Symbol', how='left')

            # Further scrape and merge Twitter data
            twitter_data2 = scrape_twitter("https://www.tradefollowers.com/active/twitter_active.jsp?tf=1m")
            recommender_list = final_list.merge(twitter_data2, on='Symbol', how='left')
            recommender_list.drop(['Volume', 'Avg Vol (3 month)'], axis=1, inplace=True)

            # Print final recommended list
            st.write('\nFinal Recommended List: ')
            st.write(recommender_list.set_index('Symbol'))

        if options=="Yahoo Recommendations":

            st.success("This segment returns the stocks recommended by ")
            if st.button("Check Recommendations"):
                # Set pandas option to display all columns
                pd.set_option('display.max_columns', None)

                # Function to scrape most active stocks from Yahoo Finance
                def scrape_most_active_stocks():
                    url = 'https://finance.yahoo.com/most-active/'
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    df = pd.read_html(str(soup), attrs={'class': 'W(100%)'})[0]
                    #df = df.drop(columns=['52 Week High'])
                    return df

                # Scrape and filter most active stocks
                movers = scrape_most_active_stocks()
                movers = movers[movers['% Change'] >= 0]

                # Scrape sentiment data from Sentdex
                def scrape_sentdex():
                    res = requests.get('http://www.sentdex.com/financial-analysis/?tf=30d')
                    soup = BeautifulSoup(res.text, 'html.parser')
                    table = soup.find_all('tr')
                    data = {'Symbol': [], 'Sentiment': [], 'Direction': [], 'Mentions': []}

                    for ticker in table:
                        ticker_info = ticker.find_all('td')
                        try:
                            data['Symbol'].append(ticker_info[0].get_text())
                            data['Sentiment'].append(ticker_info[3].get_text())
                            data['Mentions'].append(ticker_info[2].get_text())
                            trend = 'up' if ticker_info[4].find('span', {"class": "glyphicon glyphicon-chevron-up"}) else 'down'
                            data['Direction'].append(trend)
                        except:
                            continue
                    
                    return pd.DataFrame(data)

                sentdex_data = scrape_sentdex()

                # Merge most active stocks with sentiment data
                top_stocks = movers.merge(sentdex_data, on='Symbol', how='left')
                top_stocks.drop(['Market Cap', 'PE Ratio (TTM)'], axis=1, inplace=True)

                # Function to scrape Twitter data from Trade Followers
                def scrape_twitter(url):
                    res = requests.get(url)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    stock_twitter = soup.find_all('tr')
                    data = {'Symbol': [], 'Sector': [], 'Score': []}

                    for stock in stock_twitter:
                        try:
                            score = stock.find_all("td", {"class": "datalistcolumn"})
                            data['Symbol'].append(score[0].get_text().replace('$', '').strip())
                            data['Sector'].append(score[2].get_text().strip())
                            data['Score'].append(score[4].get_text().strip())
                        except:
                            continue
                    
                    return pd.DataFrame(data).dropna().drop_duplicates(subset="Symbol").reset_index(drop=True)

                # Scrape Twitter data and merge with previous data
                twitter_data = scrape_twitter("https://www.tradefollowers.com/strength/twitter_strongest.jsp?tf=1m")
                final_list = top_stocks.merge(twitter_data, on='Symbol', how='left')

                # Further scrape and merge Twitter data
                twitter_data2 = scrape_twitter("https://www.tradefollowers.com/active/twitter_active.jsp?tf=1m")
                recommender_list = final_list.merge(twitter_data2, on='Symbol', how='left')
                recommender_list.drop(['Volume', 'Avg Vol (3 month)'], axis=1, inplace=True)

                # Print final recommended list
                st.write('\nFinal Recommended List: ')
                st.write(recommender_list.set_index('Symbol'))
    
                
    elif option == 'Stock Predictions':

        pred_option = st.selectbox('Make a choice', ['sp500 PCA Analysis','Arima_Time_Series','Stock Probability Analysis','Stock regression Analysis',
                                                    'Stock price predictions','Technical Indicators Clustering', 'LSTM Predictions', 'ETF Graphical Lasso', 'Kmeans Clustering', 
                                                    'ETF Graphical Lasso', 'Kmeans Clustering'])
        if pred_option == "Arima_Time_Series":
            st.success("This segment allows you to Backtest using Arima")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):
                # Fetch data
                data = yf.download(ticker, start_date, end_date)

                # Create closing price plot with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Prices'))
                fig.update_layout(title=f"{ticker} Closing Price", xaxis_title='Dates', yaxis_title='Close Prices')
                # Display the closing price plot using Streamlit
                st.plotly_chart(fig)

                # Test for stationarity
                def test_stationarity(timeseries):
                    # Rolling statistics
                    rolmean = timeseries.rolling(12).mean()
                    rolstd = timeseries.rolling(12).std()
                    
                    fig = go.Figure()
                    # Add traces for the original series, rolling mean, and rolling standard deviation
                    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries.values, mode='lines', name='Original', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=timeseries.index, y=rolmean.values, mode='lines', name='Rolling Mean', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=timeseries.index, y=rolstd.values, mode='lines', name='Rolling Std', line=dict(color='black')))

                    # Update layout
                    fig.update_layout(
                        title='Rolling Mean & Standard Deviation',
                        xaxis_title='Dates',
                        yaxis_title='Values'
                    )

                    # Display the rolling statistics plot using Streamlit
                    st.plotly_chart(fig)
                                        
                    # Dickey-Fuller Test
                    st.write("Results of Dickey Fuller Test")
                    adft = adfuller(timeseries, autolag='AIC')
                    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
                    for key, value in adft[4].items():
                        output[f'critical value ({key})'] = value
                    st.write(output)
                    
                test_stationarity(data['Close'])

                # Decompose the series
                def days_between_dates(dt1, dt2):
                    date_format = "%d/%m/%Y"
                    a = time.mktime(time.strptime(dt1, date_format))
                    b = time.mktime(time.strptime(dt2, date_format))
                    delta = b - a
                    return int(delta / 86400)
            
                
                #difference = days_between_dates(start_date, end_date)
                #st.write(difference)


                def business_days_between_dates(start_date, end_date):
                    business_days = np.busday_count(start_date, end_date)
                    return business_days

                # Example usage:
                business_days = business_days_between_dates(start_date, end_date)
                st.write("Business days between the two dates:", business_days)


                result = seasonal_decompose(data['Close'], model='multiplicative', period=int(business_days/2))
                fig = result.plot()  
                fig.set_size_inches(16, 9)

                st.write(data['Close'])
                # Log transform
                df_log = data['Close']
                #df_log = np.log(data['Close'])
                #st.write(df_log)
                moving_avg = df_log.rolling(12).mean()
                std_dev = df_log.rolling(12).std()

                # Plot moving average
                fig = go.Figure()

                # Add traces for the moving average and standard deviation
                fig.add_trace(go.Scatter(x=std_dev.index, y=std_dev.values, mode='lines', name='Standard Deviation', line=dict(color='black')))
                fig.add_trace(go.Scatter(x=moving_avg.index, y=moving_avg.values, mode='lines', name='Mean', line=dict(color='red')))

                # Update layout
                fig.update_layout(
                    title='Moving Average',
                    xaxis_title='Dates',
                    yaxis_title='Values',
                    legend=dict(x=0, y=1)
                )

                # Display the moving average plot using Streamlit
                st.plotly_chart(fig)

                # Split data into train and test sets
                train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

                # Build and fit ARIMA model
                model = ARIMA(train_data, order=(3, 1, 2))  
                fitted = model.fit()  
                st.write(fitted.summary())

                # Forecast
                
                #fc, se, conf = fitted.forecast(len(test_data), alpha=0.05)
                fc = fitted.forecast(len(test_data), alpha=0.05)

                # Plot forecast
                fc_series = pd.Series(fc, index=test_data.index)
                st.write(fc_series)
                lower_series = pd.Series(index=test_data.index)
                upper_series = pd.Series(index=test_data.index)

                # Create stock price prediction plot with Plotly
                fig = go.Figure()

                # Plot training data
                fig.add_trace(go.Scatter(x=train_data.index, y=train_data.values, mode='lines', name='Training'))

                # Plot actual price
                fig.add_trace(go.Scatter(x=test_data.index, y=test_data.values, mode='lines', name='Actual Price', line=dict(color='blue')))

                # Plot predicted price
                fig.add_trace(go.Scatter(x=test_data.index, y=fc, mode='lines', name='Predicted Price', line=dict(color='orange')))

                # Add shaded region for confidence interval
                fig.add_trace(go.Scatter(x=lower_series.index, y=lower_series.values, mode='lines', fill=None, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                fig.add_trace(go.Scatter(x=lower_series.index, y=upper_series.values, mode='lines', fill='tonexty', line=dict(color='rgba(0,0,0,0)'), name='Confidence Interval'))

                # Update layout
                fig.update_layout(
                    title=f'{ticker} Stock Price Prediction',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    legend=dict(x=0, y=1),
                    margin=dict(l=0, r=0, t=30, b=0),
                )

                # Display the stock price prediction plot using Streamlit
                st.plotly_chart(fig)

                # Performance metrics
                mse = mean_squared_error(test_data, fc)
                mae = mean_absolute_error(test_data, fc)
                rmse = math.sqrt(mse)
                mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
                st.write(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

                # Auto ARIMA
                model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0, test='adf', max_p=3, max_q=3, m=1, d=None, seasonal=False, start_P=0, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
                print(model_autoARIMA.summary())
                model_autoARIMA.plot_diagnostics(figsize=(15,8))
                plt.show()

        if pred_option == "Stock price predictions":
            st.success("This segment allows you to predict the pricing using Linear Regression")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):
                # Set plot size
                rcParams['figure.figsize'] = 20, 10

                # Download historical data for Google stock
                data = yf.download(ticker, start_date, end_date)

                # Moving Average and other calculations
                data['MA50'] = data['Close'].rolling(50).mean()
                data['MA200'] = data['Close'].rolling(200).mean()

                # Create a Plotly figure
                fig = go.Figure()

                # Add traces for close price, 50-day MA, and 200-day MA
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{ticker} Close Price'))
                fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50 Day MA'))
                fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], mode='lines', name='200 Day MA'))

                # Update layout
                fig.update_layout(
                    title=f'{ticker} Stock Prices with Moving Averages',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend=dict(x=0, y=1),
                    margin=dict(l=0, r=0, t=30, b=0),
                )

                # Display the interactive chart using Streamlit
                st.plotly_chart(fig)        

                # Preprocessing for Linear Regression and k-Nearest Neighbors
                data.reset_index(inplace=True)
                data['Date'] = pd.to_datetime(data['Date'])
                data = add_datepart(data, 'Date')
                data.drop('Elapsed', axis=1, inplace=True)  # Remove Elapsed column

                # Scaling data to fit in graph
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(data.drop(['Close'], axis=1))

                # Train-test split
                train_size = int(len(data) * 0.8)
                train_data = data_scaled[:train_size]
                test_data = data_scaled[train_size:]

                # Separate features and target variable
                X_train, y_train = train_data[:, 1:], train_data[:, 0]
                X_test, y_test = test_data[:, 1:], test_data[:, 0]

                # Initialize SimpleImputer to handle missing values
                imputer = SimpleImputer(strategy='mean')

                # Fit and transform the imputer on the training data
                X_train_imputed = imputer.fit_transform(X_train)

                # Transform the testing data using the fitted imputer
                X_test_imputed = imputer.transform(X_test)

                # Initialize and fit Linear Regression model on the imputed data
                lr_model = LinearRegression()
                lr_model.fit(X_train_imputed, y_train)
                lr_score = lr_model.score(X_test_imputed, y_test)
                lr_predictions = lr_model.predict(X_test_imputed)
                lr_printing_score = f"Linear Regression Score: {lr_score}"
                st.write(lr_printing_score)

            # Create a Plotly figure
                fig = go.Figure()

                # Add traces for actual and predicted values
                fig.add_trace(go.Scatter(x=data.index[train_size:], y=y_test, mode='lines', name='Actual'))
                fig.add_trace(go.Scatter(x=data.index[train_size:], y=lr_predictions, mode='lines', name='Linear Regression Predictions'))

                # Update layout
                fig.update_layout(
                    title='Actual vs. Predicted Prices (Linear Regression)',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend=dict(x=0, y=1),
                    margin=dict(l=0, r=0, t=30, b=0),
                )

                # Display the interactive chart using Streamlit
                st.plotly_chart(fig)
        if pred_option == "Stock regression Analysis":

            st.success("This segment allows you to predict the next day price of a stock")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):

                # Set parameters for stock data
                stock = ticker

                # Download historical data for the specified stock
                data = yf.download(stock, start_date, end_date)

                
                # Prepare the features (X) and target (y)
                X = data.drop(['Close'], axis=1)
                y = data['Adj Close']

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

                # Initialize and train a Linear Regression model
                regression_model = LinearRegression()
                regression_model.fit(X_train, y_train)

                # Output the intercept of the model
                intercept = regression_model.intercept_
                st.write(f"The intercept for our model is: {intercept}")

                # Evaluate the model's performance
                score = regression_model.score(X_test, y_test)
                st.write(f"The score for our model is: {score}")

                # Predict the next day's closing price using the latest data
                latest_data = data.tail(1).drop(['Close'], axis=1)
                next_day_price = regression_model.predict(latest_data)[0]
                st.write(f"The predicted price for the next trading day is: {next_day_price}")

        if pred_option == "Stock Probability Analysis":

            st.success("This segment allows you to predict the movement of a stock ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):

                # Download historical data for AMD stock
                data = yf.download(ticker, start_date, end_date)
                def calculate_prereq(values):
                    # Calculate standard deviation and mean
                    std = np.std(values)
                    mean = np.mean(values)
                    return std, mean

                def calculate_distribution(mean, std):
                    # Create normal distribution with given mean and std
                    return stats.norm(mean, std)

                def extrapolate(norm, x):
                    # Probability density function
                    return norm.pdf(x)

                def values_to_norm(dicts):
                    # Convert lists of values to normal distributions
                    for dictionary in dicts:
                        for term in dictionary:
                            std, mean = calculate_prereq(dictionary[term])
                            dictionary[term] = calculate_distribution(mean, std)
                    return dicts

                def compare_possibilities(dicts, x):
                    # Compare normal distributions and return index of higher probability
                    probabilities = []
                    for dictionary in dicts:
                        dict_probs = [extrapolate(dictionary[i], x[i]) for i in range(len(x))]
                        probabilities.append(np.prod(dict_probs))
                    return probabilities.index(max(probabilities))

                # Prepare data for increase and drop scenarios
                drop = {}
                increase = {}
                for day in range(10, len(data) - 1):
                    previous_close = data['Close'][day - 10:day]
                    ratios = [previous_close[i] / previous_close[i - 1] for i in range(1, len(previous_close))]
                    if data['Close'][day + 1] > data['Close'][day]:
                        for i, ratio in enumerate(ratios):
                            increase[i] = increase.get(i, ()) + (ratio,)
                    elif data['Close'][day + 1] < data['Close'][day]:
                        for i, ratio in enumerate(ratios):
                            drop[i] = drop.get(i, ()) + (ratio,)

                # Add new ratios for prediction
                new_close = data['Close'][-11:-1]
                new_ratios = [new_close[i] / new_close[i - 1] for i in range(1, len(new_close))]
                for i, ratio in enumerate(new_ratios):
                    increase[i] = increase.get(i, ()) + (ratio,)

                # Convert ratio lists to normal distributions and make prediction
                dicts = [increase, drop]
                dicts = values_to_norm(dicts)
                prediction = compare_possibilities(dicts, new_ratios)
                st.write("Predicted Movement: ", "Increase" if prediction == 0 else "Drop")
                
        if pred_option == "sp500 PCA Analysis":
            st.write("This segment analyzes the s&p 500 stocks and identifies those with high/low PCA weights")
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):

                # Get tickers of S&P 500 stocks
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
        
        if pred_option == "Technical Indicators Clustering":
            #Fix the segmentation of all tickers on the graph
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):
                # Function to fetch S&P 100 tickers from Wikipedia
                def fetch_sp100_tickers():
                    response = requests.get("https://en.wikipedia.org/wiki/S%26P_100")
                    soup = BeautifulSoup(response.text, "lxml")
                    table = soup.find("table", {"class": "wikitable sortable"})
                    tickers = [row.findAll("td")[0].text.strip() for row in table.findAll("tr")[1:]]
                    return tickers

                # Download historical data for each ticker
                def download_stock_data(tickers):
                    all_data = pd.DataFrame()
                    for ticker in tickers:
                        try:
                            data = yf.download(ticker, start=start_date, end=end_date)
                            data['Symbol'] = ticker
                            all_data = all_data.append(data)
                        except Exception as e:
                            print(f"Error downloading {ticker}: {e}")
                    
                    return all_data

                # Add technical indicators to the data
                def add_technical_indicators(data):
                    # Example implementation for Simple Moving Average (SMA)
                    data['SMA_5'] = data['Close'].rolling(window=5).mean()
                    data['SMA_15'] = data['Close'].rolling(window=15).mean()
                    # Additional technical indicators can be added here
                    return data

                # Get list of S&P 100 tickers
                sp100_tickers = fetch_sp100_tickers()

                # Download stock data
                stock_data = download_stock_data(sp100_tickers)

                # Add technical indicators to the data
                stock_data_with_indicators = add_technical_indicators(stock_data)

                # Clustering based on technical indicators
                # This part of the script would involve applying clustering algorithms
                # such as KMeans or Gaussian Mixture Models to the technical indicators
                # Clustering based on technical indicators
                def perform_clustering(data, n_clusters=10):
                    # Selecting only the columns with technical indicators
                    indicators = data[['SMA_5', 'SMA_15']].copy()  # Add other indicators as needed
                    
                    # Impute missing values
                    imputer = SimpleImputer(strategy='mean')  # You can choose another strategy if needed
                    indicators = imputer.fit_transform(indicators)
                    
                    # KMeans Clustering
                    kmeans = KMeans(n_clusters=n_clusters)
                    kmeans.fit(indicators)
                    data['Cluster'] = kmeans.labels_

                    # Gaussian Mixture Model Clustering
                    gmm = GaussianMixture(n_components=n_clusters)
                    gmm.fit(indicators)
                    data['GMM_Cluster'] = gmm.predict(indicators)

                    return data

                # Perform clustering on the stock data
                clustered_data = perform_clustering(stock_data_with_indicators)
                st.write(clustered_data)

                # Create the figure
                fig = go.Figure()
                for cluster_num in clustered_data['Cluster'].unique():
                    cluster = clustered_data[clustered_data['Cluster'] == cluster_num]
                    for _, row in cluster.iterrows():
                        fig.add_trace(go.Scatter(x=[row['SMA_5']], y=[row['SMA_15']], mode='markers', name=f'{row["Symbol"]} - Cluster {cluster_num}'))

                # Update the layout
                fig.update_layout(title='Stock Data Clusters', xaxis_title='SMA 5', yaxis_title='SMA 15')

                # Display the Plotly chart inline
                st.plotly_chart(fig)
            
                # Analyze and interpret the clusters
                # This step involves understanding the characteristics of each cluster
                # and how they differ from each other based on the stock movement patterns they represent
                # Analyzing Clusters
                def analyze_clusters(data):
                    # Group data by clusters
                    grouped = data.groupby('Cluster')

                    # Analyze clusters
                    cluster_analysis = pd.DataFrame()
                    for name, group in grouped:
                        analysis = {
                            'Cluster': name,
                            'Average_SMA_5': group['SMA_5'].mean(),
                            'Average_SMA_15': group['SMA_15'].mean(),
                            # Add more analysis as needed
                        }
                        cluster_analysis = cluster_analysis.append(analysis, ignore_index=True)
                    
                    return cluster_analysis

                # Perform cluster analysis
                cluster_analysis = analyze_clusters(clustered_data)

                # Display analysis results
                st.write(cluster_analysis)
        if pred_option == "LSTM Predictions":

            st.success("This segment allows you to predict the movement of a stock ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):
            # Prompt user to enter a stock ticker
                # Fetch stock data
                df = yf.download(ticker, start=start_date, end=end_date)
                data = df.filter(['Close'])
                dataset = data.values
                train_data_len = math.ceil(len(dataset) * .8)
                st.write(train_data_len)

                # Scale data
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(dataset)

                # Create training data
                x_train, y_train = [], []
                for i in range(60, len(scaled_data)):
                    x_train.append(scaled_data[i-60:i, 0])
                    y_train.append(scaled_data[i, 0])
                x_train, y_train = np.array(x_train), np.array(y_train)

                st.write("Values of x_train:", x_train)

                # LSTM network
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                    LSTM(50, return_sequences=False),
                    Dense(25),
                    Dense(1)
                ])
                model.compile(optimizer="adam", loss='mean_squared_error')

                # Train the model
                model.fit(x_train, y_train, batch_size=1, epochs=5)

                # Create testing dataset
                test_data = scaled_data[train_data_len - 60:, :]
                x_test = [test_data[i-60:i, 0] for i in range(60, len(test_data))]
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                # Predictions
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)
                rmse = np.sqrt(mean_squared_error(data[train_data_len:].values, predictions))

                #The interactive graph
                fig = make_subplots()

                # Add traces for the training data
                fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))

                # Add traces for the validation data
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Valid'))

                # Add traces for the predictions
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Prediction'))

                # Update the layout
                fig.update_layout(
                    title=f"{stock.upper()} Close Price",
                    xaxis_title='Date',
                    yaxis_title='Close Price (USD)',
                    legend=dict(x=0, y=1, traceorder='normal'),
                    hovermode='x'
                )

                # Show the figure
                st.plotly_chart(fig)

                # Predict next day price
                last_60_days = data[-60:].values
                last_60_days_scaled = scaler.transform(last_60_days)
                X_test_next = np.array([last_60_days_scaled])
                X_test_next = np.reshape(X_test_next, (X_test_next.shape[0], X_test_next.shape[1], 1))
                predicted_price_next_day = scaler.inverse_transform(model.predict(X_test_next))[0][0]
                print(f"The predicted price for the next trading day is: {predicted_price_next_day:.2f}")
        
        if pred_option == "ETF Graphical Lasso":
            st.success("This segment allows you to predict the relations of various ETFS")
            number_of_years = st.text_input("How many years do you want to investigate the ETF correlations(min 10)")
            if number_of_years:
                st.success(f"The number of years captured : {number_of_years}")
            col1, col2 = st.columns([2, 2])
            with col1:
                end_date = st.date_input("End date:")
        
            if st.button("Check"):
                num_years = int(number_of_years)
                start_date = dt.datetime.now() - dt.timedelta(days=num_years * 365.25)
                end_date = dt.datetime.now()

                # ETF symbols and their respective countries
                etfs = {"EWJ": "Japan", "EWZ": "Brazil", "FXI": "China",
                        "EWY": "South Korea", "EWT": "Taiwan", "EWH": "Hong Kong",
                        "EWC": "Canada", "EWG": "Germany", "EWU": "United Kingdom",
                        "EWA": "Australia", "EWW": "Mexico", "EWL": "Switzerland",
                        "EWP": "Spain", "EWQ": "France", "EIDO": "Indonesia",
                        "ERUS": "Russia", "EWS": "Singapore", "EWM": "Malaysia",
                        "EZA": "South Africa", "THD": "Thailand", "ECH": "Chile",
                        "EWI": "Italy", "TUR": "Turkey", "EPOL": "Poland",
                        "EPHE": "Philippines", "EWD": "Sweden", "EWN": "Netherlands",
                        "EPU": "Peru", "ENZL": "New Zealand", "EIS": "Israel",
                        "EWO": "Austria", "EIRL": "Ireland", "EWK": "Belgium"}

                # Retrieve adjusted close prices for ETFs
                symbols = list(etfs.keys())
                etf_data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

                # Convert prices to log returns
                log_returns = np.log1p(etf_data.pct_change()).dropna()

                # Replace NaN values with the mean of each column
                imputer = SimpleImputer(strategy='mean')
                log_returns_normalized = log_returns / log_returns.std(axis=0)
                log_returns_normalized_imputed = imputer.fit_transform(log_returns_normalized)
                edge_model = GraphicalLassoCV(cv=10)
                edge_model.fit(log_returns_normalized_imputed)

                # Ensure the shape of the precision matrix matches the number of ETF symbols
                precision_matrix = edge_model.precision_
                # Compare the number of ETF symbols with the precision matrix dimensions
                if precision_matrix.shape[0] != len(etfs):
                    # If there's a mismatch, print additional information for debugging
                    st.write("Mismatch between the number of ETF symbols and precision matrix dimensions")

                # Create DataFrame with appropriate indices and columns
                precision_df = pd.DataFrame(precision_matrix, index=etfs.keys(), columns=etfs.keys())

                fig = go.Figure(data=go.Heatmap(
                    z=precision_df.values,
                    x=list(etfs.values()),
                    y=list(etfs.values()),
                    colorscale='Viridis'))

                fig.update_layout(title="Precision Matrix Heatmap")
                st.plotly_chart(fig)

                # Prepare data for network graph
                links = precision_df.stack().reset_index()
                links.columns = ['ETF1', 'ETF2', 'Value']
                links_filtered = links[(abs(links['Value']) > 0.17) & (links['ETF1'] != links['ETF2'])]

                # Build and display the network graph
                st.set_option('deprecation.showPyplotGlobalUse', False)
                G = nx.from_pandas_edgelist(links_filtered, 'ETF1', 'ETF2')
                pos = nx.spring_layout(G, k=0.2 * 1 / np.sqrt(len(G.nodes())), iterations=20)
                nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', edge_color='grey')
                st.pyplot()

        if pred_option == "Kmeans Clustering":
            
            st.success("This segment allows you to predict the price of a stock ticker using Kmeans clustering")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):
                # Load stock data from Dow Jones Index
                stocks = ti.tickers_dow()


                # Retrieve adjusted closing prices
                data = yf.download(stocks, start=start_date, end=end_date)['Close']

                # Calculate annual mean returns and variances
                annual_returns = data.pct_change().mean() * 252
                annual_variances = data.pct_change().std() * sqrt(252)

                # Combine returns and variances into a DataFrame
                ret_var = pd.concat([annual_returns, annual_variances], axis=1).dropna()
                ret_var.columns = ["Returns", "Variance"]

                # KMeans clustering
                X = ret_var.values
                sse = [KMeans(n_clusters=k).fit(X).inertia_ for k in range(2, 15)]

                # Convert range to list
                k_values = list(range(2, 15))

                # Plotting the elbow curve using Plotly
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=k_values, y=sse, mode='lines+markers'))
                fig1.update_layout(title="Elbow Curve", xaxis_title="Number of Clusters", yaxis_title="SSE")
                st.plotly_chart(fig1)

                # Apply KMeans with chosen number of clusters
                kmeans = KMeans(n_clusters=5).fit(X)

                # Plotting the clustering result using Plotly
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=kmeans.labels_, colorscale="Rainbow")))
                fig2.update_layout(title="KMeans Clustering of Stocks", xaxis_title="Annual Return", yaxis_title="Annual Variance")
                st.plotly_chart(fig2)

                # Creating a DataFrame with tickers and their cluster labels
                df = pd.DataFrame({'Stock': ret_var.index, 'Cluster': kmeans.labels_})
                df.set_index('Stock', inplace=True)

                st.write(df)

    elif option =='Stock Data':
        pred_option_data = st.selectbox("Choose a Data finding method:", ["Finviz Autoscraper", "High Dividend yield","Dividend History", "Fibonacci Retracement", 
                                                                        "Finviz Home scraper", "Finviz Insider Trading scraper", "Finviz stock scraper", 
                                                                        "Get Dividend calendar", "Green Line Test", "Main Indicators", "Pivots Calculator", 
                                                                        "Email Top Movers", "Stock VWAP", "Data SMS stock", "Stock Earnings", "Trading view Intraday", 
                                                                        "Trading view Recommendations", "Yahoo Finance Intraday Data"])
        if pred_option_data == "Finviz Autoscraper":

            # Fetches financial data for a list of tickers from Finviz using AutoScraper.
            def fetch_finviz_data(tickers, scraper_rule_path):
                # Create an instance of AutoScraper
                scraper = AutoScraper()

                # Load scraper rules from the specified file path
                scraper.load(scraper_rule_path)

                # Iterate over the tickers and scrape data
                for ticker in tickers:
                    url = f'https://finviz.com/quote.ashx?t={ticker}'
                    result = scraper.get_result(url)[0]

                    # Extract attributes and values
                    index = result.index('Index')
                    attributes, values = result[index:], result[:index]

                    # Create a DataFrame and display data
                    df = pd.DataFrame(zip(attributes, values), columns=['Attributes', 'Values'])
                    st.write(f'\n{ticker} Data:')
                    st.write(df.set_index('Attributes'))
            st.warning("This segmenent is under construction. The overall idea is building the finviz scraper rule table to allow us to get what we want specifically")
            num_tickers = st.number_input("Enter the number of stock tickers you want to monitor:", value=1, min_value=1, step=1)
            monitored_tickers = []
            for i in range(num_tickers):
                ticker = st.text_input(f"Enter the company's stock ticker {i+1}:")
                monitored_tickers.append(ticker)
            #scraper_rule_path = '../finviz_table'
            #fetch_finviz_data(tickers, scraper_rule_path)

        if pred_option_data == "Dividend History":

            # Function to convert datetime to string format for URL
            def format_date(date_datetime):
                date_mktime = time.mktime(date_datetime.timetuple())
                return str(int(date_mktime))

            # Function to construct URL for Yahoo Finance dividend history
            def subdomain(symbol, start, end):
                format_url = f"{symbol}/history?period1={start}&period2={end}"
                tail_url = "&interval=div%7Csplit&filter=div&frequency=1d"
                return format_url + tail_url

            # Function to define HTTP headers for request
            def header(subdomain):
                hdrs = {"authority": "finance.yahoo.com", "method": "GET", "path": subdomain,
                        "scheme": "https", "accept": "text/html,application/xhtml+xml",
                        "accept-encoding": "gzip, deflate, br", "accept-language": "en-US,en;q=0.9",
                        "cache-control": "no-cache", "cookie": "cookies", "dnt": "1", "pragma": "no-cache",
                        "sec-fetch-mode": "navigate", "sec-fetch-site": "same-origin", "sec-fetch-user": "?1",
                        "upgrade-insecure-requests": "1", "user-agent": "Mozilla/5.0"}
                return hdrs

            # Function to scrape dividend history page and return DataFrame
            def scrape_page(url, header):
                page = requests.get(url, headers=header)
                element_html = html.fromstring(page.content)
                table = element_html.xpath('//table')[0]
                table_tree = html.tostring(table, method='xml')
                df = pd.read_html(table_tree)
                return df[0]

            # Function to clean dividend data
            def clean_dividends(symbol, dividends):
                dividends = dividends.drop(len(dividends) - 1)
                dividends['Dividend'] = dividends['Dividend'].str.split().str[0].astype(float)
                dividends.name = symbol
                return dividends
            
            st.success("This segment allows you to get the dividends of a particular ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):
                symbol = ticker
                start = start_date - timedelta(days=9125)
                sub = subdomain(symbol, start, end_date)
                hdrs = header(sub)

                base_url = "https://finance.yahoo.com/quote/"
                url = base_url + sub
                dividends_df = scrape_page(url, hdrs)
                st.write(dividends_df)
                dividends = clean_dividends(symbol, dividends_df)

                st.write(dividends)
        
        if pred_option_data == "Fibonacci Retracement":
            st.success("This segment allows you to do Fibonnacci retracement max price of a particular ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Simulate"):
                # Fetch stock data from Yahoo Finance
                def fetch_stock_data(ticker, start, end):
                    return yf.download(ticker, start, end)

                # Calculate Fibonacci retracement levels
                def fibonacci_levels(price_min, price_max):
                    diff = price_max - price_min
                    return {
                        '0%': price_max,
                        '23.6%': price_max - 0.236 * diff,
                        '38.2%': price_max - 0.382 * diff,
                        '61.8%': price_max - 0.618 * diff,
                        '100%': price_min
                    }

                def plot_fibonacci_retracement(stock_data, fib_levels):
                    # Create trace for stock close price
                    trace_stock = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close', line=dict(color='black'))

                    # Create traces for Fibonacci levels
                    fib_traces = []
                    for level, price in fib_levels.items():
                        fib_trace = go.Scatter(x=stock_data.index, y=[price] * len(stock_data), mode='lines', name=f'{level} level at {price:.2f}', line=dict(color='blue', dash='dash'))
                        fib_traces.append(fib_trace)

                    # Combine traces
                    data = [trace_stock] + fib_traces

                    # Define layout
                    layout = go.Layout(
                        title=f'{ticker} Fibonacci Retracement',
                        yaxis=dict(title='Price'),
                        xaxis=dict(title='Date'),
                        legend=dict(x=0, y=1, traceorder='normal')
                    )

                    # Create figure
                    fig = go.Figure(data=data, layout=layout)

                    return fig
                            
                stock_data = fetch_stock_data(ticker, start_date, end_date)
                price_min = stock_data['Close'].min()
                price_max = stock_data['Close'].max()
                fib_levels = fibonacci_levels(price_min, price_max)
                fig = plot_fibonacci_retracement(stock_data, fib_levels)
                st.plotly_chart(fig)

        if pred_option_data == "Finviz Home scraper":
            st.success("This segment allows you to find gainers/losers today and relevant news from Finviz")
            if st.button("Confirm"):

                # Set display options for pandas
                pd.set_option('display.max_colwidth', 60)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)

                # Define URL and request headers
                url = "https://finviz.com/"
                headers = {'User-Agent': 'Mozilla/5.0'}

                # Send request to the website and parse the HTML
                req = Request(url, headers=headers)
                webpage = urlopen(req).read()
                html_content = soup(webpage, "html.parser")

                # Function to scrape a section of the page
                def scrape_section(html, attrs, columns, drop_columns, index_column, idx=0):
                    try:
                        data = pd.read_html(str(html), attrs=attrs)[idx]
                        data.columns = columns
                        data.drop(columns=drop_columns, inplace=True)
                        data.set_index(index_column, inplace=True)
                        return data
                    except Exception as e:
                        return pd.DataFrame({'Error': [str(e)]})

                # Scrape different sections of the page and print the results
                st.success('\nPositive Stocks: ')
                st.write(scrape_section(html_content, {'class': 'styled-table-new is-rounded is-condensed is-tabular-nums'}, 
                                    ['Ticker', 'Last', 'Change', 'Volume', '4', 'Signal'], ['4'], 'Ticker'))

                st.error('\nNegative Stocks: ')
                st.write(scrape_section(html_content, {'class': 'styled-table-new is-rounded is-condensed is-tabular-nums'}, 
                                    ['Ticker', 'Last', 'Change', 'Volume', '4', 'Signal'], ['4'], 'Ticker', idx=1))

                st.write('\nHeadlines: ')
                st.write(scrape_section(html_content, {'class': 'styled-table-new is-rounded is-condensed hp_news-table table-fixed'}, 
                                    ['0', 'Time', 'Headlines'], ['0'], 'Time'))

                st.write('\nUpcoming Releases: ')
                st.write(scrape_section(html_content, {'class': 'calendar_table'}, 
                                    ['Date', 'Time', '2', 'Release', 'Impact', 'For', 'Actual', 'Expected', 'Prior'], ['2'], 'Date'))

                st.write('\nUpcoming Earnings: ')
                st.write(scrape_section(html_content, {'class': 'styled-table-new is-rounded is-condensed hp_table-earnings'}, 
                                    ['Date', 'Ticker', 'Ticker', 'Ticker', 'Ticker', 'Ticker', 'Ticker', 'Ticker', 'Ticker'], [], 'Ticker'))

                st.write('\nFutures: ')
                st.write(scrape_section(html_content, {'class': 'styled-table-new is-rounded is-condensed is-tabular-nums'}, 
                                    ['Index', 'Last', 'Change', 'Change (%)'], [], 'Index', idx=2))

                st.write('\nForex Rates: ')
                st.write(scrape_section(html_content, {'class': 'styled-table-new is-rounded is-condensed is-tabular-nums'}, 
                                    ['Index', 'Last', 'Change', 'Change (%)'], [], 'Index', idx=3))

        if pred_option_data == "Finviz Insider Trading scraper":
            st.success("This segment allows you to check the latest insider trades of the market and relevant parties")
            # Set display options for pandas
            if st.button("Check"):
                pd.set_option('display.max_colwidth', 60)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)

                # Function to fetch HTML content from URL
                def fetch_html(url):
                    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    webpage = urlopen(req).read()
                    return soup(webpage, "html.parser")

                # Function to extract insider trades table
                def extract_insider_trades(html_soup):
                    trades = pd.read_html(str(html_soup), attrs={'class': 'styled-table-new is-rounded is-condensed mt-2 w-full'})[0]
                    return trades

                # Process the insider trades data
                def process_insider_trades(trades):
                    # Rename columns and sort by date
                    trades.columns = ['Ticker', 'Owner', 'Relationship', 'Date', 'Transaction', 'Cost', '#Shares', 'Value ($)', '#Shares Total', 'SEC Form 4']
                    trades.sort_values('Date', ascending=False, inplace=True)

                    # Set the date column as index and drop unnecessary rows
                    trades.set_index('Date', inplace=True)

                    return trades

                # Main function to scrape insider trades
                def scrape_insider_trades():
                    try:
                        url = "https://finviz.com/insidertrading.ashx"
                        html_soup = fetch_html(url)
                        trades = extract_insider_trades(html_soup)
                        processed_trades = process_insider_trades(trades)
                        return processed_trades
                    except Exception as e:
                        return e

                # Call the function and print the result
                st.write('\nInsider Trades:')
                st.write(scrape_insider_trades())

        if pred_option_data == "Finviz stock scraper":

            st.success("This segment allows to analyze overview a ticker from finviz")
            stock = st.text_input("Please type the stock ticker")
            if stock:
                st.success(f"Stock ticker captured : {stock}")
            if st.button("check"):
                # Set display options for pandas dataframes
                pd.set_option('display.max_colwidth', 25)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)


                # Set up scraper
                url = f"https://finviz.com/quote.ashx?t={stock.strip().upper()}&p=d"
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                webpage = urlopen(req).read()
                html_content = soup(webpage, "html.parser")

                # Function to get fundamental ratios
                def get_fundamentals():
                    try:
                        # Get data from finviz and convert to pandas dataframe
                        df = pd.read_html(str(html_content), attrs = {"class":"js-snapshot-table snapshot-table2 screener_snapshot-table-body"})[0]

                        # Resetting the combined columns lists 
                        combined_column1 = []
                        combined_column2 = []

                        # Looping through the DataFrame to combine data with adjustment for odd number of columns
                        for i in range(0, len(df.columns), 2):
                            combined_column1.extend(df.iloc[:, i].tolist())
                            # Check if the next column exists before adding it, otherwise add None
                            if i + 1 < len(df.columns):
                                combined_column2.extend(df.iloc[:, i + 1].tolist())
                            else:
                                combined_column2.extend([None] * len(df))  # Add None for missing values

                        # Creating a new DataFrame with the combined columns
                        combined_df = pd.DataFrame({'Attributes': combined_column1, 'Values': combined_column2})
                        combined_df = combined_df.set_index('Attributes')
                        return combined_df
                    except Exception as e:
                        return e

                # Function to get recent news articles
                def get_news():
                    try:
                        news = pd.read_html(str(html_content), attrs={"class": "fullview-news-outer"})[0]
                        news.columns = ['DateTime', 'Headline']
                        news.set_index('DateTime', inplace=True)
                        return news
                    except Exception as e:
                        return e

                # Function to get recent insider trades
                def get_insider():
                    try:
                        insider = pd.read_html(str(html_content), attrs={"class": "body-table"})[0]
                        insider.set_index('Date', inplace=True)
                        return insider
                    except Exception as e:
                        return e

                # Function to get analyst price targets
                def get_price_targets():
                    try:
                        targets = pd.read_html(str(html_content), attrs={"class": "js-table-ratings"})[0]
                        targets.set_index('Date', inplace=True)
                        return targets
                    except Exception as e:
                        return e

                # Print out the resulting dataframes for each category
                st.write('Fundamental Ratios:')
                st.write(get_fundamentals())

                st.write('\nRecent News:')
                st.write(get_news())

                st.warning('\nRecent Insider Trades:')
                st.write(get_insider())

                st.write('\nAnalyst Price Targets:')
                st.write(get_price_targets())

        if pred_option_data == "Get Dividend calendar":
            st.success("This segment allows you to observe the Dividend companies of some companies")     

            month = st.date_input("Enter the Month to test")
            if month:
                # Convert the selected date to a string in the format "%Y-%m-%d"
                month_str = month.strftime("%Y-%m-%d")
                # Split the string into year and month
                year, month, _ = month_str.split('-')
                st.success(f"Month captured: Year - {year}, Month - {month}")

            if st.button("Check"):
                #Set pandas option to display all columns
                pd.set_option('display.max_columns', None)

                class DividendCalendar:
                    def __init__(self, year, month):
                        # Initialize with the year and month for the dividend calendar
                        self.year = year
                        self.month = month
                        self.url = 'https://api.nasdaq.com/api/calendar/dividends'
                        self.hdrs = {
                            'Accept': 'text/plain, */*',
                            'DNT': "1",
                            'Origin': 'https://www.nasdaq.com/',
                            'Sec-Fetch-Mode': 'cors',
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0)'
                        }
                        self.calendars = []  # Store all calendar DataFrames

                    def date_str(self, day):
                        # Convert a day number into a formatted date string
                        return dt.date(self.year, self.month, day).strftime('%Y-%m-%d')

                    def scraper(self, date_str):
                        # Scrape dividend data from NASDAQ API for a given date
                        response = requests.get(self.url, headers=self.hdrs, params={'date': date_str})
                        return response.json()

                    def dict_to_df(self, dictionary):
                        # Convert the JSON data from the API into a pandas DataFrame
                        rows = dictionary.get('data').get('calendar').get('rows', [])
                        calendar_df = pd.DataFrame(rows)
                        self.calendars.append(calendar_df)
                        return calendar_df

                    def calendar(self, day):
                        # Fetch dividend data for a specific day and convert it to DataFrame
                        date_str = self.date_str(day)
                        dictionary = self.scraper(date_str)
                        return self.dict_to_df(dictionary)

                def get_dividends(year, month):
                    try:
                        # Create an instance of DividendCalendar for the given year and month
                        dc = DividendCalendar(year, month)
                        days_in_month = calendar.monthrange(year, month)[1]

                        # Iterate through each day of the month and scrape dividend data
                        for day in range(1, days_in_month + 1):
                            dc.calendar(day)

                        # Combine all the scraped data into a single DataFrame
                        concat_df = pd.concat(dc.calendars).dropna(how='any')
                        concat_df = concat_df.set_index('companyName').reset_index()
                        concat_df = concat_df.drop(columns=['announcement_Date'])
                        concat_df.columns = ['Company Name', 'Ticker', 'Dividend Date', 'Payment Date', 
                                            'Record Date', 'Dividend Rate', 'Annual Rate']
                        concat_df = concat_df.sort_values(['Annual Rate', 'Dividend Date'], ascending=[False, False])
                        concat_df = concat_df.drop_duplicates()
                        return concat_df
                    except Exception as e:
                        return e
                    

                st.write(get_dividends(int(year),int(month)))
            
        if pred_option_data == "Green Line Test":

            st.success("This segment allows you to see if a ticker hit a greenline")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            
            min_date = datetime(1980, 1, 1)

            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            # Button to trigger the simulation
            if st.button("Simulate"):
                st.success("Simulation started...")

                # Set the start and end dates for historical data retrieval
                start = dt.datetime.combine(start_date, dt.datetime.min.time())
                end = dt.datetime.combine(end_date, dt.datetime.min.time())

                # Fetch historical stock data
                df = yf.download(ticker, start, end)
                price = df['Adj Close'][-1]

                # Filter out days with very low trading volume
                df.drop(df[df["Volume"] < 1000].index, inplace=True)

                # Get the monthly maximum of the 'High' column
                df_month = df.groupby(pd.Grouper(freq="M"))["High"].max()

                # Initialize variables for the green line analysis
                glDate, lastGLV, currentDate, currentGLV, counter = 0, 0, "", 0, 0

                # Loop through monthly highs to determine the most recent green line value
                for index, value in df_month.items():
                    if value > currentGLV:
                        currentGLV = value
                        currentDate = index
                        counter = 0
                    if value < currentGLV:
                        counter += 1
                        if counter == 3 and ((index.month != end_date.month) or (index.year != end_date.year)):
                            if currentGLV != lastGLV:
                                print(currentGLV)
                            glDate = currentDate
                            lastGLV = currentGLV
                            counter = 0

                # Determine the message to display based on green line value and current price
                if lastGLV == 0:
                    message = f"{ticker} has not formed a green line yet"
                else:
                    diff = price/lastGLV - 1
                    diff = round(diff * 100, 3)
                    if lastGLV > 1.15 * price:
                        message = f"\n{ticker.upper()}'s current price ({round(price, 2)}) is {diff}% away from its last green line value ({round(lastGLV, 2)})"
                    else:
                        if lastGLV < 1.05 * price:
                            st.write(f"\n{ticker.upper()}'s last green line value ({round(lastGLV, 2)}) is {diff}% greater than its current price ({round(price, 2)})")
                            message = ("Last Green Line: "+str(round(lastGLV, 2))+" on "+str(glDate.strftime('%Y-%m-%d')))
                            
                            # Plot interactive graph with Plotly
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df.index[-120:], y=df['Close'].tail(120), mode='lines', name='Close Price'))
                            fig.add_hline(y=lastGLV, line_dash="dash", line_color="green", name='Last Green Line')
                            fig.update_layout(title=f"{ticker.upper()}'s Close Price Green Line Test", xaxis_title='Dates', yaxis_title='Close Price')
                            st.plotly_chart(fig)
                        else:
                            message = ("Last Green Line: "+str(round(lastGLV, 2))+" on "+str(glDate.strftime('%Y-%m-%d')))
                            # Plot interactive graph with Plotly
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
                            fig.add_hline(y=lastGLV, line_dash="dash", line_color="green", name='Last Green Line')
                            fig.update_layout(title=f"{ticker.upper()}'s Close Price Green Line Test", xaxis_title='Dates', yaxis_title='Close Price')
                            st.plotly_chart(fig)
                
                st.write(message)


        if pred_option_data == "High Dividend yield":
            st.success("This program allows you to simulate the Dividends given by a company")
            if st.button("Check"):

                # Create an instance of the financial model prep class for API access
                demo = API_FMPCLOUD

                # Read ticker symbols from a saved pickle file
                symbols = ti.tickers_sp500()

                # Initialize a dictionary to store dividend yield data for each company
                yield_dict = {}

                # Loop through each ticker symbol in the list
                for company in symbols:
                    try:
                        # Fetch company data from FinancialModelingPrep API using the ticker symbol
                        companydata = requests.get(f'https://fmpcloud.io/api/v3/profile/{company}?apikey={demo}')
                        st.write(companydata)
                        companydata = companydata.json()  # Convert response to JSON format
                        
                        # Extract relevant data for dividend calculation
                        latest_Annual_Dividend = companydata[0]['lastDiv']
                        price = companydata[0]['price']
                        market_Capitalization = companydata[0]['mktCap']
                        name = companydata[0]['companyName']
                        exchange = companydata[0]['exchange']

                        # Calculate the dividend yield
                        dividend_yield = latest_Annual_Dividend / price

                        # Store the extracted data in the yield_dict dictionary
                        yield_dict[company] = {
                            'dividend_yield': dividend_yield,
                            'latest_price': price,
                            'latest_dividend': latest_Annual_Dividend,
                            'market_cap': market_Capitalization / 1000000,  # Convert market cap to millions
                            'company_name': name,
                            'exchange': exchange
                        }

                    except Exception as e:
                        # Skip to the next ticker if there's an error with the current one
                        print(f"Error processing {company}: {e}")
                        continue

                # Convert the yield_dict dictionary to a pandas DataFrame
                yield_dataframe = pd.DataFrame.from_dict(yield_dict, orient='index')

                # Sort the DataFrame by dividend yield in descending order
                yield_dataframe = yield_dataframe.sort_values('dividend_yield', ascending=False)

                # Display the sorted DataFrame
                st.write(yield_dataframe)

        if pred_option_data == "Main Indicators":
            st.success("This segment allows you to see if a ticker hit a greenline")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            
            min_date = datetime(1980, 1, 1)

            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            # Button to trigger the simulation
            if st.button("Simulate"):

                symbol, start, end = ticker, start_date, end_date

                # Convert string dates to datetime objects
                start = pd.to_datetime(start)
                end = pd.to_datetime(end)

                # Download stock data from Yahoo Finance
                data = pdr.get_data_yahoo(symbol, start, end)

                # Display Adjusted Close Price
                st.header(f"Adjusted Close Price\n {symbol}")
                st.line_chart(data["Adj Close"])

                # Calculate and display SMA and EMA
                data["SMA"] = ta.SMA(data["Adj Close"], timeperiod=20)
                data["EMA"] = ta.EMA(data["Adj Close"], timeperiod=20)
                st.header(f"Simple Moving Average vs. Exponential Moving Average\n {symbol}")
                st.line_chart(data[["Adj Close", "SMA", "EMA"]])

                # Calculate and display Bollinger Bands
                data["upper_band"], data["middle_band"], data["lower_band"] = ta.BBANDS(data["Adj Close"], timeperiod=20)
                st.header(f"Bollinger Bands\n {symbol}")
                st.line_chart(data[["Adj Close", "upper_band", "middle_band", "lower_band"]])

                # Calculate and display MACD
                data["macd"], data["macdsignal"], data["macdhist"] = ta.MACD(data["Adj Close"], fastperiod=12, slowperiod=26, signalperiod=9)
                st.header(f"Moving Average Convergence Divergence\n {symbol}")
                st.line_chart(data[["macd", "macdsignal"]])

                # Calculate and display CCI
                data["CCI"] = ta.CCI(data["High"], data["Low"], data["Close"], timeperiod=14)
                st.header(f"Commodity Channel Index\n {symbol}")
                st.line_chart(data["CCI"])

                # Calculate and display RSI
                data["RSI"] = ta.RSI(data["Adj Close"], timeperiod=14)
                st.header(f"Relative Strength Index\n {symbol}")
                st.line_chart(data["RSI"])

                # Calculate and display OBV
                data["OBV"] = ta.OBV(data["Adj Close"], data["Volume"]) / 10**6
                st.header(f"On Balance Volume\n {symbol}")
                st.line_chart(data["OBV"])

        if pred_option_data == "Pivots Calculator":
            st.write("This part calculates the pivot shifts within one day")
            # Prompt user for stock ticker input
            stock = st.text_input('Enter a ticker: ')
            interval = st.text_input('Enter an Interval : default for 1d')
            
            if st.button("Check") :
                # Fetch historical data for the specified stock using yfinance
                ticker = yf.Ticker(stock)
                df = ticker.history(interval)

                # Extract data for the last trading day and remove unnecessary columns
                last_day = df.tail(1).copy().drop(columns=['Dividends', 'Stock Splits'])

                # Calculate pivot points and support/resistance levels
                # Pivot point formula: (High + Low + Close) / 3
                last_day['Pivot'] = (last_day['High'] + last_day['Low'] + last_day['Close']) / 3
                last_day['R1'] = 2 * last_day['Pivot'] - last_day['Low']  # Resistance 1
                last_day['S1'] = 2 * last_day['Pivot'] - last_day['High']  # Support 1
                last_day['R2'] = last_day['Pivot'] + (last_day['High'] - last_day['Low'])  # Resistance 2
                last_day['S2'] = last_day['Pivot'] - (last_day['High'] - last_day['Low'])  # Support 2
                last_day['R3'] = last_day['Pivot'] + 2 * (last_day['High'] - last_day['Low'])  # Resistance 3
                last_day['S3'] = last_day['Pivot'] - 2 * (last_day['High'] - last_day['Low'])  # Support 3

                # Display calculated pivot points and support/resistance levels for the last trading day
                st.write(last_day)

                # Fetch intraday data for the specified stock
                data = yf.download(tickers=stock, period="1d", interval="1m")

                # Extract 'Close' prices from the intraday data for plotting
                df = data['Close']

                # Create Plotly figure
                fig = go.Figure()

                # Plot intraday data
                fig.add_trace(go.Scatter(x=df.index, y=df.values, mode='lines', name='Price'))

                # Plot support and resistance levels
                for level, color in zip(['R1', 'S1', 'R2', 'S2', 'R3', 'S3'], ['blue', 'blue', 'green', 'green', 'red', 'red']):
                    fig.add_trace(go.Scatter(x=df.index, y=[last_day[level].iloc[0]] * len(df.index),
                                            mode='lines', name=level, line=dict(color=color, dash='dash')))

                # Customize layout
                fig.update_layout(title=f"{stock.upper()} - {dt.date.today()}",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                showlegend=True)

                # Display Plotly figure
                st.plotly_chart(fig)
                            
        if pred_option_data == "Email Top Movers":

            st.success("This segment allows you to get updates on Top stock movers")
            email_address = st.text_input("Enter your Email address")
            if email_address:
                message_three = (f"Email address captured is {email_address}")
                st.success(message_three)

            if st.button("Check"):
                # Function to scrape top winner stocks from Yahoo Finance
                def scrape_top_winners():
                    url = 'https://finance.yahoo.com/gainers/'
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Parse the HTML content and extract data into a DataFrame
                    df = pd.read_html(str(soup), attrs={'class': 'W(100%)'})[0]
                    st.write(df)
                    # Drop irrelevant columns for simplicity
                    df = df.drop(columns=['52 Week Range'])
                    return df

                # Retrieve top gainers data and filter based on a certain percentage change
                df = scrape_top_winners()
                #df_filtered = df[df['% Change'] >= 5]

                # Save the filtered DataFrame to a CSV file
                today = dt.date.today()
                file_name = "Top Gainers " + str(today) + ".csv"
                df.to_csv(file_name)

                # Function to send an email with the top gainers CSV file as an attachment
                def send_email():
                    # Email sender and recipient details (to be filled)
                    email_sender = EMAIL_ADDRESS  # Sender's email
                    email_password = EMAIL_PASSWORD  # Sender's email password
                    email_recipient = email_address  # Recipient's email

                    # Email content setup
                    msg = MIMEMultipart()
                    email_message = "Attached are today's market movers"
                    msg['From'] = email_sender
                    msg['To'] = email_recipient
                    msg['Subject'] = "Stock Market Movers"
                    msg.attach(MIMEText(email_message, 'plain'))

                    # Attaching the CSV file to the email
                    attachment_location = file_name
                    if attachment_location != '':
                        filename = os.path.basename(attachment_location)
                        attachment = open(attachment_location, 'rb')
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', "attachment; filename=%s" % filename)
                        msg.attach(part)

                    # Send the email using SMTP protocol
                    try:
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.ehlo()
                        server.starttls()
                        server.login(email_sender, email_password)
                        text = msg.as_string()
                        server.sendmail(email_sender, email_recipient, text)
                        print('Email sent successfully.')
                        server.quit()
                    except Exception as e:
                        print(f'Failed to send email: {e}')
                    
                    return schedule.CancelJob

                # Schedule the email to be sent every day at a specific time (e.g., 4:00 PM)
                schedule.every().day.at("16:00").do(send_email)

                # Run the scheduled task
                while True:
                    schedule.run_pending()
                    time.sleep(1)

        if pred_option_data == "Stock VWAP":
            st.success("This segment allows us to analyze the Volume Weighted Average Price of a ticker")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):
                # Override yfinance with pandas datareader
                yf.pdr_override()
                # Define the stock symbol to analyze
                stock = ticker

                # Function to retrieve stock data for a specified period
                def get_symbol(symbol):
                    # Fetch the stock data
                    df = pdr.get_data_yahoo(symbol, start_date, end_date)
                    return df

                # Function to calculate VWAP (Volume Weighted Average Price)
                def VWAP():
                    df = get_symbol(stock)

                    # Calculate typical price
                    df['Typical_Price'] = (df['High'] + df['Low'] + df['Adj Close']) / 3
                    df['TP_Volume'] = df['Typical_Price'] * df['Volume']

                    # Calculate VWAP
                    cumulative_TP_V = df['TP_Volume'].sum()
                    cumulative_V = df['Volume'].sum()
                    vwap = cumulative_TP_V / cumulative_V
                    return vwap

                # Displaying the VWAP for the specified stock
                print("VWAP: ", VWAP())

                # Function to update VWAP with a different method
                def update_VWAP():
                    df = get_symbol(stock)

                    # Calculate weighted prices
                    df['OpenxVolume'] = df['Open'] * df['Volume']
                    df['HighxVolume'] = df['High'] * df['Volume']
                    df['LowxVolume'] = df['Low'] * df['Volume']
                    df['ClosexVolume'] = df['Adj Close'] * df['Volume']

                    # Calculate VWAP components
                    sum_volume = df['Volume'].sum()
                    sum_x_OV = df['OpenxVolume'].sum() / sum_volume
                    sum_x_HV = df['HighxVolume'].sum() / sum_volume
                    sum_x_LV = df['LowxVolume'].sum() / sum_volume
                    sum_x_CV = df['ClosexVolume'].sum() / sum_volume
                    average_volume_each = (sum_x_OV + sum_x_HV + sum_x_LV + sum_x_OV) / 4

                    # Calculate updated VWAP
                    new_vwap = ((df['Adj Close'][-1] - average_volume_each) + (df['Adj Close'][-1] + average_volume_each)) / 2
                    return new_vwap

                # Display the updated VWAP
                st.write("Updated VWAP: ", update_VWAP())

                # Function to add a VWAP column to the stock data
                def add_VWAP_column():
                    df = get_symbol(stock)

                    # Calculate weighted prices
                    df['OpenxVolume'] = df['Open'] * df['Volume']
                    df['HighxVolume'] = df['High'] * df['Volume']
                    df['LowxVolume'] = df['Low'] * df['Volume']
                    df['ClosexVolume'] = df['Adj Close'] * df['Volume']

                    # Calculate and add the VWAP column
                    vwap_column = (df[['OpenxVolume', 'HighxVolume', 'LowxVolume', 'ClosexVolume']].mean(axis=1)) / df['Volume']
                    df['VWAP'] = vwap_column
                    return df

                # Print the stock data with the added VWAP column
                st.write(add_VWAP_column())

        if pred_option_data == "Data SMS stock":
            st.success("This segment allows us to get the stock data SMS")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            email_address = st.text_input("Enter your Email address")
            if email_address:
                message_three = (f"Email address captured is {email_address}")
                st.success(message_three)

            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):
                # Function to send an email message
                def send_message(text, sender_email, receiver_email, password):
                    try:
                        # Create and configure the email message
                        msg = MIMEMultipart()
                        msg['From'] = sender_email
                        msg['To'] = receiver_email
                        msg['Subject'] = "Stock Data"
                        msg.attach(MIMEText(text, 'plain'))

                        # Establish SMTP connection and send the email
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls()
                        server.login(sender_email, password)
                        server.sendmail(sender_email, receiver_email, msg.as_string())
                        server.quit()
                        print('Email sent successfully')
                    except Exception as e:
                        print(f'Error sending email: {e}')

                # Function to retrieve and process stock data
                def get_data(ticker):
                    sender_email = EMAIL_ADDRESS  # Sender's email
                    receiver_email = email_address  # Receiver's email
                    password = EMAIL_PASSWORD  # Sender's email password

                    try:
                        # Fetch historical stock data
                        stock  = ticker
                        df = yf.download(stock, start_date, end_date)
                        st.write(f'Retrieving data for {stock}')

                        # Compute current price and other metrics
                        price = round(df['Adj Close'][-1], 2)
                        df["rsi"] = ta.RSI(df["Adj Close"])
                        rsi = round(df["rsi"].tail(14).mean(), 2)

                        # Scrape and analyze news sentiment
                        finviz_url = 'https://finviz.com/quote.ashx?t='
                        req = Request(url=finviz_url + stock, headers={'user-agent': 'Mozilla/5.0'})
                        response = urlopen(req).read()
                        html = BeautifulSoup(response, "html.parser")
                        news_df = pd.read_html(str(html), attrs={'id': 'news-table'})[0]
                        # Process news data
                        news_df.columns = ['datetime', 'headline']
                        news_df['date'] = pd.to_datetime(news_df['datetime'].str.split(' ').str[0], errors='coerce')
                        news_df['date'] = news_df['date'].fillna(method='ffill')
                        news_df['sentiment'] = news_df['headline'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
                        sentiment = round(news_df['sentiment'].mean(), 2)

                        # Prepare and send email message
                        output = f"\nTicker: {stock}\nCurrent Price: {price}\nNews Sentiment: {sentiment}\nRSI: {rsi}"
                        send_message(output, sender_email, receiver_email, password)

                    except Exception as e:
                        st.write(f'Error processing data for {stock}: {e}')
                        
                get_data(ticker)

        if pred_option_data == "Stock Earnings":
            #TODO - fix the meta error from earnings df directly
            st.success("This segment allows us to get the stock data SMS")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            email_address = st.text_input("Enter your Email address")
            if email_address:
                message_three = (f"Email address captured is {email_address}")
                st.success(message_three)

            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):

                yf.pdr_override()

                # Setting pandas display options
                pd.set_option('display.max_columns', None)

                # Download earnings report for a specific date
                report_date = datetime.now().date()
                yec = YahooEarningsCalendar()

                # Fetch earnings data for the specified dates
                earnings_df_list = yec.earnings_between(start_date, end_date)

                # Create a DataFrame from the earnings data and drop unnecessary columns
                earnings_day_df = pd.DataFrame(earnings_df_list)
                earnings_day_df.drop(columns=['gmtOffsetMilliSeconds', 'quoteType', 'timeZoneShortName'], inplace=True)

                # Rename columns for clarity and reorganize them
                earnings_day_df.columns = ['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']
                earnings_day_df = earnings_day_df[['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']]

                # Adjust datetime to local timezone
                earnings_day_df['DateTime'] = pd.to_datetime(earnings_day_df['DateTime']).dt.tz_localize(None)

                # Print the DataFrame
                st.write(earnings_day_df)

                # Download earnings for a range of dates
                DAYS_AHEAD = 7
                start_date = datetime.now().date()
                end_date = (datetime.now().date() + timedelta(days=DAYS_AHEAD))

                # Fetch earnings data between specified dates
                earnings_range_list = yec.earnings_between(start_date, end_date)

                # Create a DataFrame from the fetched data and drop unnecessary columns
                earnings_range_df = pd.DataFrame(earnings_range_list)
                earnings_range_df.drop(columns=['gmtOffsetMilliSeconds', 'quoteType', 'timeZoneShortName'], inplace=True)

                # Rename columns for clarity and reorganize them
                earnings_range_df.columns = ['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']
                earnings_range_df = earnings_range_df[['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']]

                # Adjust datetime to local timezone
                earnings_range_df['DateTime'] = pd.to_datetime(earnings_range_df['DateTime']).dt.tz_localize(None)

                # Print the DataFrame
                print(earnings_range_df)

                # Download earnings for a specific ticker within a date range
                TICKER = 'AAPL'
                DAYS_AHEAD = 180
                start_date = datetime.now().date()
                end_date = (datetime.now().date() + timedelta(days=DAYS_AHEAD))

                # Fetch earnings data for the specified ticker
                earnings_ticker_list = yec.get_earnings_of(TICKER)

                # Create a DataFrame from the fetched data and filter by date range
                earnings_ticker_df = pd.DataFrame(earnings_ticker_list)
                earnings_ticker_df['report_date'] = earnings_ticker_df['startdatetime'].apply(lambda x: dateutil.parser.isoparse(x).date())
                earnings_ticker_df = earnings_ticker_df[earnings_ticker_df['report_date'].between(start_date, end_date)].sort_values('report_date')
                earnings_ticker_df.drop(columns=['gmtOffsetMilliSeconds', 'quoteType', 'timeZoneShortName', 'report_date'], inplace=True)

                # Rename columns for clarity and reorganize them
                earnings_ticker_df.columns = ['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']
                earnings_ticker_df = earnings_ticker_df[['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']]

                # Adjust datetime to local timezone
                earnings_ticker_df['DateTime'] = pd.to_datetime(earnings_ticker_df['DateTime']).dt.tz_localize(None)

                print(earnings_ticker_df)


        if pred_option_data == "Trading view Intraday":
            if st.button("Check"):
                # Function to filter relevant data from websocket messages
                def filter_raw_message(text):
                    try:
                        found = re.search('"m":"(.+?)",', text).group(1)
                        found2 = re.search('"p":(.+?"}"])}', text).group(1)
                        print(found)
                        print(found2)
                        return found, found2
                    except AttributeError:
                        print("Error in filtering message")

                # Function to generate a random session ID
                def generateSession():
                    stringLength = 12
                    letters = string.ascii_lowercase
                    random_string = ''.join(random.choice(letters) for i in range(stringLength))
                    return "qs_" + random_string

                # Function to generate a random chart session ID
                def generateChartSession():
                    stringLength = 12
                    letters = string.ascii_lowercase
                    random_string = ''.join(random.choice(letters) for i in range(stringLength))
                    return "cs_" + random_string

                # Function to prepend header for websocket message
                def prependHeader(st):
                    return "~m~" + str(len(st)) + "~m~" + st

                # Function to construct JSON message for websocket
                def constructMessage(func, paramList):
                    return json.dumps({"m": func, "p": paramList}, separators=(',', ':'))

                # Function to create a full message with header and JSON payload
                def createMessage(func, paramList):
                    return prependHeader(constructMessage(func, paramList))

                # Function to send a raw message over the websocket
                def sendRawMessage(ws, message):
                    ws.send(prependHeader(message))

                # Function to send a full message with header and JSON payload
                def sendMessage(ws, func, args):
                    ws.send(createMessage(func, args))

                # Function to extract data from websocket message and save to CSV file
                def generate_csv(a):
                    out = re.search('"s":\[(.+?)\}\]', a).group(1)
                    x = out.split(',{\"')
                    
                    with open('data_file.csv', mode='w', newline='') as data_file:
                        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        data_writer.writerow(['index', 'date', 'open', 'high', 'low', 'close', 'volume'])
                        
                        for xi in x:
                            xi = re.split('\[|:|,|\]', xi)
                            ind = int(xi[1])
                            ts = datetime.datetime.fromtimestamp(float(xi[4])).strftime("%Y/%m/%d, %H:%M:%S")
                            data_writer.writerow([ind, ts, float(xi[5]), float(xi[6]), float(xi[7]), float(xi[8]), float(xi[9])])

                # Initialize headers for websocket connection
                headers = json.dumps({'Origin': 'https://data.tradingview.com'})

                # Create a connection to the websocket
                ws = create_connection('wss://data.tradingview.com/socket.io/websocket', headers=headers)

                # Generate session and chart session IDs
                session = generateSession()
                chart_session = generateChartSession()

                # Send various messages to establish the websocket connection and start streaming data
                sendMessage(ws, "set_auth_token", ["unauthorized_user_token"])
                sendMessage(ws, "chart_create_session", [chart_session, ""])
                sendMessage(ws, "quote_create_session", [session])
                sendMessage(ws,"quote_set_fields", [session,"ch","chp","current_session","description","local_description","language","exchange","fractional","is_tradable","lp","lp_time","minmov","minmove2","original_name","pricescale","pro_name","short_name","type","update_mode","volume","currency_code","rchp","rtc"])
                sendMessage(ws, "quote_add_symbols",[session, "NASDAQ:AAPL", {"flags":['force_permission']}])
                sendMessage(ws, "quote_fast_symbols", [session,"NASDAQ:AAPL"])
                sendMessage(ws, "resolve_symbol", [chart_session,"symbol_1","={\"symbol\":\"NASDAQ:AAPL\",\"adjustment\":\"splits\",\"session\":\"extended\"}"])
                sendMessage(ws, "create_series", [chart_session, "s1", "s1", "symbol_1", "1", 5000])


                # Receiving and printing data from the websocket
                data = ""
                while True:
                    try:
                        result = ws.recv()
                        data += result + "\n"
                        st.write(result)
                    except Exception as e:
                        st.write(e)
                        break

                # Generating a CSV from the received data
                st.write(data)
                #generate_csv(data)

        if pred_option_data == "Trading view Recommendations":
            
            st.success("This segment allows us to get recommendations fro Trading View")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            email_address = st.text_input("Enter your Email address")
            if email_address:
                message_three = (f"Email address captured is {email_address}")
                st.success(message_three)

            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):
        
                interval = '1D'

                # Getting lists of tickers from NASDAQ, NYSE, and AMEX
                nasdaq = ti.tickers_nasdaq()
                nyse = ti.tickers_nyse()
                amex = ti.tickers_amex()

                # Define valid time intervals
                type_intervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1D', '1W', '1M']

                # Set up Selenium WebDriver
                options = Options()
                options.add_argument("--headless")
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

                # Helper function to parse recommendation data
                def parse_recommendation(recommendation_elements, counter_elements, rec_index, sell_index, buy_index):
                    rec = recommendation_elements[rec_index].get_attribute('innerHTML')
                    sell_signals = int(counter_elements[sell_index].get_attribute('innerHTML'))
                    neutral_signals = int(counter_elements[sell_index + 1].get_attribute('innerHTML'))
                    buy_signals = int(counter_elements[buy_index].get_attribute('innerHTML'))
                    return rec, sell_signals, neutral_signals, buy_signals

                # Helper function to display recommendations
                def display_recommendations(analysis):
                    for key in analysis:
                        st.write(f"\n{key} Recommendation: ")
                        rec, sell, neutral, buy = analysis[key]
                        st.write(f"Recommendation: {rec}")
                        st.write(f"Sell Signals: {sell}")
                        st.write(f"Neutral Signals: {neutral}")
                        st.write(f"Buy Signals: {buy}")

                # Helper function to scrape tables
                def scrape_tables(html):
                    tables = pd.read_html(html, attrs = {'class': 'table-hvDpy38G'})
                    return tables[0], tables[1], tables[2]  # Oscillator, Moving Averages, and Pivots tables

                # Helper function to print tables
                def print_tables(oscillator_table, ma_table, pivots_table):
                    st.write("\nOscillator Table:")
                    st.write(oscillator_table)
                    st.write("\nMoving Average Table:")
                    st.write(ma_table)
                    st.write("\nPivots Table:")
                    st.write(pivots_table)

                # Process each ticker 
                try:
                    # Determine the exchange of the ticker
                    if ticker in nasdaq:
                        exchange = 'NASDAQ'
                    elif ticker in nyse:
                        exchange = 'NYSE'
                    elif ticker in amex:
                        exchange = 'AMEX'
                    else:
                        print(f"Could not find the exchange for {ticker}")
                        
                    # Get the current price of the ticker
                    df = pdr.get_data_yahoo(ticker)
                    price = round(df["Adj Close"][-1], 2)

                    # Open TradingView page for the ticker
                    driver.get(f"https://www.tradingview.com/symbols/{exchange}-{ticker}/technicals")
                    time.sleep(1)

                    # Display ticker, interval, and price information
                    st.write('\nTicker: ' + ticker)
                    st.write('Interval: ' + interval)
                    st.write('Price: ' + str(price))

                    # Switch to the specified interval on TradingView page
                    for type_interval in type_intervals:
                        if interval == type_interval:
                            # n = type_intervals.index(type_interval)
                            element = driver.find_element(By.XPATH, f'//*[@id="{interval}"]')
                            # print (len(element))
                            element.click()
                            break

                    time.sleep(1)

                    # Scrape and display Overall, Oscillator, and Moving Average Recommendations
                    recommendation_elements = driver.find_elements(By.CLASS_NAME, "speedometerText-Tat_6ZmA")
                    counter_elements = driver.find_elements(By.CLASS_NAME, "counterNumber-kg4MJrFB")
                    analysis = {
                        'Overall': parse_recommendation(recommendation_elements, counter_elements, 1, 3, 5),
                        'Oscillator': parse_recommendation(recommendation_elements, counter_elements, 0, 0, 2),
                        'Moving Average': parse_recommendation(recommendation_elements, counter_elements, 2, 6, 8)
                    }
                    display_recommendations(analysis)

                    # Scrape and display tables for Oscillator, Moving Averages, and Pivots
                    html = driver.page_source
                    oscillator_table, ma_table, pivots_table = scrape_tables(html)
                    print_tables(oscillator_table, ma_table, pivots_table)

                except Exception as e:
                    st.write(f'Could not retrieve stats for {ticker} due to {e}')

                driver.close()
    elif option =='Stock Analysis':
        pred_option_analysis = st.selectbox('Make a choice', ['Backtest All Indicators',
                                                    'CAPM Analysis',
                                                    'Earnings Sentiment Analysis',
                                                    'Intrinsic Value analysis',
                                                    'Kelly Criterion',
                                                    'MA Backtesting',
                                                    'Ols Regression',
                                                    'Perfomance Risk Analysis',
                                                    'Risk/Returns Analysis',
                                                    'Seasonal Stock Analysis',
                                                    'SMA Histogram',
                                                    'SP500 COT Sentiment Analysis',
                                                    'SP500 Valuation',
                                                    'Stock Pivot Resistance',
                                                    'Stock Profit/Loss Analysis',
                                                    'Stock Return Statistical Analysis',
                                                    'VAR Analysis',
                                                    'Stock Returns'])

        if pred_option_analysis == "Backtest All Indicators":
            st.success("This segment allows us to Backtest Most Technical Indicators of a ticker")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            min_date = datetime(1980, 1, 1)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Backtest"):
                start_date =  str(start_date)
                end_date  = str(end_date)

                pd.set_option("display.max_columns", None)
                pd.set_option("display.max_rows", None)


                def get_stock_backtest_data(ticker, start, end):
                    date_fmt = "%Y-%m-%d"

                    start_date_buffer = datetime.strptime(start_date, date_fmt) - timedelta(days=365)
                    start_date_buffer = start_date_buffer.strftime(date_fmt)

                    df = yf.download(ticker, start=start_date_buffer, end=end_date)

                    return df

                df = get_stock_backtest_data(ticker, start_date, end_date)
                df["CLOSE_PREV"] = df.Close.shift(1)

                k_band = taf.volatility.KeltnerChannel(df.High, df.Low, df.Close, 10)

                df["K_BAND_UB"] = k_band.keltner_channel_hband().round(4)
                df["K_BAND_LB"] = k_band.keltner_channel_lband().round(4)

                df[["K_BAND_UB", "K_BAND_LB"]].dropna().head()

                df["LONG"] = (df.Close <= df.K_BAND_LB) & (df.CLOSE_PREV > df.K_BAND_LB)
                df["EXIT_LONG"] = (df.Close >= df.K_BAND_UB) & (df.CLOSE_PREV < df.K_BAND_UB)

                df["SHORT"] = (df.Close >= df.K_BAND_UB) & (df.CLOSE_PREV < df.K_BAND_UB)
                df["EXIT_SHORT"] = (df.Close <= df.K_BAND_LB) & (df.CLOSE_PREV > df.K_BAND_LB)

                df.LONG = df.LONG.shift(1)
                df.EXIT_LONG = df.EXIT_LONG.shift(1)
                df.SHORT = df.SHORT.shift(1)
                df.EXIT_SHORT = df.EXIT_SHORT.shift(1)

                st.write(df[["LONG", "EXIT_LONG", "SHORT", "EXIT_SHORT"]].dropna().head())

                def strategy_KeltnerChannel_origin(df, **kwargs):
                    n = kwargs.get("n", 10)
                    data = df.copy()

                    k_band = taf.volatility.KeltnerChannel(data.High, data.Low, data.Close, n)

                    data["K_BAND_UB"] = k_band.keltner_channel_hband().round(4)
                    data["K_BAND_LB"] = k_band.keltner_channel_lband().round(4)

                    data["CLOSE_PREV"] = data.Close.shift(1)

                    data["LONG"] = (data.Close <= data.K_BAND_LB) & (data.CLOSE_PREV > data.K_BAND_LB)
                    data["EXIT_LONG"] = (data.Close >= data.K_BAND_UB) & (
                        data.CLOSE_PREV < data.K_BAND_UB
                    )

                    data["SHORT"] = (data.Close >= data.K_BAND_UB) & (data.CLOSE_PREV < data.K_BAND_UB)
                    data["EXIT_SHORT"] = (data.Close <= data.K_BAND_LB) & (
                        data.CLOSE_PREV > data.K_BAND_LB
                    )

                    data.LONG = data.LONG.shift(1)
                    data.EXIT_LONG = data.EXIT_LONG.shift(1)
                    data.SHORT = data.SHORT.shift(1)
                    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

                    return data

                df = strategy_KeltnerChannel_origin(df, n=10)

                def strategy_BollingerBands(df, **kwargs):
                    n = kwargs.get("n", 10)
                    n_rng = kwargs.get("n_rng", 2)
                    data = df.copy()

                    boll = taf.volatility.BollingerBands(data.Close, n, n_rng)

                    data["BOLL_LBAND_INDI"] = boll.bollinger_lband_indicator()
                    data["BOLL_UBAND_INDI"] = boll.bollinger_hband_indicator()

                    data["CLOSE_PREV"] = data.Close.shift(1)

                    data["LONG"] = data.BOLL_LBAND_INDI == 1
                    data["EXIT_LONG"] = data.BOLL_UBAND_INDI == 1

                    data["SHORT"] = data.BOLL_UBAND_INDI == 1
                    data["EXIT_SHORT"] = data.BOLL_LBAND_INDI == 1

                    data.LONG = data.LONG.shift(1)
                    data.EXIT_LONG = data.EXIT_LONG.shift(1)
                    data.SHORT = data.SHORT.shift(1)
                    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

                    return data

                # df = get_stock_backtest_data(ticker, start_date, end_date)
                # strategy_BollingerBands(df, n=10, n_rng=2)

                def strategy_MA(df, **kwargs):
                    n = kwargs.get("n", 50)
                    ma_type = kwargs.get("ma_type", "sma")
                    ma_type = ma_type.strip().lower()
                    data = df.copy()

                    if ma_type == "sma":
                        sma = taf.trend.SMAIndicator(data.Close, n)
                        data["MA"] = sma.sma_indicator().round(4)
                    elif ma_type == "ema":
                        ema = taf.trend.EMAIndicator(data.Close, n)
                        data["MA"] = ema.ema_indicator().round(4)

                    data["CLOSE_PREV"] = data.Close.shift(1)

                    data["LONG"] = (data.Close > data.MA) & (data.CLOSE_PREV <= data.MA)
                    data["EXIT_LONG"] = (data.Close < data.MA) & (data.CLOSE_PREV >= data.MA)

                    data["SHORT"] = (data.Close < data.MA) & (data.CLOSE_PREV >= data.MA)
                    data["EXIT_SHORT"] = (data.Close > data.MA) & (data.CLOSE_PREV <= data.MA)

                    data.LONG = data.LONG.shift(1)
                    data.EXIT_LONG = data.EXIT_LONG.shift(1)
                    data.SHORT = data.SHORT.shift(1)
                    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

                    return data

                # df = get_stock_backtest_data(ticker, start_date, end_date)
                # strategy_SMA(df, n=10, ma_type='ema')

                def strategy_MACD(df, **kwargs):
                    n_slow = kwargs.get("n_slow", 26)
                    n_fast = kwargs.get("n_fast", 12)
                    n_sign = kwargs.get("n_sign", 9)
                    data = df.copy()

                    macd = taf.trend.MACD(data.Close, n_slow, n_fast, n_sign)

                    data["MACD_DIFF"] = macd.macd_diff().round(4)
                    data["MACD_DIFF_PREV"] = data.MACD_DIFF.shift(1)

                    data["LONG"] = (data.MACD_DIFF > 0) & (data.MACD_DIFF_PREV <= 0)
                    data["EXIT_LONG"] = (data.MACD_DIFF < 0) & (data.MACD_DIFF_PREV >= 0)

                    data["SHORT"] = (data.MACD_DIFF < 0) & (data.MACD_DIFF_PREV >= 0)
                    data["EXIT_SHORT"] = (data.MACD_DIFF > 0) & (data.MACD_DIFF_PREV <= 0)

                    data.LONG = data.LONG.shift(1)
                    data.EXIT_LONG = data.EXIT_LONG.shift(1)
                    data.SHORT = data.SHORT.shift(1)
                    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

                    return data

                # df = get_stock_backtest_data(ticker, start_date, end_date)
                # strategy_MACD(df, n_slow=26, n_fast=12, n_sign=9)

                def strategy_RSI(df, **kwargs):
                    n = kwargs.get("n", 14)
                    data = df.copy()

                    rsi = taf.momentum.RSIIndicator(data.Close, n)

                    data["RSI"] = rsi.rsi().round(4)
                    data["RSI_PREV"] = data.RSI.shift(1)

                    data["LONG"] = (data.RSI > 30) & (data.RSI_PREV <= 30)
                    data["EXIT_LONG"] = (data.RSI < 70) & (data.RSI_PREV >= 70)

                    data["SHORT"] = (data.RSI < 70) & (data.RSI_PREV >= 70)
                    data["EXIT_SHORT"] = (data.RSI > 30) & (data.RSI_PREV <= 30)

                    data.LONG = data.LONG.shift(1)
                    data.EXIT_LONG = data.EXIT_LONG.shift(1)
                    data.SHORT = data.SHORT.shift(1)
                    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

                    return data

                # df = get_stock_backtest_data(ticker, start_date, end_date)
                # strategy_RSI(df, n_slow=26, n_fast=12, n_sign=9)

                def strategy_WR(df, **kwargs):
                    n = kwargs.get("n", 14)
                    data = df.copy()

                    wr = taf.momentum.WilliamsRIndicator(data.High, data.Low, data.Close, n)

                    data["WR"] = wr._wr.round(4)
                    data["WR_PREV"] = data.WR.shift(1)

                    data["LONG"] = (data.WR > -80) & (data.WR_PREV <= -80)
                    data["EXIT_LONG"] = (data.WR < -20) & (data.WR_PREV >= -20)

                    data["SHORT"] = (data.WR < -20) & (data.WR_PREV >= -20)
                    data["EXIT_SHORT"] = (data.WR > -80) & (data.WR_PREV <= -80)

                    data.LONG = data.LONG.shift(1)
                    data.EXIT_LONG = data.EXIT_LONG.shift(1)
                    data.SHORT = data.SHORT.shift(1)
                    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

                    return data

                # df = get_stock_backtest_data(ticker, start_date, end_date)
                # strategy_WR(df, n_slow=26, n_fast=12, n_sign=9)

                def strategy_Stochastic_fast(df, **kwargs):
                    k = kwargs.get("k", 20)
                    d = kwargs.get("d", 5)
                    data = df.copy()

                    sto = taf.momentum.StochasticOscillator(data.High, data.Low, data.Close, k, d)

                    data["K"] = sto.stoch().round(4)
                    data["D"] = sto.stoch_signal().round(4)
                    data["DIFF"] = data["K"] - data["D"]
                    data["DIFF_PREV"] = data.DIFF.shift(1)

                    data["LONG"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
                    data["EXIT_LONG"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)

                    data["SHORT"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
                    data["EXIT_SHORT"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)

                    data.LONG = data.LONG.shift(1)
                    data.EXIT_LONG = data.EXIT_LONG.shift(1)
                    data.SHORT = data.SHORT.shift(1)
                    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

                    return data

                # df = get_stock_backtest_data(ticker, start_date, end_date)
                # strategy_Stochastic_fast(df, k=20, d=5)

                def strategy_Stochastic_slow(df, **kwargs):
                    k = kwargs.get("k", 20)
                    d = kwargs.get("d", 5)
                    dd = kwargs.get("dd", 3)
                    data = df.copy()

                    sto = taf.momentum.StochasticOscillator(data.High, data.Low, data.Close, k, d)

                    data["K"] = sto.stoch().round(4)
                    data["D"] = sto.stoch_signal().round(4)

                    ma = taf.trend.SMAIndicator(data.D, dd)
                    data["DD"] = ma.sma_indicator().round(4)

                    data["DIFF"] = data["D"] - data["DD"]
                    data["DIFF_PREV"] = data.DIFF.shift(1)

                    data["LONG"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
                    data["EXIT_LONG"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)

                    data["SHORT"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
                    data["EXIT_SHORT"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)

                    data.LONG = data.LONG.shift(1)
                    data.EXIT_LONG = data.EXIT_LONG.shift(1)
                    data.SHORT = data.SHORT.shift(1)
                    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

                    return data

                # df = get_stock_backtest_data(ticker, start_date, end_date)
                # strategy_Stochastic_slow(df, k=20, d=5, dd=3)

                def strategy_Ichmoku(df, **kwargs):
                    n_conv = kwargs.get("n_conv", 9)
                    n_base = kwargs.get("n_base", 26)
                    n_span_b = kwargs.get("n_span_b", 26)
                    data = df.copy()

                    ichmoku = taf.trend.IchimokuIndicator(data.High, data.Low, n_conv, n_base, n_span_b)

                    data["BASE"] = ichmoku.ichimoku_base_line().round(4)
                    data["CONV"] = ichmoku.ichimoku_conversion_line().round(4)

                    data["DIFF"] = data["CONV"] - data["BASE"]
                    data["DIFF_PREV"] = data.DIFF.shift(1)

                    data["LONG"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
                    data["EXIT_LONG"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)

                    data["SHORT"] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
                    data["EXIT_SHORT"] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)

                    data.LONG = data.LONG.shift(1)
                    data.EXIT_LONG = data.EXIT_LONG.shift(1)
                    data.SHORT = data.SHORT.shift(1)
                    data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

                    return data

                # df = get_stock_backtest_data(ticker, start_date, end_date)
                # strategy_Ichmoku(df, n_conv=9, n_base=26, n_span_b=26)

                bt_df = df[(df.index >= start_date) & (df.index <= end_date)]

                def prepare_stock_ta_backtest_data(
                    df, start_date, end_date, strategy, **strategy_params
                ):
                    df_strategy = strategy(df, **strategy_params)
                    bt_df = df_strategy[
                        (df_strategy.index >= start_date) & (df_strategy.index <= end_date)
                    ]
                    return bt_df

                bt_df = prepare_stock_ta_backtest_data(
                    df, start_date, end_date, strategy_KeltnerChannel_origin, n=10
                )

                bt_df.head()

                balance = 1000000
                pnl = 0
                position = 0

                last_signal = "hold"
                last_price = 0
                c = 0

                trade_date_start = []
                trade_date_end = []
                trade_days = []
                trade_side = []
                trade_pnl = []
                trade_ret = []

                cum_value = []

                for index, row in bt_df.iterrows():
                    # check and close any positions
                    if row.EXIT_LONG and last_signal == "long":
                        trade_date_end.append(row.name)
                        trade_days.append(c)

                        pnl = (row.Open - last_price) * position
                        trade_pnl.append(pnl)
                        trade_ret.append((row.Open / last_price - 1) * 100)

                        balance = balance + row.Open * position

                        position = 0
                        last_signal = "hold"

                        c = 0

                    elif row.EXIT_SHORT and last_signal == "short":
                        trade_date_end.append(row.name)
                        trade_days.append(c)

                        pnl = (row.Open - last_price) * position
                        trade_pnl.append(pnl)
                        trade_ret.append((last_price / row.Open - 1) * 100)

                        balance = balance + pnl

                        position = 0
                        last_signal = "hold"

                        c = 0

                    # check signal and enter any possible position
                    if row.LONG and last_signal != "long":
                        last_signal = "long"
                        last_price = row.Open
                        trade_date_start.append(row.name)
                        trade_side.append("long")

                        position = int(balance / row.Open)
                        cost = position * row.Open
                        balance = balance - cost

                        c = 0

                    elif row.SHORT and last_signal != "short":
                        last_signal = "short"
                        last_price = row.Open
                        trade_date_start.append(row.name)
                        trade_side.append("short")

                        position = int(balance / row.Open) * -1

                        c = 0

                    # compute market value and count days for any possible poisition
                    if last_signal == "hold":
                        market_value = balance
                    elif last_signal == "long":
                        c = c + 1
                        market_value = position * row.Close + balance
                    else:
                        c = c + 1
                        market_value = (row.Close - last_price) * position + balance

                    cum_value.append(market_value)

                cum_ret_df = pd.DataFrame(cum_value, index=bt_df.index, columns=["CUM_RET"])
                cum_ret_df["CUM_RET"] = (cum_ret_df.CUM_RET / 1000000 - 1) * 100
                cum_ret_df["BUY_HOLD"] = (bt_df.Close / bt_df.Open.iloc[0] - 1) * 100
                cum_ret_df["ZERO"] = 0
                cum_ret_df.plot(figsize=(15, 5))

                st.write(cum_ret_df.iloc[[-1]].round(2))

                size = min(len(trade_date_start), len(trade_date_end))

                tarde_dict = {
                    "START": trade_date_start[:size],
                    "END": trade_date_end[:size],
                    "SIDE": trade_side[:size],
                    "DAYS": trade_days[:size],
                    "PNL": trade_pnl[:size],
                    "RET": trade_ret[:size],
                }

                trade_df = pd.DataFrame(tarde_dict)
                st.write(trade_df.head())

                num_trades = trade_df.groupby("SIDE").count()[["START"]]
                num_trades_win = trade_df[trade_df.PNL > 0].groupby("SIDE").count()[["START"]]

                avg_days = trade_df.groupby("SIDE").mean()[["DAYS"]]

                avg_ret = trade_df.groupby("SIDE").mean()[["RET"]]
                avg_ret_win = trade_df[trade_df.PNL > 0].groupby("SIDE").mean()[["RET"]]
                avg_ret_loss = trade_df[trade_df.PNL < 0].groupby("SIDE").mean()[["RET"]]

                std_ret = trade_df.groupby("SIDE").std()[["RET"]]

                detail_df = pd.concat(
                    [num_trades, num_trades_win, avg_days, avg_ret, avg_ret_win, avg_ret_loss, std_ret],
                    axis=1,
                    sort=False,
                )

                detail_df.columns = [
                    "NUM_TRADES",
                    "NUM_TRADES_WIN",
                    "AVG_DAYS",
                    "AVG_RET",
                    "AVG_RET_WIN",
                    "AVG_RET_LOSS",
                    "STD_RET",
                ]
                st.write(detail_df.round(2))

                # Stop Loss

                balance = 1000000
                pnl = 0
                position = 0

                stop_loss_lvl = -2

                last_signal = "hold"
                last_price = 0
                c = 0

                trade_date_start = []
                trade_date_end = []
                trade_days = []
                trade_side = []
                trade_pnl = []
                trade_ret = []

                cum_value = []

                for index, row in bt_df.iterrows():
                    # check and close any positions
                    if row.EXIT_LONG and last_signal == "long":
                        trade_date_end.append(row.name)
                        trade_days.append(c)

                        pnl = (row.Open - last_price) * position
                        trade_pnl.append(pnl)
                        trade_ret.append((row.Open / last_price - 1) * 100)

                        balance = balance + row.Open * position

                        position = 0
                        last_signal = "hold"

                        c = 0

                    elif row.EXIT_SHORT and last_signal == "short":
                        trade_date_end.append(row.name)
                        trade_days.append(c)

                        pnl = (row.Open - last_price) * position
                        trade_pnl.append(pnl)
                        trade_ret.append((last_price / row.Open - 1) * 100)

                        balance = balance + pnl

                        position = 0
                        last_signal = "hold"

                        c = 0

                    # check signal and enter any possible position
                    if row.LONG and last_signal != "long":
                        last_signal = "long"
                        last_price = row.Open
                        trade_date_start.append(row.name)
                        trade_side.append("long")

                        position = int(balance / row.Open)
                        cost = position * row.Open
                        balance = balance - cost

                        c = 0

                    elif row.SHORT and last_signal != "short":
                        last_signal = "short"
                        last_price = row.Open
                        trade_date_start.append(row.name)
                        trade_side.append("short")

                        position = int(balance / row.Open) * -1

                        c = 0

                    # check stop loss
                    if (
                        last_signal == "long"
                        and c > 0
                        and (row.Low / last_price - 1) * 100 <= stop_loss_lvl
                    ):
                        c = c + 1

                        trade_date_end.append(row.name)
                        trade_days.append(c)

                        stop_loss_price = last_price + round(last_price * (stop_loss_lvl / 100), 4)

                        pnl = (stop_loss_price - last_price) * position
                        trade_pnl.append(pnl)
                        trade_ret.append((stop_loss_price / last_price - 1) * 100)

                        balance = balance + stop_loss_price * position

                        position = 0
                        last_signal = "hold"

                        c = 0

                    elif (
                        last_signal == "short"
                        and c > 0
                        and (last_price / row.High - 1) * 100 <= stop_loss_lvl
                    ):
                        c = c + 1

                        trade_date_end.append(row.name)
                        trade_days.append(c)

                        stop_loss_price = last_price - round(last_price * (stop_loss_lvl / 100), 4)

                        pnl = (stop_loss_price - last_price) * position
                        trade_pnl.append(pnl)
                        trade_ret.append((last_price / stop_loss_price - 1) * 100)

                        balance = balance + pnl

                        position = 0
                        last_signal = "hold"

                        c = 0

                    # compute market value and count days for any possible poisition
                    if last_signal == "hold":
                        market_value = balance
                    elif last_signal == "long":
                        c = c + 1
                        market_value = position * row.Close + balance
                    else:
                        c = c + 1
                        market_value = (row.Close - last_price) * position + balance

                    cum_value.append(market_value)

                cum_ret_df = pd.DataFrame(cum_value, index=bt_df.index, columns=["CUM_RET"])
                cum_ret_df["CUM_RET"] = (cum_ret_df.CUM_RET / 1000000 - 1) * 100
                cum_ret_df["BUY_HOLD"] = (bt_df.Close / bt_df.Open.iloc[0] - 1) * 100
                cum_ret_df["ZERO"] = 0
                cum_ret_df.plot(figsize=(15, 5))

                st.write(cum_ret_df.iloc[[-1]].round(2))

                size = min(len(trade_date_start), len(trade_date_end))

                tarde_dict = {
                    "START": trade_date_start[:size],
                    "END": trade_date_end[:size],
                    "SIDE": trade_side[:size],
                    "DAYS": trade_days[:size],
                    "PNL": trade_pnl[:size],
                    "RET": trade_ret[:size],
                }

                trade_df = pd.DataFrame(tarde_dict)
                st.write(trade_df.head())

                num_trades = trade_df.groupby("SIDE").count()[["START"]]
                num_trades_win = trade_df[trade_df.PNL > 0].groupby("SIDE").count()[["START"]]

                avg_days = trade_df.groupby("SIDE").mean()[["DAYS"]]

                avg_ret = trade_df.groupby("SIDE").mean()[["RET"]]
                avg_ret_win = trade_df[trade_df.PNL > 0].groupby("SIDE").mean()[["RET"]]
                avg_ret_loss = trade_df[trade_df.PNL < 0].groupby("SIDE").mean()[["RET"]]

                std_ret = trade_df.groupby("SIDE").std()[["RET"]]

                detail_df = pd.concat(
                    [num_trades, num_trades_win, avg_days, avg_ret, avg_ret_win, avg_ret_loss, std_ret],
                    axis=1,
                    sort=False,
                )

                detail_df.columns = [
                    "NUM_TRADES",
                    "NUM_TRADES_WIN",
                    "AVG_DAYS",
                    "AVG_RET",
                    "AVG_RET_WIN",
                    "AVG_RET_LOSS",
                    "STD_RET",
                ]
                st.write(detail_df.round(2))

                mv_df = pd.DataFrame(cum_value, index=bt_df.index, columns=["MV"])
                st.write(mv_df.head())

                days = len(mv_df)

                roll_max = mv_df.MV.rolling(window=days, min_periods=1).max()
                drawdown_val = mv_df.MV - roll_max
                drawdown_pct = (mv_df.MV / roll_max - 1) * 100

                st.write("Max Drawdown Value:", round(drawdown_val.min(), 0))
                st.write("Max Drawdown %:", round(drawdown_pct.min(), 2))

                def run_stock_ta_backtest(bt_df, stop_loss_lvl=None):
                    balance = 1000000
                    pnl = 0
                    position = 0

                    last_signal = "hold"
                    last_price = 0
                    c = 0

                    trade_date_start = []
                    trade_date_end = []
                    trade_days = []
                    trade_side = []
                    trade_pnl = []
                    trade_ret = []

                    cum_value = []

                    for index, row in bt_df.iterrows():
                        # check and close any positions
                        if row.EXIT_LONG and last_signal == "long":
                            trade_date_end.append(row.name)
                            trade_days.append(c)

                            pnl = (row.Open - last_price) * position
                            trade_pnl.append(pnl)
                            trade_ret.append((row.Open / last_price - 1) * 100)

                            balance = balance + row.Open * position

                            position = 0
                            last_signal = "hold"

                            c = 0

                        elif row.EXIT_SHORT and last_signal == "short":
                            trade_date_end.append(row.name)
                            trade_days.append(c)

                            pnl = (row.Open - last_price) * position
                            trade_pnl.append(pnl)
                            trade_ret.append((last_price / row.Open - 1) * 100)

                            balance = balance + pnl

                            position = 0
                            last_signal = "hold"

                            c = 0

                        # check signal and enter any possible position
                        if row.LONG and last_signal != "long":
                            last_signal = "long"
                            last_price = row.Open
                            trade_date_start.append(row.name)
                            trade_side.append("long")

                            position = int(balance / row.Open)
                            cost = position * row.Open
                            balance = balance - cost

                            c = 0

                        elif row.SHORT and last_signal != "short":
                            last_signal = "short"
                            last_price = row.Open
                            trade_date_start.append(row.name)
                            trade_side.append("short")

                            position = int(balance / row.Open) * -1

                            c = 0

                        if stop_loss_lvl:
                            # check stop loss
                            if (
                                last_signal == "long"
                                and (row.Low / last_price - 1) * 100 <= stop_loss_lvl
                            ):
                                c = c + 1

                                trade_date_end.append(row.name)
                                trade_days.append(c)

                                stop_loss_price = last_price + round(
                                    last_price * (stop_loss_lvl / 100), 4
                                )

                                pnl = (stop_loss_price - last_price) * position
                                trade_pnl.append(pnl)
                                trade_ret.append((stop_loss_price / last_price - 1) * 100)

                                balance = balance + stop_loss_price * position

                                position = 0
                                last_signal = "hold"

                                c = 0

                            elif (
                                last_signal == "short"
                                and (last_price / row.Low - 1) * 100 <= stop_loss_lvl
                            ):
                                c = c + 1

                                trade_date_end.append(row.name)
                                trade_days.append(c)

                                stop_loss_price = last_price - round(
                                    last_price * (stop_loss_lvl / 100), 4
                                )

                                pnl = (stop_loss_price - last_price) * position
                                trade_pnl.append(pnl)
                                trade_ret.append((last_price / stop_loss_price - 1) * 100)

                                balance = balance + pnl

                                position = 0
                                last_signal = "hold"

                                c = 0

                        # compute market value and count days for any possible poisition
                        if last_signal == "hold":
                            market_value = balance
                        elif last_signal == "long":
                            c = c + 1
                            market_value = position * row.Close + balance
                        else:
                            c = c + 1
                            market_value = (row.Close - last_price) * position + balance

                        cum_value.append(market_value)

                    # generate analysis
                    # performance over time
                    cum_ret_df = pd.DataFrame(cum_value, index=bt_df.index, columns=["CUM_RET"])
                    cum_ret_df["CUM_RET"] = (cum_ret_df.CUM_RET / 1000000 - 1) * 100
                    cum_ret_df["BUY_HOLD"] = (bt_df.Close / bt_df.Open.iloc[0] - 1) * 100
                    cum_ret_df["ZERO"] = 0

                    # trade stats
                    size = min(len(trade_date_start), len(trade_date_end))

                    tarde_dict = {
                        "START": trade_date_start[:size],
                        "END": trade_date_end[:size],
                        "SIDE": trade_side[:size],
                        "DAYS": trade_days[:size],
                        "PNL": trade_pnl[:size],
                        "RET": trade_ret[:size],
                    }

                    trade_df = pd.DataFrame(tarde_dict)

                    num_trades = trade_df.groupby("SIDE").count()[["START"]]
                    num_trades_win = trade_df[trade_df.PNL > 0].groupby("SIDE").count()[["START"]]

                    avg_days = trade_df.groupby("SIDE").mean()[["DAYS"]]

                    avg_ret = trade_df.groupby("SIDE").mean()[["RET"]]
                    avg_ret_win = trade_df[trade_df.PNL > 0].groupby("SIDE").mean()[["RET"]]
                    avg_ret_loss = trade_df[trade_df.PNL < 0].groupby("SIDE").mean()[["RET"]]

                    std_ret = trade_df.groupby("SIDE").std()[["RET"]]

                    detail_df = pd.concat(
                        [
                            num_trades,
                            num_trades_win,
                            avg_days,
                            avg_ret,
                            avg_ret_win,
                            avg_ret_loss,
                            std_ret,
                        ],
                        axis=1,
                        sort=False,
                    )

                    detail_df.columns = [
                        "NUM_TRADES",
                        "NUM_TRADES_WIN",
                        "AVG_DAYS",
                        "AVG_RET",
                        "AVG_RET_WIN",
                        "AVG_RET_LOSS",
                        "STD_RET",
                    ]

                    detail_df.round(2)

                    # max drawdown
                    mv_df = pd.DataFrame(cum_value, index=bt_df.index, columns=["MV"])

                    days = len(mv_df)

                    roll_max = mv_df.MV.rolling(window=days, min_periods=1).max()
                    drawdown_val = mv_df.MV - roll_max
                    drawdown_pct = (mv_df.MV / roll_max - 1) * 100

                    # return all stats
                    return {
                        "cum_ret_df": cum_ret_df,
                        "max_drawdown": {
                            "value": round(drawdown_val.min(), 0),
                            "pct": round(drawdown_pct.min(), 2),
                        },
                        "trade_stats": detail_df,
                    }

                result = run_stock_ta_backtest(bt_df)

                result["cum_ret_df"].plot(figsize=(15, 5))

                st.write("Max Drawdown:", result["max_drawdown"]["pct"], "%")

                result["trade_stats"]

                ticker = ticker
                start_date = start_date
                end_date = end_date

                df = get_stock_backtest_data(ticker, start_date, end_date)

                bt_df = prepare_stock_ta_backtest_data(
                    df, start_date, end_date, strategy_KeltnerChannel_origin, n=10
                )

                result = run_stock_ta_backtest(bt_df)

                result["cum_ret_df"].plot(figsize=(15, 5))
                st.write("Max Drawdown:", result["max_drawdown"]["pct"], "%")
                result["trade_stats"]

                ticker = ticker
                start_date = start_date
                end_date = end_date

                df = get_stock_backtest_data(ticker, start_date, end_date)

                n_list = [i for i in range(10, 30, 5)]
                stop_loss_lvl = [-i for i in range(2, 5, 1)]
                stop_loss_lvl.append(None)

                result_dict = {"n": [], "l": [], "return": [], "max_drawdown": []}

                for n in n_list:
                    for l in stop_loss_lvl:
                        bt_df = prepare_stock_ta_backtest_data(
                            df, start_date, end_date, strategy_KeltnerChannel_origin, n=n
                        )

                        result = run_stock_ta_backtest(bt_df, stop_loss_lvl=l)

                        result_dict["n"].append(n)
                        result_dict["l"].append(l)
                        result_dict["return"].append(result["cum_ret_df"].iloc[-1, 0])
                        result_dict["max_drawdown"].append(result["max_drawdown"]["pct"])

                df = pd.DataFrame(result_dict)
                st.write(df.sort_values("return", ascending=False))

                from itertools import product

                a = [5, 10]
                b = [1, 3]
                c = [2, 4]

                list(product(a, b, c))
                param_list = [a, b, c]

                list(product(*param_list))

                def test_func(**kwargs):
                    a = kwargs.get("a", 10)
                    b = kwargs.get("b", 2)
                    c = kwargs.get("c", 2)

                    st.write(a, b, c)

                test_func(a=1, b=2, c=3)

                param_dict = {"a": 1, "b": 2, "c": 3}

                test_func(**param_dict)

                param_name = ["a", "b", "c"]
                param = [1, 2, 3]

                dict(zip(param_name, param))
                dict(zip(["a", "b", "c"], [1, 2, 3]))

                a = [5, 10]
                b = [1, 3]
                c = [2, 4]

                param_list = [a, b, c]
                param_name = ["a", "b", "c"]
                param_dict_list = [dict(zip(param_name, param)) for param in list(product(*param_list))]
                param_dict_list

                strategies = [
                    {
                        "func": strategy_KeltnerChannel_origin,
                        "param": {"n": [i for i in range(10, 35, 5)]},
                    },
                    {
                        "func": strategy_BollingerBands,
                        "param": {"n": [i for i in range(10, 35, 5)], "n_rng": [1, 2]},
                    },
                    {
                        "func": strategy_MA,
                        "param": {"n": [i for i in range(10, 110, 10)], "ma_type": ["sma", "ema"]},
                    },
                    {
                        "func": strategy_MACD,
                        "param": {
                            "n_slow": [i for i in range(10, 16)],
                            "n_fast": [i for i in range(20, 26)],
                            "n_sign": [i for i in range(5, 11)],
                        },
                    },
                    {"func": strategy_RSI, "param": {"n": [i for i in range(5, 21)]}},
                    {"func": strategy_WR, "param": {"n": [i for i in range(5, 21)]}},
                    {
                        "func": strategy_Stochastic_fast,
                        "param": {"k": [i for i in range(15, 26)], "d": [i for i in range(5, 11)]},
                    },
                    {
                        "func": strategy_Stochastic_slow,
                        "param": {
                            "k": [i for i in range(15, 26)],
                            "d": [i for i in range(5, 11)],
                            "dd": [i for i in range(1, 6)],
                        },
                    },
                    {
                        "func": strategy_Ichmoku,
                        "param": {
                            "n_conv": [i for i in range(5, 16)],
                            "n_base": [i for i in range(20, 36)],
                            "n_span_b": [26],
                        },
                    },
                ]

                for s in strategies:
                    func = s["func"]
                    param = s["param"]

                    param_name = []
                    param_list = []

                    for k in param:
                        param_name.append(k)
                        param_list.append(param[k])

                    param_dict_list = [
                        dict(zip(param_name, param)) for param in list(product(*param_list))
                    ]

                    st.write(len(param_dict_list))

                ticker = ticker
                start_date = start_date
                end_date = end_date

                df = get_stock_backtest_data(ticker, start_date, end_date)

                stop_loss_lvl = [-i for i in range(2, 6, 1)]
                stop_loss_lvl.append(None)

                result_dict = {
                    "strategy": [],
                    "param": [],
                    "stoploss": [],
                    "return": [],
                    "max_drawdown": [],
                }

                for s in strategies:
                    func = s["func"]
                    param = s["param"]

                    strategy_name = str(func).split(" ")[1]

                    param_name = []
                    param_list = []

                    for k in param:
                        param_name.append(k)
                        param_list.append(param[k])

                    param_dict_list = [
                        dict(zip(param_name, param)) for param in list(product(*param_list))
                    ]
                    total_param_dict = len(param_dict_list)

                    c = 0

                    for param_dict in param_dict_list:
                        clear_output()
                        c = c + 1
                        st.write(
                            "Running backtest for {} - ({}/{})".format(
                                strategy_name, c, total_param_dict
                            )
                        )

                        for l in stop_loss_lvl:
                            bt_df = prepare_stock_ta_backtest_data(
                                df, start_date, end_date, func, **param_dict
                            )

                            result = run_stock_ta_backtest(bt_df, stop_loss_lvl=l)

                            result_dict["strategy"].append(strategy_name)
                            result_dict["param"].append(str(param_dict))
                            result_dict["stoploss"].append(l)
                            result_dict["return"].append(result["cum_ret_df"].iloc[-1, 0])
                            result_dict["max_drawdown"].append(result["max_drawdown"]["pct"])

                df = pd.DataFrame(result_dict)
                df.to_csv(f"{ticker}_{start_date}_{end_date}_backtest.csv")
                st.write(df.sort_values("return", ascending=True).head(50))
                st.write(df.sort_values("return", ascending=False).head(50))

        if pred_option_analysis == "CAPM Analysis": 
            #TODO fix the CAPM analysis quagmire of returning empty datasets from the NASDAQ index
            st.success("This segment allows you to do CAPM Analysis")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            min_date = datetime(1980, 1, 1)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):

            # Fetches stock data for a given ticker.
                def get_stock_data(ticker, start, end):
                    return pdr.DataReader(ticker, start_date, end_date)

                # Calculates expected return using CAPM.                
                def calculate_expected_return(stock, index, risk_free_return):
                    # Check if the index is DateTimeIndex, if not, convert it
                    if not isinstance(stock.index, pd.DatetimeIndex):
                        stock.index = pd.to_datetime(stock.index)
                    if not isinstance(index.index, pd.DatetimeIndex):
                        index.index = pd.to_datetime(index.index)
                    
                    # Resample to monthly data
                    return_stock = stock.resample('M').last()['Adj Close']
                    return_index = index.resample('M').last()['Adj Close']

                    # Create DataFrame with returns
                    df = pd.DataFrame({'stock_close': return_stock, 'index_close': return_index})
                    df[['stock_return', 'index_return']] = np.log(df / df.shift(1))
                    df = df.dropna()

                    # Check if df contains non-empty vectors
                    if len(df['index_return']) == 0 or len(df['stock_return']) == 0:
                        raise ValueError("Empty vectors found in DataFrame df")

                    # Calculate beta and alpha
                    beta, alpha = np.polyfit(df['index_return'], df['stock_return'], deg=1)
                    
                    # Calculate expected return
                    expected_return = risk_free_return + beta * (df['index_return'].mean() * 12 - risk_free_return)
                    return expected_return
                
                # Risk-free return rate
                risk_free_return = 0.02

                # Define time period
                start = start_date
                end = end_date
                
                # Get all tickers in NASDAQ
                #nasdaq_tickers = ti.tickers_nasdaq()
                sp_500 =  ti.tickers_sp500()

                # Index ticker
                index_ticker = '^GSPC'

                # Fetch index data
                try:
                    index_data = get_stock_data(index_ticker, start, end)
                except RemoteDataError:
                    st.write("Failed to fetch index data.")
                    return

                # Loop through NASDAQ tickers
                for ticker in sp_500:
                    try:
                        # Fetch stock data
                        stock_data = get_stock_data(ticker, start, end)

                        # Calculate expected return
                        expected_return = calculate_expected_return(stock_data, index_data, risk_free_return)

                        # Output expected return
                        st.write(f'{ticker}: Expected Return: {expected_return}')

                    except (RemoteDataError, gaierror):
                        st.write(f"Data not available for ticker: {ticker}")


        if pred_option_analysis == "Earnings Sentiment Analysis":
            # Retrieves earnings call transcript from API
            # TODO identify the error with the API with key error 0
            def get_earnings_call_transcript(api_key, company, quarter, year):
                url = f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{company}?quarter={quarter}&year={year}&apikey={api_key}'
                response = requests.get(url)
                return response.json()[0]['content']

            # Performs sentiment analysis on the transcript.
            def analyze_sentiment(transcript):
                sentiment_call = TextBlob(transcript)
                return sentiment_call

            # Counts the number of positive, negative, and neutral sentences.
            def count_sentiments(sentiment_call):
                positive, negative, neutral = 0, 0, 0
                all_sentences = []

                for sentence in sentiment_call.sentences:
                    polarity = sentence.sentiment.polarity
                    if polarity < 0:
                        negative += 1
                    elif polarity > 0:
                        positive += 1
                    else:
                        neutral += 1
                    all_sentences.append(polarity)

                return positive, negative, neutral, np.array(all_sentences)

            st.success("This segment allows us to get The sentiments of a company's earnings")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):
                api_key = API_FMPCLOUD
                company = ticker

                # Get transcript and perform sentiment analysis
                transcript = get_earnings_call_transcript(api_key, company, 3, 2020)
                sentiment_call = analyze_sentiment(transcript)

                # Count sentiments and calculate mean polarity
                positive, negative, neutral, all_sentences = count_sentiments(sentiment_call)
                mean_polarity = all_sentences.mean()

                # Print results
                st.write(f"Earnings Call Transcript for {company}:\n{transcript}\n")
                st.write(f"Overall Sentiment: {sentiment_call.sentiment}")
                st.write(f"Positive Sentences: {positive}, Negative Sentences: {negative}, Neutral Sentences: {neutral}")
                st.write(f"Average Sentence Polarity: {mean_polarity}")

                # Print very positive sentences
                print("\nHighly Positive Sentences:")
                for sentence in sentiment_call.sentences:
                    if sentence.sentiment.polarity > 0.8:
                        st.write(sentence)

        if pred_option_analysis == "Estimating Returns":

            # Fetches stock data and calculates annual returns.
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
            st.success("This segment allows us to get The returns of a company and see whether it would be between 2 and 3 for a specified period")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):
                stock_ticker = ticker
                higher_bound, lower_bound = 0.3, 0.2


                # Retrieve and process stock data
                df_returns = get_stock_returns(stock_ticker, start_date, end_date)
                plot_return_distribution(df_returns['Return'], stock_ticker)

                # Estimate probability
                prob = estimate_return_probability(df_returns['Return'], lower_bound, higher_bound)
                st.write(f'The probability of returns falling between {lower_bound} and {higher_bound} for {stock_ticker.upper()} is: {prob}')
                
        if pred_option_analysis == "Kelly Criterion":
            st.success("This segment allows us to determine the optimal size of our investment based on a series of bets or investments in order to maximize long-term growth of capital")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):
                # Define stock symbol and time frame for analysis
                symbol = ticker

                # Download stock data using yfinance package
                stock_data = yf.download(symbol, start=start_date, end=end_date)

                # Calculate daily returns and drop rows with missing data
                stock_data['Returns'] = stock_data['Adj Close'].pct_change()
                stock_data.dropna(inplace=True)

                # Display the first few rows of the data for verification
                st.write(stock_data.head())

                # Calculate Kelly Criterion
                # Extract positive (wins) and negative (losses) returns
                wins = stock_data['Returns'][stock_data['Returns'] > 0]
                losses = stock_data['Returns'][stock_data['Returns'] <= 0]

                # Calculate win ratio and win-loss ratio
                win_ratio = len(wins) / len(stock_data['Returns'])
                win_loss_ratio = np.mean(wins) / np.abs(np.mean(losses))

                # Apply Kelly Criterion formula
                kelly_criterion = win_ratio - ((1 - win_ratio) / win_loss_ratio)

                # Print the Kelly Criterion percentage
                st.write('Kelly Criterion: {:.3f}%'.format(kelly_criterion * 100))

        if pred_option_analysis == "Intrinsic Value analysis":
            st.success("This segment allows us to determine the optimal size of our investment based on a series of bets or investments in order to maximize long-term growth of capital")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):
            # Set options for pandas display
                pd.set_option('float_format', '{:f}'.format)

                # API and stock configuration
                base_url = "https://financialmodelingprep.com/api/v3/"
                apiKey = "demo"  # Note: Demo API only works for AAPL stock
                ticker = 'AAPL'
                current_price = pdr.get_data_yahoo(ticker)['Adj Close'][-1]

                # Function to retrieve JSON data from URL
                def json_data(url):
                    response = urlopen(url)
                    data = response.read().decode("utf-8")
                    return json.loads(data)

                # Retrieve financial statements
                def get_financial_statements():
                    # Income statement
                    income = pd.DataFrame(json_data(f'{base_url}income-statement/{ticker}?apikey={apiKey}'))
                    income = income.set_index('date').apply(pd.to_numeric, errors='coerce')

                    # Cash flow statement
                    cash_flow = pd.DataFrame(json_data(f'{base_url}cash-flow-statement/{ticker}?apikey={apiKey}'))
                    cash_flow = cash_flow.set_index('date').apply(pd.to_numeric, errors='coerce')

                    # Balance sheet
                    balance_sheet = pd.DataFrame(json_data(f'{base_url}balance-sheet-statement/{ticker}?apikey={apiKey}'))
                    balance_sheet = balance_sheet.set_index('date').apply(pd.to_numeric, errors='coerce')

                    return income, cash_flow, balance_sheet

                income, cash_flow, balance_sheet = get_financial_statements()

                # Retrieve and process metrics from finviz
                def get_finviz_data(ticker):
                    url = f"http://finviz.com/quote.ashx?t={ticker}"
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    soup = BeautifulSoup(requests.get(url, headers=headers).content, features="lxml")

                    metrics = ['Beta', 'EPS next 5Y', 'Shs Outstand']
                    finviz_dict = {}
                    for m in metrics:   
                        finviz_dict[m] = soup.find(text=m).find_next(class_='snapshot-td2').text

                    # Process and convert metrics to appropriate formats
                    for key, value in finviz_dict.items():
                        if value[-1] in ['%', 'B', 'M']:
                            value = float(value[:-1])
                            if value[-1] == 'B':
                                value *= 1e9
                            elif value[-1] == 'M':
                                value *= 1e6
                        finviz_dict[key] = float(value)
                    return finviz_dict

                finviz_data = get_finviz_data(ticker)
                beta = finviz_data['Beta']

                # Determine the discount rate based on beta
                discount = 7 + beta * 2.5
                if beta < 1:
                    discount = 6

                # Calculate intrinsic value
                def calc_intrinsic_value(cash_flow, total_debt, liquid_assets, eps_growth_5Y, eps_growth_6Y_to_10Y, eps_growth_11Y_to_20Y, shs_outstanding, discount):   
                    eps_growth_5Y /= 100
                    eps_growth_6Y_to_10Y /= 100
                    eps_growth_11Y_to_20Y /= 100
                    discount /= 100

                    cf_list = []
                    for year in range(1, 21):
                        growth_rate = eps_growth_5Y if year <= 5 else eps_growth_6Y_to_10Y if year <= 10 else eps_growth_11Y_to_20Y
                        cash_flow *= (1 + growth_rate)
                        discounted_cf = cash_flow / ((1 + discount)**year)
                        cf_list.append(discounted_cf)

                    intrinsic_value = (sum(cf_list) - total_debt + liquid_assets) / shs_outstanding
                    return intrinsic_value

                intrinsic_value = calc_intrinsic_value(cash_flow.iloc[-1]['freeCashFlow'],
                                                    balance_sheet.iloc[-1]['totalDebt'],
                                                    balance_sheet.iloc[-1]['cashAndShortTermInvestments'],
                                                    finviz_data['EPS next 5Y'],
                                                    finviz_data['EPS next 5Y'] / 2,
                                                    np.minimum(finviz_data['EPS next 5Y'] / 2, 4),
                                                    finviz_data['Shs Outstand'],
                                                    discount)

                # Calculate deviation from intrinsic value
                percent_from_intrinsic_value = round((1 - current_price / intrinsic_value) * 100, 2)

                # Display data in a DataFrame
                data = {
                    'Attributes': ['Intrinsic Value', 'Current Price', 'Intrinsic Value % from Price', 'Free Cash Flow', 'Total Debt', 'Cash and ST Investments', 'EPS Growth 5Y', 'EPS Growth 6Y to 10Y', 'EPS Growth 11Y to 20Y', 'Discount Rate', 'Shares Outstanding'],
                    'Values': [intrinsic_value, current_price, percent_from_intrinsic_value, cash_flow.iloc[-1]['freeCashFlow'], balance_sheet.iloc[-1]['totalDebt'], balance_sheet.iloc[-1]['cashAndShortTermInvestments'], finviz_data['EPS next 5Y'], finviz_data['EPS next 5Y'] / 2, np.minimum(finviz_data['EPS next 5Y'] / 2, 4), discount, finviz_data['Shs Outstand']]
                }
                df = pd.DataFrame(data).set_index('Attributes')
                st.write(df)

        if pred_option_analysis == "MA Backtesting":
            st.success("This segment allows us to determine the optimal size of our investment based on a series of bets or investments in order to maximize long-term growth of capital")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            initial_capital = st.text_input("Enter the initial amount you want to backtest with")
            if initial_capital:
                message_four = (f"Amount captured : {initial_capital}")
                st.success(message_four)
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):
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

        if pred_option_analysis == "Ols Regression":
            st.success("This segment allows us to determine the optimal size of our investment based on a series of bets or investments in order to maximize long-term growth of capital")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            
            if st.button("Check"):
                # Configure the stock symbol, start, and end dates for data
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

        if pred_option_analysis == "Perfomance Risk Analysis":
            st.success("This segment allows us to determine the perfomance risk of a ticker against S&P 500 using Alpha, beta and R squared")
            ticker = st.text_input("Enter the ticker you want to test")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):
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
            
        if pred_option_analysis == "Risk/Returns Analysis":

            st.success("This code allows us to see the contextual risk of related tickers")
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            # Replace this with your actual sectors and tickers
            sectors = {
                "Technology": ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'INTC', 'CSCO', 'ADBE', 'AVGO', 'PYPL'],
                "Health Care": ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'MDT', 'DHR', 'AMGN', 'LLY'],
                "Financials": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'USB'],
                "Consumer Discretionary": ['AMZN', 'TSLA', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'BKNG', 'LOW', 'CMG'],
                "Communication Services": ['GOOGL', 'META', 'DIS', 'CMCSA', 'NFLX', 'T', 'CHTR', 'DISCA', 'FOXA', 'VZ'],
                "Industrials": ['BA', 'HON', 'UNP', 'UPS', 'CAT', 'LMT', 'MMM', 'GD', 'GE', 'CSX'],
                "Consumer Staples": ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'MO', 'EL', 'KHC', 'MDLZ'],
                "Utilities": ['NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'PEG', 'XEL', 'ED'],
                "Real Estate": ['AMT', 'PLD', 'CCI', 'EQIX', 'WY', 'PSA', 'DLR', 'BXP', 'O', 'SBAC'],
                "Materials": ['LIN', 'APD', 'SHW', 'DD', 'ECL', 'DOW', 'NEM', 'PPG', 'VMC', 'FCX']
            }

            pred_option_analysis = st.selectbox("Select Analysis", ["Risk/Returns Analysis"])
            selected_sector = st.selectbox('Select Sector', list(sectors.keys()))
            if st.button("Start Analysis"):
                # Downloading and processing stock data
                df = pd.DataFrame()
                for symbol in sectors[selected_sector]:
                    df[symbol] = yf.download(symbol, start_date, end_date)['Adj Close']
                # Dropping rows with missing values
                df = df.dropna()
                # Calculating percentage change in stock prices
                rets = df.pct_change(periods=3)

                # Creating correlation matrix heatmap
                corr = rets.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.index,
                    y=corr.columns,
                    colorscale='Blues'
                ))
                fig.update_layout(
                    title="Correlation Matrix Heatmap",
                    xaxis_title="Stock Symbols",
                    yaxis_title="Stock Symbols"
                )
                st.plotly_chart(fig)

                # Plotting bar charts for risk and average returns
                fig = go.Figure()
                fig.add_trace(go.Bar(x=rets.columns, y=rets.std(), name='Risk (Std. Dev.)', marker_color='red'))
                fig.add_trace(go.Bar(x=rets.columns, y=rets.mean(), name='Average Returns', marker_color='blue'))
                fig.update_layout(
                    title="Risk and Average Returns",
                    xaxis_title="Stock Symbols",
                    yaxis_title="Value",
                    barmode='group'
                )
                st.plotly_chart(fig)

                # Stacked bar chart for risk vs return
                fig = go.Figure()
                for i, symbol in enumerate(sectors[selected_sector]):
                    fig.add_trace(go.Bar(x=[symbol], y=[rets.mean()[i]], name='Average of Returns', marker_color='blue'))
                    fig.add_trace(go.Bar(x=[symbol], y=[rets.std()[i]], name='Risk of Returns', marker_color='red'))
                fig.update_layout(
                    title='Risk vs Average Returns',
                    xaxis_title='Stock Symbols',
                    yaxis_title='Value',
                    barmode='stack'
                )
                st.plotly_chart(fig)

                # Scatter plot for expected returns vs risk
                fig = go.Figure()
                for i in range(len(rets.columns)):
                    fig.add_trace(go.Scatter(x=rets.mean(), y=rets.std(), mode='markers', text=rets.columns))
                    fig.update_layout(
                        title='Risk vs Expected Returns',
                        xaxis_title='Expected Returns',
                        yaxis_title='Risk'
                    )
                    st.plotly_chart(fig)

                # Display table with risk vs expected returns
                risk_returns_table = pd.DataFrame({'Risk': rets.std(), 'Expected Returns': rets.mean()})
                st.write("Table: Risk vs Expected Returns")
                st.write(risk_returns_table)
        if pred_option_analysis == "Seasonal Stock Analysis":
            st.write("This segment allows us to analyze the seasonal returns of s&p 500")
            if st.button("Analyze"):
                # Scrape a list of the S&P 500 components using your ticker function
                sp500_tickers = ti.tickers_sp500()
                threshold = 0.80
                # Upload a list of the S&P 500 components downloaded from Wikipedia using your ticker function
                df_sp500_tickers = pd.DataFrame(sp500_tickers, columns=["Symbol"])

                # Loops through the S&P 500 tickers, downloads the data from Yahoo and creates a DataFrame for each ticker.
                dfs = []
                for ticker in ti.tickers_sp500():
                    yf_ticker = yf.Ticker(ticker)
                    data = yf_ticker.history(period="max")
                    data.reset_index(inplace=True)
                    data.drop_duplicates(subset="Date", keep="first", inplace=True)
                    data['Symbol'] = ticker
                    dfs.append(data)

                # Concatenate all DataFrames into a single DataFrame
                df_combined = pd.concat(dfs, ignore_index=True)

                # Creates the dataframe container for the stats data.
                df_tradelist = pd.DataFrame(
                    index=[],
                    columns=[
                        "my_ticker",
                        "hold_per",
                        "pct_uprows",
                        "max_up_return",
                        "min_up_return",
                        "avg_up_return",
                        "avg_down_return",
                        "exp_return",
                        "stdev_returns",
                        "pct_downside",
                        "worst_return",
                        "least_pain_pt",
                        "total_years",
                        "max_consec_beat",
                        "best_buy_date",
                        "best_sell_date",
                        "analyzed_years",
                    ],
                )
                df_tradelist.head()

                # Separate out the date column into separate month, year and day values.
                def separate_date_column():
                    global dfr
                    dfr["Month"] = pd.DatetimeIndex(dfr["Date"]).month
                    dfr["Day"] = pd.DatetimeIndex(dfr["Date"]).day
                    dfr["Year"] = pd.DatetimeIndex(dfr["Date"]).year
                    dfr["M-D"] = dfr["Month"].astype(str) + "-" + dfr["Day"].astype(str)
                    pd.set_option("display.max_rows", len(dfr))

                # Pivot the table to show years across the top and Month-Day values in the first column on the left.
                def pivot_the_table():
                    global dfr_pivot
                    dfr_pivot = dfr.pivot(index="M-D", columns="Year", values="Returns")
                    dfr_pivot.reset_index(level=0, inplace=True)
                    dfr_pivot = pd.DataFrame(dfr_pivot)
                    dfr_pivot.columns.name = "Index"
                    dfr_pivot.fillna(method="ffill", inplace=True)

                # Add additional calculated columns to facilitate statistic calculations for each stock.
                def add_calculated_items():
                    global dfr_pivot
                    global lookback
                    global start

                    # The lookback figure is the number (must be an integer) of years back from last year that you want to include
                    lookback = 20
                    start = 1
                    if lookback > len(dfr_pivot.columns) - 1:
                        start = 1
                    else:
                        start = len(dfr_pivot.columns) - lookback
                    dfr_pivot["YearCount"] = dfr_pivot.count(axis=1, numeric_only=True)
                    dfr_pivot["Lookback"] = lookback
                    dfr_pivot["UpCount"] = dfr_pivot[
                        dfr_pivot.iloc[:, start: len(dfr_pivot.columns) - 2] > 0
                    ].count(axis=1)
                    dfr_pivot["DownCount"] = dfr_pivot[
                        dfr_pivot.iloc[:, start: len(dfr_pivot.columns)] < 0
                    ].count(axis=1)
                    dfr_pivot["PctUp"] = dfr_pivot["UpCount"] / dfr_pivot["Lookback"]
                    dfr_pivot["PctDown"] = dfr_pivot["DownCount"] / dfr_pivot["Lookback"]
                    dfr_pivot["AvgReturn"] = dfr_pivot.iloc[:, start: len(dfr_pivot.columns) - 6].mean(
                        axis=1
                    )
                    dfr_pivot["StDevReturns"] = dfr_pivot.iloc[
                        :, start: len(dfr_pivot.columns) - 7
                    ].std(axis=1)
                    dfr_pivot["67PctDownside"] = dfr_pivot["AvgReturn"] - dfr_pivot["StDevReturns"]
                    dfr_pivot["MaxReturn"] = dfr_pivot.iloc[:, start: len(dfr_pivot.columns) - 9].max(
                        axis=1
                    )
                    dfr_pivot["MinReturn"] = dfr_pivot.iloc[:, start: len(dfr_pivot.columns) - 10].min(
                        axis=1
                    )

                # Calculate the trading statistics for the rolling holding periods for the stock.
                def calc_trading_stats():
                    global interval
                    global dfr_pivot
                    global pct_uprows
                    global max_up_return
                    global min_up_return
                    global avg_up_return
                    global avg_down_return
                    global exp_return
                    global stdev_returns
                    global pct_downside
                    global worst_return
                    global least_pain_pt
                    global total_years
                    global n_consec
                    global max_n_consec
                    global max_consec_beat
                    global best_sell_date
                    global best_buy_date
                    global analyzed_years
                    global lookback

                    pct_uprows = (
                        (
                            dfr_pivot.loc[dfr_pivot["PctUp"] > threshold, "PctUp"].count()
                            / dfr_pivot.loc[:, "PctUp"].count()
                        )
                        .astype(float)
                        .round(4)
                    )
                    max_up_return = dfr_pivot.loc[dfr_pivot["PctUp"] > threshold, "MaxReturn"].max()
                    min_up_return = dfr_pivot.loc[dfr_pivot["PctUp"] > threshold, "MinReturn"].min()
                    avg_up_return = dfr_pivot.loc[dfr_pivot["PctUp"] > 0.5, "AvgReturn"].mean()
                    avg_up_return = np.float64(avg_up_return).round(4)
                    avg_down_return = dfr_pivot.loc[dfr_pivot["PctDown"] > 0.5, "AvgReturn"].mean()
                    avg_down_return = np.float64(avg_down_return).round(4)
                    exp_return = round(dfr_pivot["AvgReturn"].mean(), 4)
                    stdev_returns = dfr_pivot["StDevReturns"].mean()
                    stdev_returns = np.float64(stdev_returns).round(4)
                    worst_return = dfr_pivot["MinReturn"].min()
                    pct_downside = exp_return - stdev_returns
                    pct_downside = np.float64(pct_downside).round(4)
                    least_pain_pt = dfr_pivot.loc[dfr_pivot["PctUp"] > threshold, "67PctDownside"].max()
                    total_years = dfr_pivot["YearCount"].max()
                    analyzed_years = lookback
                    n_consec = 0
                    max_n_consec = 0

                    for x in dfr_pivot["PctUp"]:
                        if x > threshold:
                            n_consec += 1
                        else:  # check for new max, then start again from 1
                            max_n_consec = max(n_consec, max_n_consec)
                            n_consec = 1
                    max_consec_beat = max_n_consec
                    try:
                        best_sell_date = dfr_pivot.loc[
                            dfr_pivot["67PctDownside"] == least_pain_pt, "M-D"
                        ].iloc[0]
                    except:
                        best_sell_date = "nan"
                    try:
                        row = (
                            dfr_pivot.loc[dfr_pivot["M-D"] == best_sell_date, "M-D"].index[0] - interval
                        )
                        col = dfr_pivot.columns.get_loc("M-D")
                        best_buy_date = dfr_pivot.iloc[row, col]
                    except:
                        best_buy_date = "nan"

                # Convert prices to holding period returns based on a specified number of trading days.
                def convert_prices_to_periods():
                    global dfr
                    global dperiods
                    dfr = df.pct_change(periods=dperiods)
                    dfr.reset_index(level=0, inplace=True)
                    dfr.rename(columns={"Close": "Returns"}, inplace=True)
                    dfr = dfr.round(4)
                # Reset the index and round the float values to 4 decimals.
                def sortbydate_resetindex_export():
                    global dfr_pivot
                    dfr_pivot["Date"] = "2000-" + dfr_pivot["M-D"].astype(str)
                    dfr_pivot["Date"] = pd.to_datetime(dfr_pivot["Date"], infer_datetime_format=True)
                    dfr_pivot.sort_values(by="Date", ascending=True, inplace=True)
                    dfr_pivot.reset_index(inplace=True)
                    dfr_pivot = dfr_pivot.round(4)
                # This module grabs each ticker file, transforms it, and calculates the statistics needed for a 90-day holding period.
                def calc_3month_returns():
                    global dfr
                    global dfr_pivot
                    global df_tradelist
                    global dfr_3mo
                    global df_statsdata_3mo
                    global threshold
                    global hold_per
                    global dperiods
                    global interval
                    dperiods = 60
                    hold_per = "3 Mos"
                    interval = 90
                    convert_prices_to_periods()
                    separate_date_column()
                    pivot_the_table()
                    add_calculated_items()
                    sortbydate_resetindex_export()

                    dfr_3mo = pd.DataFrame(dfr_pivot)
                    calc_trading_stats()
                    filter_and_append_stats()

                    df_statsdata_3mo = df_statsdata.copy()


                # This module grabs each ticker file, transforms it, and calculates the statistics needed for a 60-day holding period.
                def calc_2month_returns():
                    global dfr
                    global dfr_pivot
                    global df_tradelist
                    global dfr_2mo
                    global df_statsdata_2mo
                    global threshold
                    global hold_per
                    global dperiods
                    global interval
                    dperiods = 40
                    hold_per = "2 Mos"
                    interval = 60
                    convert_prices_to_periods()
                    separate_date_column()
                    pivot_the_table()
                    add_calculated_items()
                    sortbydate_resetindex_export()

                    dfr_2mo = pd.DataFrame(dfr_pivot)
                    calc_trading_stats()
                    filter_and_append_stats()

                    df_statsdata_2mo = df_statsdata.copy()


                # This module grabs each ticker file, transforms it, and calculates the statistics needed for a 30-day holding period.
                def calc_1month_returns():
                    global dfr
                    global dfr_pivot
                    global df_tradelist
                    global dfr_1mo
                    global df_statsdata_1mo
                    global threshold
                    global hold_per
                    global dperiods
                    global interval
                    dperiods = 20
                    hold_per = "1 Mo"
                    interval = 30
                    convert_prices_to_periods(df)
                    separate_date_column()
                    pivot_the_table()
                    add_calculated_items()
                    sortbydate_resetindex_export()

                    dfr_1mo = pd.DataFrame(dfr_pivot)
                    calc_trading_stats()
                    filter_and_append_stats()

                    df_statsdata_1mo = df_statsdata.copy()

                # If the pct_uprows and history conditions are met, then create the array of stat values and append it to the recommended trade list.
                def filter_and_append_stats():
                    global statsdata
                    global df_statsdata
                    global df_tradelist

                    # Save the stats data separately to export to Excel for further research on each ticker if desired.
                    statsdata = np.array(
                        [
                            my_ticker,
                            hold_per,
                            pct_uprows,
                            max_up_return,
                            min_up_return,
                            avg_up_return,
                            avg_down_return,
                            exp_return,
                            stdev_returns,
                            pct_downside,
                            worst_return,
                            least_pain_pt,
                            total_years,
                            max_consec_beat,
                            best_buy_date,
                            best_sell_date,
                            analyzed_years,
                        ]
                    )
                    df_statsdata = pd.DataFrame(
                        statsdata.reshape(-1, len(statsdata)),
                        columns=[
                            "my_ticker",
                            "hold_per",
                            "pct_uprows",
                            "max_up_return",
                            "min_up_return",
                            "avg_up_return",
                            "avg_down_return",
                            "exp_return",
                            "stdev_returns",
                            "pct_downside",
                            "worst_return",
                            "least_pain_pt",
                            "total_years",
                            "max_consec_beat",
                            "best_buy_date",
                            "best_sell_date",
                            "analyzed_years",
                        ],
                    )
                    if pct_uprows > 0.1:
                        if total_years > 9:
                            df_tradelist = df_tradelist.append(
                                dict(zip(df_tradelist.columns, statsdata)), ignore_index=True
                            )

                for index, ticker in df_sp500_tickers.iterrows():
                    my_ticker = ticker["Symbol"]
                    calc_1month_returns()
                    calc_2month_returns()
                    calc_3month_returns()
                    filter_and_append_stats()

                # Make a copy and convert the trade list to a Pandas dataframe.
                df_tradelist = pd.DataFrame(df_tradelist)

                # Clean it up by removing rows with NaN's and infinity values and dropping duplicates.
                df_tradelist.replace("inf", np.nan, inplace=True)
                df_tradelist.dropna(inplace=True)
                df_tradelist = df_tradelist[~df_tradelist.max_up_return.str.contains("nan")]
                df_tradelist = df_tradelist[~df_tradelist.avg_down_return.str.contains("nan")]
                df_tradelist.sort_values(by=["pct_uprows"], ascending=False)
                df_tradelist.drop_duplicates(subset="my_ticker", keep="first", inplace=True)
                df_tradelist.tail(10)
                df_tradelist.head()


    
        if pred_option_analysis == "SMA Histogram":

            st.success("This segment allows you to see the SMA movement of the stock")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)

            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):
                stock = ticker
                # Fetch stock data
                df = yf.download(stock, start_date, end_date)

                # Calculate Simple Moving Average (SMA)
                sma = 50
                df['SMA' + str(sma)] = df['Adj Close'].rolling(window=sma).mean()
                
                # Calculate percentage change from SMA
                df['PC'] = ((df["Adj Close"] / df['SMA' + str(sma)]) - 1) * 100

                # Calculating statistics
                mean = df["PC"].mean()
                stdev = df["PC"].std()
                current = df["PC"].iloc[-1]
                yday = df["PC"].iloc[-2]

                # Histogram settings
                bins = np.arange(-100, 100, 1)

                # Plotting histogram
                fig = go.Figure()

                # Add histogram trace
                fig.add_trace(go.Histogram(x=df["PC"], histnorm='percent', nbinsx=len(bins), name='Count'))

                # Adding vertical lines for mean, std deviation, current and yesterday's percentage change
                for i in range(-3, 4):
                    fig.add_shape(
                        dict(type="line", x0=mean + i * stdev, y0=0, x1=mean + i * stdev, y1=100, line=dict(color="gray", dash="dash"),
                            opacity=0.5 + abs(i)/6)
                    )
                fig.add_shape(
                    dict(type="line", x0=current, y0=0, x1=current, y1=100, line=dict(color="red"), name='Today')
                )
                fig.add_shape(
                    dict(type="line", x0=yday, y0=0, x1=yday, y1=100, line=dict(color="blue"), name='Yesterday')
                )

                # Update layout
                fig.update_layout(
                    title=f"{stock} - % From {sma} SMA Histogram since {start_date.year}",
                    xaxis_title=f'Percent from {sma} SMA (bin size = 1)',
                    yaxis_title='Percentage of Total',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )

                st.plotly_chart(fig)

        if pred_option_analysis == "SP500 COT Sentiment Analysis":
            st.success("This segment allows us to look at COT Sentiment analysis of tickers for the past 1 or more years")
                # Function to download and extract COT file
            def download_and_extract_cot_file(url, file_name):
                # Download and extract COT file
                with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                with zipfile.ZipFile(file_name) as zf:
                    zf.extractall()

                # Download and process COT data for the last 5 years
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):
                this_year = end_date.year
                start_year = start_date.year
                frames = []
                for year in range(start_year, this_year):
                    #TODO - fix the link CFTC link
                    url = f'https://www.cftc.gov/files/dea/history/fut_fin_xls_{year}.zip'
                    download_and_extract_cot_file(url, f'{year}.zip')
                    os.rename('FinFutYY.xls', f'{year}.xls')

                    data = pd.read_excel(f'{year}.xls')
                    data = data.set_index('Report_Date_as_MM_DD_YYYY')
                    data.index = pd.to_datetime(data.index)
                    data = data[data['Market_and_Exchange_Names'] == 'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE']
                    frames.append(data)

                # Concatenate yearly data frames
                df = pd.concat(frames)
                df.to_csv('COT_sp500_data.csv')

                # Read data for plotting
                df = pd.read_csv('COT_sp500_data.csv', index_col=0)
                df.index = pd.to_datetime(df.index)

                # Plotting Line Chart
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Dealer_Long_All'], mode='lines', name='Dealer Long'))
                fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Lev_Money_Long_All'], mode='lines', name='Leveraged Long'))
                fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Dealer_Short_All'], mode='lines', name='Dealer Short'))
                fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Lev_Money_Short_All'], mode='lines', name='Leveraged Short'))
                fig1.update_layout(title='Net Positions - Line Chart', xaxis_title='Date', yaxis_title='Percentage')
                st.plotly_chart(fig1)

                # Box Plot
                fig2 = go.Figure()
                fig2.add_trace(go.Box(y=df['Pct_of_OI_Dealer_Long_All'], name='Dealer Long'))
                fig2.add_trace(go.Box(y=df['Pct_of_OI_Dealer_Short_All'], name='Dealer Short'))
                fig2.add_trace(go.Box(y=df['Pct_of_OI_Lev_Money_Long_All'], name='Leveraged Money Long'))
                fig2.add_trace(go.Box(y=df['Pct_of_OI_Lev_Money_Short_All'], name='Leveraged Money Short'))
                fig2.update_layout(title='Distribution of Open Interest by Category', yaxis_title='Percentage')
                st.plotly_chart(fig2)

                filtered_df = df.loc[start_date:end_date]
                # Plotting Line Chart
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Dealer_Long_All'], mode='lines', name='Dealer Long'))
                fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Lev_Money_Long_All'], mode='lines', name='Leveraged Long'))
                fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Dealer_Short_All'], mode='lines', name='Dealer Short'))
                fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Lev_Money_Short_All'], mode='lines', name='Leveraged Short'))
                fig3.update_layout(title='Net Positions - Line Chart', xaxis_title='Date', yaxis_title='Percentage')
                st.plotly_chart(fig3)

        if pred_option_analysis == "SP500 Valuation":
            if st.button("Check"):
                # Load the S&P 500 data
                #sp_df = pd.read_excel("http://www.stern.nyu.edu/~adamodar/pc/datasets/spearn.xls", sheet_name="Sheet1")
                sp_df = []
                for ticker in ti.tickers_sp500():
                    yf_ticker = yf.Ticker(ticker)
                    data = yf_ticker.history(period="max")
                    data.reset_index(inplace=True)
                    data.drop_duplicates(subset="Date", keep="first", inplace=True)
                    data['Symbol'] = ticker
                    sp_df.append(data)

                clean_df = sp_df
                # Clean the data
                '''
                clean_df = sp_df.drop([i for i in range(6)], axis=0)
                rename_dict = {}
                for i in sp_df.columns:
                    rename_dict[i] = sp_df.loc[6, i]
                clean_df = clean_df.rename(rename_dict, axis=1)
                clean_df = clean_df.drop(6, axis=0)
                clean_df = clean_df.drop(clean_df.index[-1], axis=0)
                clean_df = clean_df.set_index("Year")
                '''
                # Calculate earnings and dividend growth rates
                clean_df["earnings_growth"] = clean_df["Earnings"] / clean_df["Earnings"].shift(1) - 1
                clean_df["dividend_growth"] = clean_df["Dividends"] / clean_df["Dividends"].shift(1) - 1

                # Calculate 10-year mean growth rates for earnings and dividends
                clean_df["earnings_10yr_mean_growth"] = clean_df["earnings_growth"].expanding(10).mean()
                clean_df["dividends_10yr_mean_growth"] = clean_df["dividend_growth"].expanding(10).mean()

                # Plot earnings growth rates and 10-year mean growth rates
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=clean_df.index, y=clean_df["earnings_growth"], mode='lines', name='Year over Year Earnings Growth'))
                fig.add_trace(go.Scatter(x=clean_df.index, y=clean_df["earnings_10yr_mean_growth"], mode='lines', name='Rolling 10 Year Mean'))
                fig.update_layout(title="Earnings Growth Rates", xaxis_title="Year", yaxis_title="Earnings Growth")
                st.plotly_chart(fig)

                # Base case valuation of S&P 500
                valuations = []

                # Slower recovery
                eps_growth_2020 = (11.88 + 17.76 + 25 + 25) / (34.95 + 35.08 + 33.99 + 35.72) - 1
                bad_eps_2020 = clean_df.iloc[-1]["Earnings"] * (1 + eps_growth_2020)
                eps_next = 24 * 4
                eps_growth = [0, 0.15, 0.22, 0.20, 0.16, 0.13, 0.11, 0.09, 0.08, 0.08]

                bad_df = pd.DataFrame()
                bad_df["earnings"] = (np.array(eps_growth) + 1).cumprod() * eps_next
                bad_df["dividends"] = 0.50 * bad_df["earnings"]
                bad_df["year"] = [i for i in range(2021, 2031)]
                bad_df.set_index("year", inplace=True)

                pv_dividends = 0
                for i in range(bad_df.shape[0]):
                    pv_dividends += bad_df["dividends"].iloc[i] / (1 + 0.08) ** i

                terminal_value = (bad_df["dividends"].iloc[-1] * (1 + 0.04) / (0.08 - 0.04))
                valuations.append(pv_dividends + terminal_value / (1 + 0.08) ** 10)

                # Double dip
                eps_growth_2020 = (11.88 + 17.76 + 25 + 25) / (34.95 + 35.08 + 33.99 + 35.72) - 1
                worst_eps_2020 = clean_df.iloc[-1]["Earnings"] * (1 + eps_growth_2020)
                eps_next = 24 * 4
                eps_growth = [0, -0.1, 0, 0.25, 0.25, 0.15, 0.12, 0.10, 0.08, 0.08]

                worst_df = pd.DataFrame()
                worst_df["earnings"] = (np.array(eps_growth) + 1).cumprod() * eps_next
                worst_df["dividends"] = 0.50 * worst_df["earnings"]
                worst_df["year"] = [i for i in range(2021, 2031)]
                worst_df.set_index("year", inplace=True)

                pv_dividends = 0
                for i in range(worst_df.shape[0]):
                    pv_dividends += worst_df["dividends"].iloc[i] / (1 + 0.08) ** i

                terminal_value = (worst_df["dividends"].iloc[-1] * (1 + 0.04) / (0.08 - 0.04))
                valuations.append(pv_dividends + terminal_value / (1 + 0.08) ** 10)

                # V-shaped EPS growth
                eps_growth_2020 = (11.88 + 17.76 + 28.27 + 31.78) / (34.95 + 35.08 + 33.99 + 35.72) - 1
                eps_2020 = clean_df.iloc[-1]["Earnings"] * (1 + eps_growth_2020)
                eps_next = 28.27 + 31.78 + 32.85 + 36.77
                eps_growth = [
                    0,
                    (clean_df.iloc[-1]["Earnings"]) / eps_next - 1,
                    0.18,
                    0.14,
                    0.10,
                    0.08,
                    0.08,
                    0.08,
                    0.08,
                    0.08,
                ]

                value_df = pd.DataFrame()
                value_df["earnings"] = (np.array(eps_growth) + 1).cumprod() * eps_next
                value_df["dividends"] = 0.50 * value_df["earnings"]
                value_df["year"] = [i for i in range(2021, 2031)]
                value_df.set_index("year", inplace=True)

                pv_dividends = 0
                for i in range(value_df.shape[0]):
                    pv_dividends += value_df["dividends"].iloc[i] / (1 + 0.08) ** i

                terminal_value = (value_df["dividends"].iloc[-1] * (1 + 0.04) / (0.08 - 0.04))
                valuations.append(pv_dividends + terminal_value / (1 + 0.08) ** 10)

                # Interactive visualization of earnings scenarios
                earnings_scenarios = pd.DataFrame()
                earnings_scenarios["Actual"] = pd.concat([clean_df.tail(15)["Earnings"], pd.Series([eps_2020], index=[2020]), value_df["earnings"]], axis=0)
                earnings_scenarios["Base Estimate"] = pd.concat([clean_df.tail(15)["Earnings"] * 0, pd.Series([eps_2020], index=[2020]) * 0, value_df["earnings"]], axis=0)
                earnings_scenarios["Bad Estimate"] = pd.concat([clean_df.tail(15)["Earnings"] * 0, pd.Series([bad_eps_2020], index=[2020]) * 0, bad_df["earnings"]], axis=0)
                earnings_scenarios["Worst Estimate"] = pd.concat([clean_df.tail(15)["Earnings"] * 0, pd.Series([worst_eps_2020], index=[2020]) * 0, worst_df["earnings"]], axis=0)

                fig = go.Figure()
                for column in earnings_scenarios.columns:
                    fig.add_trace(go.Bar(x=earnings_scenarios.index, y=earnings_scenarios[column], name=column))

                fig.update_layout(title="S&P 500 Earnings Per Share Scenarios", xaxis_title="Year", yaxis_title="Earnings Per Share")
                st.plotly_chart(fig)

                # Interactive visualization of valuations
                valuation_data = pd.DataFrame(valuations, columns=["Valuation"], index=["Slower Recovery", "Double Dip", "V-shaped"])

                fig = go.Figure()
                fig.add_trace(go.Bar(x=valuation_data.index, y=valuation_data["Valuation"], name="Valuation", marker_color="blue"))

                fig.update_layout(title="S&P 500 Valuations", xaxis_title="Scenario", yaxis_title="Valuation")
                st.plotly_chart(fig)

        if pred_option_analysis == "Stock Pivot Resistance":
            # Function to fetch and plot stock data with pivot points
            def plot_stock_pivot_resistance(stock_symbol, start_date, end_date):
                df = yf.download(stock_symbol, start_date, end_date)
                # Plot high prices
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["High"], mode='lines', name='High'))
                
                # Initialize variables to find and store pivot points
                pivots = []
                dates = []
                counter = 0
                last_pivot = 0
                window_size = 10
                window = [0] * window_size
                date_window = [0] * window_size

                # Identify pivot points
                for i, high_price in enumerate(df["High"]):
                    window = window[1:] + [high_price]
                    date_window = date_window[1:] + [df.index[i]]

                    current_max = max(window)
                    if current_max == last_pivot:
                        counter += 1
                    else:
                        counter = 0

                    if counter == 5:
                        last_pivot = current_max
                        last_date = date_window[window.index(last_pivot)]
                        pivots.append(last_pivot)
                        dates.append(last_date)
                # Plot resistance levels for each pivot point
                for i in range(len(pivots)):
                    time_delta = dt.timedelta(days=30)
                    fig.add_shape(type="line",
                                x0=dates[i], y0=pivots[i],
                                x1=dates[i] + time_delta, y1=pivots[i],
                                line=dict(color="red", width=4, dash="solid")
                                )
                # Configure plot settings
                fig.update_layout(title=stock_symbol.upper() + ' Resistance Points', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)
        
            st.write ("This segment allows us to analyze the pivot resistance points of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                plot_stock_pivot_resistance(ticker,start_date, end_date)


        if pred_option_analysis == "Stock Profit/Loss Analysis":

            def calculate_stock_profit_loss(symbol, start_date, end_date, initial_investment):
                # Download stock data
                dataset = yf.download(symbol, start_date, end_date)

                # Calculate the number of shares and investment values
                shares = initial_investment / dataset['Adj Close'][0]
                current_value = shares * dataset['Adj Close'][-1]

                # Calculate profit or loss and related metrics
                profit_or_loss = current_value - initial_investment
                percentage_gain_or_loss = (profit_or_loss / current_value) * 100
                percentage_returns = (current_value - initial_investment) / initial_investment * 100
                net_gains_or_losses = (dataset['Adj Close'][-1] - dataset['Adj Close'][0]) / dataset['Adj Close'][0] * 100
                total_return = ((current_value / initial_investment) - 1) * 100

                # Calculate profit and loss for each day
                dataset['PnL'] = shares * (dataset['Adj Close'].diff())

                # Visualize the profit and loss
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dataset.index, y=dataset['PnL'], mode='lines', name='Profit/Loss'))
                fig.update_layout(title=f'Profit and Loss for {symbol} Each Day', xaxis_title='Date', yaxis_title='Profit/Loss')
                st.plotly_chart(fig)

                # Display financial analysis
                st.write(f"Financial Analysis for {symbol}")
                st.write('-' * 50)
                st.write(f"Profit or Loss: ${profit_or_loss:.2f}")
                st.write(f"Percentage Gain or Loss: {percentage_gain_or_loss:.2f}%")
                st.write(f"Percentage of Returns: {percentage_returns:.2f}%")
                st.write(f"Net Gains or Losses: {net_gains_or_losses:.2f}%")
                st.write(f"Total Returns: {total_return:.2f}%")

            st.write ("This segment allows us to analyze the stock profit/loss returns")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio_one = st.number_input("Enter your initial investment")
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):

                calculate_stock_profit_loss(ticker, start_date, end_date, portfolio_one)
            
        if pred_option_analysis == "Stock Return Statistical Analysis":
            def analyze_stock_returns(symbol, start, end):
                # Download stock data
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

            st.write ("This segment allows you to view stock returns using some statistcal methods")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):

                analyze_stock_returns(ticker, start_date, end_date)

        if pred_option_analysis == "VAR Analysis":
            def calculate_var(stock, start, end):
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

            st.write ("This segment allows you to view the average stock returns monthly")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            if st.button("Check"):
                st.write(years)
                calculate_var(ticker, start_date, end_date)


        if pred_option_analysis == "Stock Returns":
            def view_stock_returns(symbol, num_years):
                # Fetch stock data for the given number of years
                start_date = dt.date.today() - dt.timedelta(days=365 * num_years)
                end_date = dt.date.today()
                dataset = yf.download(symbol, start_date, end_date)
                
                # Plot Adjusted Close Price over time
                fig_adj_close = go.Figure()
                fig_adj_close.add_trace(go.Scatter(x=dataset.index, y=dataset['Adj Close'], mode='lines', name='Adj Close'))
                fig_adj_close.update_layout(title=f'{symbol} Closing Price Chart', xaxis_title='Date', yaxis_title='Price', showlegend=True)

                # Monthly Returns Analysis
                monthly_dataset = dataset.asfreq('BM')
                monthly_dataset['Returns'] = dataset['Adj Close'].pct_change().dropna()
                monthly_dataset['Month_Name'] = monthly_dataset.index.strftime('%b-%Y')
                monthly_dataset['ReturnsPositive'] = monthly_dataset['Returns'] > 0
                colors = monthly_dataset['ReturnsPositive'].map({True: 'g', False: 'r'})
                
                fig_monthly_returns = px.bar(monthly_dataset, x=monthly_dataset.index, y='Returns', color=colors,
                                            labels={'x': 'Month', 'y': 'Returns'}, title='Monthly Returns')
                fig_monthly_returns.update_xaxes(tickvals=monthly_dataset.index, ticktext=monthly_dataset['Month_Name'], tickangle=45)

                # Yearly Returns Analysis
                yearly_dataset = dataset.asfreq('BY')
                yearly_dataset['Returns'] = dataset['Adj Close'].pct_change().dropna()
                yearly_dataset['ReturnsPositive'] = yearly_dataset['Returns'] > 0
                colors_year = yearly_dataset['ReturnsPositive'].map({True: 'g', False: 'r'})
                
                fig_yearly_returns = px.bar(yearly_dataset, x=yearly_dataset.index.year, y='Returns', color=colors_year,
                                            labels={'x': 'Year', 'y': 'Returns'}, title='Yearly Returns')

                # Show the interactive plots
                st.plotly_chart(fig_adj_close)
                st.plotly_chart(fig_monthly_returns)
                st.plotly_chart(fig_yearly_returns)

            st.write ("This segment allows you to view the average stock returns yearly")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            if st.button("Check"):
                st.write(years)
                view_stock_returns(ticker, years)
    


    elif option =='Technical Indicators':
        pred_option_Technical_Indicators = st.selectbox('Make a choice', [
                                                                'Exponential Moving Average (EMA)',
                                                                'EMA Volume',
                                                                'Positive Volume Trend (PVT)',
                                                                'Exponential Weighted Moving Average (EWMA)',
                                                                'Weighted Smoothing Moving Average (WSMA)',
                                                                'Z Score Indicator (Z Score)',
                                                                'Absolute Price Oscillator (APO)',
                                                                'Acceleration Bands',
                                                                'Accumulation Distribution Line',
                                                                'Aroon',
                                                                'Aroon Oscillator',
                                                                'Average Directional Index (ADX)',
                                                                'Average True Range (ATR)',
                                                                'Balance of Power',
                                                                'Beta Indicator',
                                                                'Bollinger Bands',
                                                                'Bollinger Bandwidth',
                                                                'Breadth Indicator',
                                                                'Candle Absolute Returns',
                                                                'GANN lines angles',
                                                                'GMMA',
                                                                'Moving Average Convergence Divergence (MACD)',
                                                                'MA high low',
                                                                'Price Volume Trend Indicator (PVI)',
                                                                'Price Volume Trend (PVT)',
                                                                'Rate of Change (ROC)',
                                                                'Return on Investment (ROI)',
                                                                'Relative Strength Index (RSI)',
                                                                'RSI BollingerBands',
                                                                'Simple Moving Average (SMA)',
                                                                'Weighted Moving Average (WMA)',
                                                                'Triangular Moving Average (TRIMA)',
                                                                'Time-Weighted Average Price (TWAP)',
                                                                'Volume Weighted Average Price (VWAP)',
                                                                'Central Pivot Range (CPR)',
                                                                'Chaikin Money Flow',
                                                                'Chaikin Oscillator',
                                                                'Commodity Channel Index (CCI)',
                                                                'Correlation Coefficient',
                                                                'Covariance',
                                                                'Detrended Price Oscillator (DPO)',
                                                                'Donchain Channel',
                                                                'Double Exponential Moving Average (DEMA)',
                                                                'Dynamic Momentum Index',
                                                                'Ease of Movement',
                                                                'Force Index',
                                                                'Geometric Return Indicator',
                                                                'Golden/Death Cross',
                                                                'High Minus Low',
                                                                'Hull Moving Average',
                                                                'Keltner Channels',
                                                                'Linear Regression',
                                                                'Linear Regression Slope',
                                                                'Linear Weighted Moving Average (LWMA)',
                                                                'McClellan Oscillator',
                                                                'Momentum',
                                                                'Moving Average Envelopes',
                                                                'Moving Average High/Low',
                                                                'Moving Average Ribbon',
                                                                'Moving Average Envelopes (MMA)',
                                                                'Moving Linear Regression',
                                                                'New Highs/New Lows',
                                                                'Pivot Point',
                                                                'Money Flow Index (MFI)',
                                                                'Price Channels',
                                                                'Price Relative',
                                                                'Realized Volatility',
                                                                'Relative Volatility Index',
                                                                'Smoothed Moving Average',
                                                                'Speed Resistance Lines',
                                                                'Standard Deviation Volatility',
                                                                'Stochastic RSI',
                                                                'Stochastic Fast',
                                                                'Stochastic Full',
                                                                'Stochastic Slow',
                                                                'Super Trend',
                                                                'True Strength Index',
                                                                'Ultimate Oscillator',
                                                                'Variance Indicator',
                                                                'Volume Price Confirmation Indicator',
                                                                'Volume Weighted Moving Average (VWMA)'])

        if pred_option_Technical_Indicators == "Exponential Moving Average (EMA)":
            st.success("This program allows you to view the Exponential Moving Average of a stock ")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):
                    symbol = ticker
                    start = start_date
                    end = end_date

                    # Read data
                    df = yf.download(symbol, start, end)

                    n = 15
                    df["EMA"] = (
                        df["Adj Close"].ewm(ignore_na=False, span=n, min_periods=n, adjust=True).mean()
                    )

                    dfc = df.copy()
                    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
                    # dfc = dfc.dropna()
                    dfc = dfc.reset_index()
                    dfc["Date"] = mdates.date2num(dfc["Date"].tolist())
                    dfc["Date"] = pd.to_datetime(dfc["Date"])  # Convert Date column to datetime
                    dfc["Date"] = dfc["Date"].apply(mdates.date2num)

                    # Plotting Moving Average using Plotly
                    trace_adj_close = go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close')
                    trace_ema = go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA')
                    layout_ma = go.Layout(title="Stock Closing Price of " + str(n) + "-Day Exponential Moving Average",
                                        xaxis=dict(title="Date"), yaxis=dict(title="Price"))
                    fig_ma = go.Figure(data=[trace_adj_close, trace_ema], layout=layout_ma)

                    # Plotting Candlestick with EMA using Plotly
                    dfc = df.copy()
                    dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]

                    trace_candlestick = go.Candlestick(x=dfc.index,
                                                    open=dfc['Open'],
                                                    high=dfc['High'],
                                                    low=dfc['Low'],
                                                    close=dfc['Close'],
                                                    name='Candlestick')

                    trace_ema = go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA')


                    trace_volume = go.Bar(x=dfc.index, y=dfc['Volume'], marker=dict(color=dfc['VolumePositive'].map({True: 'green', False: 'red'})),
                                        name='Volume')

                    layout_candlestick = go.Layout(title="Stock " + symbol + " Closing Price",
                                                xaxis=dict(title="Date", type='date', tickformat='%d-%m-%Y'),
                                                yaxis=dict(title="Price"),
                                                yaxis2=dict(title="Volume", overlaying='y', side='right'))
                    fig_candlestick = go.Figure(data=[trace_candlestick, trace_ema, trace_volume], layout=layout_candlestick)


                    # Display Plotly figures in Streamlit
                    st.plotly_chart(fig_ma)
                    st.warning("Click candlestick, EMA or Volume to tinker with the graph")
                    st.plotly_chart(fig_candlestick)                

        if pred_option_Technical_Indicators == "EMA Volume":
            st.success("This program allows you to view EMA But focusing on Volume of a stock ")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):
                    symbol = ticker
                    start = start_date
                    end = end_date

                    # Read data
                    df = yf.download(symbol, start, end)

                    n = 15
                    df["EMA"] = (
                        df["Adj Close"].ewm(ignore_na=False, span=n, min_periods=n, adjust=True).mean()
                    )

                    df["EMA"] = df["Adj Close"].ewm(span=n, adjust=False).mean()  # Recalculate EMA
                    df["VolumePositive"] = df["Open"] < df["Adj Close"]

                    # Create traces
                    trace_candlestick = go.Candlestick(x=df.index,
                                                    open=df['Open'],
                                                    high=df['High'],
                                                    low=df['Low'],
                                                    close=df['Close'],
                                                    name='Candlestick')

                    trace_volume = go.Bar(x=df.index, y=df['Volume'],
                                        marker=dict(color=df['VolumePositive'].map({True: 'green', False: 'red'})),
                                        name='Volume')

                    trace_ema = go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA', line=dict(color='green'))

                    # Create layout
                    layout = go.Layout(title="Stock " + symbol + " Closing Price",
                                    xaxis=dict(title="Date", type='date', tickformat='%d-%m-%Y'),
                                    yaxis=dict(title="Price"),
                                    yaxis2=dict(title="Volume", overlaying='y', side='right'))

                    # Create figure
                    fig = go.Figure(data=[trace_candlestick, trace_ema, trace_volume], layout=layout)

                    # Display Plotly figure in Streamlit
                    st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Exponential Weighted Moving Average (EWMA)":

            st.success("This program allows you to view the WMA But focusing on Volume of a stock ")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):
                    symbol = ticker
                    start = start_date
                    end = end_date
                    df = yf.download(symbol, start, end)

                    n = 7
                    df["EWMA"] = df["Adj Close"].ewm(ignore_na=False, min_periods=n - 1, span=n).mean()
                    # Plotting Candlestick with EWMA
                    fig_candlestick = go.Figure()

                    # Plot candlestick
                    fig_candlestick.add_trace(go.Candlestick(x=df.index,
                                                            open=df['Open'],
                                                            high=df['High'],
                                                            low=df['Low'],
                                                            close=df['Close'],
                                                            name='Candlestick'))
                    # Plot volume
                    volume_color = df['Open'] < df['Close']  # Color based on close > open
                    volume_color = volume_color.map({True: 'green', False: 'red'}) 
                    fig_candlestick.add_trace(go.Bar(x=df.index,
                                                    y=df['Volume'],
                                                    marker_color=volume_color,  
                                                    name='Volume'))

                    # Plot EWMA
                    fig_candlestick.add_trace(go.Scatter(x=df.index,
                                                        y=df['EWMA'],
                                                        mode='lines',
                                                        name='EWMA',
                                                        line=dict(color='red')))

                    # Update layout
                    fig_candlestick.update_layout(title="Stock " + symbol + " Closing Price",
                                                xaxis=dict(title="Date"),
                                                yaxis=dict(title="Price"),
                                                yaxis2=dict(title="Volume", overlaying='y', side='right'),
                                                legend=dict(yanchor="top", y=1, xanchor="left", x=0))

                    # Display Plotly figure in Streamlit
                    st.plotly_chart(fig_candlestick)

        if pred_option_Technical_Indicators == "GANN lines angles":

            st.success("This program allows you to view the GANN lines of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Line Chart
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))

                # Add diagonal line
                x_lim = [df.index[0], df.index[-1]]
                y_lim = [df["Adj Close"].iloc[0], df["Adj Close"].iloc[-1]]
                fig_line.add_trace(go.Scatter(x=x_lim, y=y_lim, mode='lines', line=dict(color='red'), name='45 degree'))

                fig_line.update_layout(title="Stock of GANN Angles", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_line)

                # GANN Angles
                angles = [82.5, 75, 71.25, 63.75, 45, 26.25, 18.75, 15, 7.5]

                fig_gann = go.Figure()
                fig_gann.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))

                for angle in angles:
                    angle_radians = np.radians(angle)
                    y_values = np.tan(angle_radians) * np.arange(len(df))
                    fig_gann.add_trace(go.Scatter(x=df.index, y=y_values, mode='markers', name=str(angle)))

                fig_gann.update_layout(title="Stock of GANN Angles", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_gann)

                # Candlestick with GANN Lines Angles
                fig_candlestick = go.Figure()

                # Candlestick
                fig_candlestick.add_trace(go.Candlestick(x=df.index,
                                                        open=df['Open'],
                                                        high=df['High'],
                                                        low=df['Low'],
                                                        close=df['Close'],
                                                        name='Candlestick'))

                # GANN Angles
                for angle in angles:
                    angle_radians = np.radians(angle)
                    y_values = np.tan(angle_radians) * np.arange(len(df))
                    fig_candlestick.add_trace(go.Scatter(x=df.index, y=y_values, mode='markers', name=str(angle)))

                # Volume
                fig_candlestick.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=np.where(df['Open'] < df['Close'], 'green', 'red')))

                fig_candlestick.update_layout(title="Stock Closing Price", xaxis_title="Date", yaxis_title="Price", 
                                            yaxis2=dict(title="Volume", overlaying='y', side='right', tickformat=',.0f'))
                st.plotly_chart(fig_candlestick)

        if pred_option_Technical_Indicators == "GMMA":

            st.success("This program allows you to view the GANN lines of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Short-term for EMA
                df["EMA3"] = df["Adj Close"].ewm(span=3, adjust=False).mean()
                df["EMA5"] = df["Adj Close"].ewm(span=5, adjust=False).mean()
                df["EMA8"] = df["Adj Close"].ewm(span=8, adjust=False).mean()
                df["EMA10"] = df["Adj Close"].ewm(span=10, adjust=False).mean()
                df["EMA12"] = df["Adj Close"].ewm(span=12, adjust=False).mean()
                df["EMA15"] = df["Adj Close"].ewm(span=15, adjust=False).mean()

                # Long-term for EMA
                df["EMA30"] = df["Adj Close"].ewm(span=30, adjust=False).mean()
                df["EMA35"] = df["Adj Close"].ewm(span=35, adjust=False).mean()
                df["EMA40"] = df["Adj Close"].ewm(span=40, adjust=False).mean()
                df["EMA45"] = df["Adj Close"].ewm(span=45, adjust=False).mean()
                df["EMA50"] = df["Adj Close"].ewm(span=50, adjust=False).mean()
                df["EMA60"] = df["Adj Close"].ewm(span=60, adjust=False).mean()

                EMA_Short = df[["EMA3", "EMA5", "EMA8", "EMA10", "EMA12", "EMA15"]]
                EMA_Long = df[["EMA30", "EMA35", "EMA40", "EMA45", "EMA50", "EMA60"]]

                # Short-term for SMA
                df["SMA3"] = df["Adj Close"].rolling(window=3).mean()
                df["SMA5"] = df["Adj Close"].rolling(window=5).mean()
                df["SMA8"] = df["Adj Close"].rolling(window=8).mean()
                df["SMA10"] = df["Adj Close"].rolling(window=10).mean()
                df["SMA12"] = df["Adj Close"].rolling(window=12).mean()
                df["SMA15"] = df["Adj Close"].rolling(window=15).mean()

                # Long-term for SMA
                df["SMA30"] = df["Adj Close"].rolling(window=30).mean()
                df["SMA35"] = df["Adj Close"].rolling(window=35).mean()
                df["SMA40"] = df["Adj Close"].rolling(window=40).mean()
                df["SMA45"] = df["Adj Close"].rolling(window=45).mean()
                df["SMA50"] = df["Adj Close"].rolling(window=50).mean()
                df["SMA60"] = df["Adj Close"].rolling(window=60).mean()

                SMA_Short = df[["SMA3", "SMA5", "SMA8", "SMA10", "SMA12", "SMA15"]]
                SMA_Long = df[["SMA30", "SMA35", "SMA40", "SMA45", "SMA50", "SMA60"]]

                # Plot EMA
                fig_ema = go.Figure()
                fig_ema.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                for col in EMA_Short.columns:
                    fig_ema.add_trace(go.Scatter(x=df.index, y=EMA_Short[col], mode='lines', name=col, line=dict(color='blue')))
                for col in EMA_Long.columns:
                    fig_ema.add_trace(go.Scatter(x=df.index, y=EMA_Long[col], mode='lines', name=col, line=dict(color='orange')))
                fig_ema.update_layout(title="Guppy Multiple Moving Averages of EMA", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_ema)

                # Plot SMA
                fig_sma = go.Figure()
                fig_sma.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                for col in SMA_Short.columns:
                    fig_sma.add_trace(go.Scatter(x=df.index, y=SMA_Short[col], mode='lines', name=col, line=dict(color='blue')))
                for col in SMA_Long.columns:
                    fig_sma.add_trace(go.Scatter(x=df.index, y=SMA_Long[col], mode='lines', name=col, line=dict(color='orange')))
                fig_sma.update_layout(title="Guppy Multiple Moving Averages of SMA", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_sma)

                st.warning("Untick the volume to view the candlesticks and the movement lines")

                # Candlestick with GMMA
                fig_candlestick = go.Figure()

                # Candlestick
                fig_candlestick.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))

                # Plot EMA on Candlestick
                for col in EMA_Short.columns:
                    fig_candlestick.add_trace(go.Scatter(x=df.index, y=EMA_Short[col], mode='lines', name=col, line=dict(color='orange')))
                for col in EMA_Long.columns:
                    fig_candlestick.add_trace(go.Scatter(x=df.index, y=EMA_Long[col], mode='lines', name=col, line=dict(color='blue')))

                # Volume
                fig_candlestick.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=np.where(df['Open'] < df['Close'], 'green', 'red')))

                fig_candlestick.update_layout(title="Stock Closing Price", xaxis_title="Date", yaxis_title="Price",
                                            yaxis2=dict(title="Volume", overlaying='y', side='right', tickformat=',.0f'))  # Set tick format to not show in millions
                st.plotly_chart(fig_candlestick)


        if pred_option_Technical_Indicators == "Moving Average Convergence Divergence (MACD)":
            st.success("This program allows you to view the GANN lines of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)
                # Read data
                df = yf.download(symbol, start, end)

                df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(
                    df["Adj Close"], fastperiod=12, slowperiod=26, signalperiod=9
                )
                df = df.dropna()
                df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")

                # Line Chart
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig_line.add_hline(y=df["Adj Close"].mean(), line_dash="dash", line_color="red", name='Mean')

                # Volume
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=df['Volume'].map(lambda x: 'green' if x > 0 else 'red')))
                # MACD
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"], mode='lines', name='MACD'))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df["macdsignal"], mode='lines', name='Signal'))
                fig_macd.add_trace(go.Bar(x=df.index, y=df["macdhist"], name='Histogram', marker_color=df['macdhist'].map(lambda x: 'green' if x > 0 else 'red')))

                st.plotly_chart(fig_line)
                st.plotly_chart(fig_volume)
                st.plotly_chart(fig_macd)
            
        if pred_option_Technical_Indicators == "Money Flow Index (MFI)":
            st.success("This program allows you to view the GANN lines of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Money Flow Index (MFI)
                def calculate_mfi(high, low, close, volume, period=14):
                    typical_price = (high + low + close) / 3
                    raw_money_flow = typical_price * volume
                    positive_flow = (raw_money_flow * (close > close.shift(1))).rolling(window=period).sum()
                    negative_flow = (raw_money_flow * (close < close.shift(1))).rolling(window=period).sum()
                    money_ratio = positive_flow / negative_flow
                    mfi = 100 - (100 / (1 + money_ratio))
                    return mfi

                df["MFI"] = calculate_mfi(df["High"], df["Low"], df["Close"], df["Volume"])    
        
                # Plot interactive chart for closing price
                fig_close = go.Figure()
                fig_close.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
                fig_close.update_layout(
                    title="Interactive Chart for Closing Price",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Price"),
                    legend=dict(x=0, y=1),
                )

                # Find min and max price
                min_price = df["Close"].min()
                max_price = df["Close"].max()

                # Plot interactive chart for MFI
                fig_mfi = go.Figure()
                fig_mfi.add_trace(go.Scatter(x=df.index, y=df["MFI"], mode="lines", name="MFI"))
                fig_mfi.update_layout(
                    title="Interactive Chart for Money Flow Index (MFI)",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="MFI"),
                    legend=dict(x=0, y=1),
                    shapes=[
                        dict(
                            type="line",
                            x0=df.index[0],
                            y0=min_price,
                            x1=df.index[-1],
                            y1=min_price,
                            line=dict(color="blue", width=1, dash="dash"),
                        ),
                        dict(
                            type="line",
                            x0=df.index[0],
                            y0=max_price,
                            x1=df.index[-1],
                            y1=max_price,
                            line=dict(color="red", width=1, dash="dash"),
                        ),
                    ],
                )
                
                # Plot interactive chart
                fig_main = go.Figure()

                # Add closing price trace
                fig_main.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

                # Add MFI trace
                fig_main.add_trace(go.Scatter(x=df.index, y=df["MFI"], mode="lines", name="MFI"))

                # Update layout
                fig_main.update_layout(
                    title="Interactive Chart with Money Flow Index (MFI)",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Price"),
                    yaxis2=dict(title="MFI", overlaying="y", side="right"),
                    legend=dict(x=0, y=1),
                )

                # Show interactive chart
                st.plotly_chart(fig_close)
                st.plotly_chart(fig_mfi)
                st.plotly_chart(fig_main) 


        if pred_option_Technical_Indicators == "MA high low":
            st.success("This program allows you to view the MA High/low of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                
                # Read data
                df = yf.download(symbol, start, end)

                df["MA_High"] = df["High"].rolling(10).mean()
                df["MA_Low"] = df["Low"].rolling(10).mean()
                df = df.dropna()

                # Moving Average Line Chart
                fig1 = px.line(df, x=df.index, y=["Adj Close", "MA_High", "MA_Low"], title="Moving Average of High and Low for Stock")
                fig1.update_xaxes(title_text="Date")
                fig1.update_yaxes(title_text="Price")
                st.plotly_chart(fig1)

                # Candlestick with Moving Averages High and Low
                fig2 = go.Figure(data=[go.Candlestick(x=df.index,
                                                    open=df['Open'],
                                                    high=df['High'],
                                                    low=df['Low'],
                                                    close=df['Adj Close'])])

                fig2.add_trace(go.Scatter(x=df.index, y=df['MA_High'], mode='lines', name='MA High'))
                fig2.add_trace(go.Scatter(x=df.index, y=df['MA_Low'], mode='lines', name='MA Low'))

                fig2.update_layout(title="Candlestick with Moving Averages High and Low",
                                xaxis_title="Date",
                                yaxis_title="Price")

                st.plotly_chart(fig2)
        if pred_option_Technical_Indicators == "Price Volume Trend Indicator (PVI)":
            st.success("This program allows you to visualize the PVI of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date  
                # Read data
                df = yf.download(symbol, start, end)

                returns = df["Adj Close"].pct_change()
                vol_increase = df["Volume"].shift(1) < df["Volume"]
                pvi = pd.Series(data=np.nan, index=df["Adj Close"].index, dtype="float64")

                pvi.iloc[0] = 1000
                for i in range(1, len(pvi)):
                    if vol_increase.iloc[i]:
                        pvi.iloc[i] = pvi.iloc[i - 1] * (1.0 + returns.iloc[i])
                    else:
                        pvi.iloc[i] = pvi.iloc[i - 1]

                pvi = pvi.replace([np.inf, -np.inf], np.nan).fillna(1000)

                df["PVI"] = pd.Series(pvi)

                # Line Chart with PVI
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig1.add_trace(go.Scatter(x=df.index, y=df["PVI"], mode='lines', name='Positive Volume Index'))

                fig1.update_layout(title="Adj Close and Positive Volume Index (PVI) Over Time",
                                xaxis_title="Date",
                                yaxis_title="Price/PVI")

                st.plotly_chart(fig1)

                # Candlestick with PVI
                dfc = df.copy()
                dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
                dfc = dfc.reset_index()
                dfc["Date"] = mdates.date2num(dfc["Date"].tolist())

                fig2 = go.Figure()

                # Candlestick chart
                fig2.add_trace(go.Candlestick(x=dfc['Date'],
                                open=dfc['Open'],
                                high=dfc['High'],
                                low=dfc['Low'],
                                close=dfc['Adj Close'], name='Candlestick'))

                # Volume bars
                fig2.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], marker_color=dfc.VolumePositive.map({True: "green", False: "red"}), name='Volume'))

                fig2.add_trace(go.Scatter(x=df.index, y=df["PVI"], mode='lines', name='Positive Volume Index', line=dict(color='green')))

                fig2.update_layout(title="Candlestick Chart with Positive Volume Index (PVI)",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                xaxis_rangeslider_visible=False)

                st.plotly_chart(fig2)

        if pred_option_Technical_Indicators == "Positive Volume Trend (PVT)":
            st.success("This program allows you to view the PVT trend of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                df = yf.download(symbol, start, end)

                # Calculate Price Volume Trend (PVT)
                df["Momentum_1D"] = (df["Adj Close"] - df["Adj Close"].shift(1)).fillna(0)
                df["PVT"] = (df["Momentum_1D"] / df["Adj Close"].shift(1)) * df["Volume"]
                df["PVT"] = df["PVT"] - df["PVT"].shift(1)
                df["PVT"] = df["PVT"].fillna(0)

                # Line Chart with PVT
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig1.add_trace(go.Scatter(x=df.index, y=df["PVT"], mode='lines', name='Price Volume Trend'))

                fig1.update_layout(title="Adj Close and Price Volume Trend (PVT) Over Time",
                                xaxis_title="Date",
                                yaxis_title="Price/PVT")

                st.plotly_chart(fig1)

                # Candlestick with PVT
                dfc = df.copy()
                dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
                dfc = dfc.reset_index()
                dfc["Date"] = mdates.date2num(dfc["Date"].tolist())

                fig2 = go.Figure()

                # Candlestick chart
                fig2.add_trace(go.Candlestick(x=dfc['Date'],
                                open=dfc['Open'],
                                high=dfc['High'],
                                low=dfc['Low'],
                                close=dfc['Adj Close'], name='Candlestick'))

                # Volume bars
                fig2.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], marker_color=dfc.VolumePositive.map({True: "green", False: "red"}), name='Volume'))

                fig2.add_trace(go.Scatter(x=df.index, y=df["PVT"], mode='lines', name='Price Volume Trend', line=dict(color='blue')))

                fig2.update_layout(title="Candlestick Chart with Price Volume Trend (PVT)",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                xaxis_rangeslider_visible=False)

                st.plotly_chart(fig2)

        if pred_option_Technical_Indicators == "Rate of Change (ROC)":
            st.success("This program allows you to view the ROC of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Rate of Change (ROC)
                n = 12
                df["ROC"] = ((df["Adj Close"] - df["Adj Close"].shift(n)) / df["Adj Close"].shift(n)) * 100

                # Line Chart with ROC
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig1.add_trace(go.Scatter(x=df.index, y=df["ROC"], mode='lines', name='Rate of Change'))

                fig1.update_layout(title="Adj Close and Rate of Change (ROC) Over Time",
                                xaxis_title="Date",
                                yaxis_title="Price/ROC")

                st.plotly_chart(fig1)

                # Candlestick with ROC
                dfc = df.copy()
                dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
                dfc = dfc.reset_index()
                dfc["Date"] = mdates.date2num(dfc["Date"].tolist())

                fig2 = go.Figure()

                # Candlestick chart
                fig2.add_trace(go.Candlestick(x=dfc['Date'],
                                open=dfc['Open'],
                                high=dfc['High'],
                                low=dfc['Low'],
                                close=dfc['Adj Close'], name='Candlestick'))

                # Volume bars
                fig2.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], marker_color=dfc.VolumePositive.map({True: "green", False: "red"}), name='Volume'))

                fig2.add_trace(go.Scatter(x=df.index, y=df["ROC"], mode='lines', name='Rate of Change', line=dict(color='blue')))

                fig2.update_layout(title="Candlestick Chart with Rate of Change (ROC)",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                xaxis_rangeslider_visible=False)

                st.plotly_chart(fig2)

        if pred_option_Technical_Indicators == "Return on Investment (ROI)":
            st.success("This program allows you to view the ROI over time of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Return on Investment (ROI)
                df["ROI"] = ((df["Adj Close"] - df["Adj Close"].shift(1)) / df["Adj Close"].shift(1) * 100)

                # Line Chart with ROI
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig1.add_trace(go.Scatter(x=df.index, y=df["ROI"], mode='lines', name='Return on Investment'))

                fig1.update_layout(title="Adj Close and Return on Investment (ROI) Over Time",
                                xaxis_title="Date",
                                yaxis_title="Price/ROI")

                st.plotly_chart(fig1)

                # Candlestick with ROI
                dfc = df.copy()
                dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
                dfc = dfc.reset_index()
                dfc["Date"] = pd.to_datetime(dfc["Date"])
                dfc["Date"] = dfc["Date"].apply(mdates.date2num)

                fig2 = go.Figure()

                # Candlestick chart
                fig2.add_trace(go.Candlestick(x=dfc['Date'],
                                open=dfc['Open'],
                                high=dfc['High'],
                                low=dfc['Low'],
                                close=dfc['Adj Close'], name='Candlestick'))

                # Volume bars
                fig2.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], marker_color=dfc.VolumePositive.map({True: "green", False: "red"}), name='Volume'))

                fig2.add_trace(go.Scatter(x=df.index, y=df["ROI"], mode='lines', name='Return on Investment', line=dict(color='red')))

                fig2.update_layout(title="Candlestick Chart with Return on Investment (ROI)",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                xaxis_rangeslider_visible=False)

                st.plotly_chart(fig2)

                
        if pred_option_Technical_Indicators == "Relative Strength Index (RSI)":
            st.success("This program allows you to view the RSI over time of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                n = 14  # Number of period
                change = df["Adj Close"].diff(1)
                df["Gain"] = change.mask(change < 0, 0)
                df["Loss"] = abs(change.mask(change > 0, 0))
                df["AVG_Gain"] = df.Gain.rolling(n).mean()
                df["AVG_Loss"] = df.Loss.rolling(n).mean()
                df["RS"] = df["AVG_Gain"] / df["AVG_Loss"]
                df["RSI"] = 100 - (100 / (1 + df["RS"]))

                # Create RSI plot
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))

                fig1.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='Relative Strength Index'))

                fig1.update_layout(title=symbol + " Closing Price and Relative Strength Index (RSI)",
                                xaxis_title="Date",
                                yaxis_title="Price/RSI")

                st.plotly_chart(fig1)

                # Candlestick with RSI
                dfc = df.copy()
                dfc["VolumePositive"] = dfc["Open"] < dfc["Adj Close"]
                dfc = dfc.reset_index()
                dfc["Date"] = pd.to_datetime(dfc["Date"])
                dfc["Date"] = dfc["Date"].apply(mdates.date2num)

                fig2 = go.Figure()

                # Candlestick chart
                fig2.add_trace(go.Candlestick(x=dfc['Date'],
                                open=dfc['Open'],
                                high=dfc['High'],
                                low=dfc['Low'],
                                close=dfc['Adj Close'], name='Candlestick'))

                # Volume bars
                fig2.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], marker_color=dfc.VolumePositive.map({True: "green", False: "red"}), name='Volume'))

                fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='Relative Strength Index', line=dict(color='blue')))

                fig2.update_layout(title=symbol + " Candlestick Chart with Relative Strength Index (RSI)",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                xaxis_rangeslider_visible=False)

                st.plotly_chart(fig2)

        if pred_option_Technical_Indicators == "RSI BollingerBands":
            st.success("This program allows you to view the RSI with emphasis on BollingerBands over time of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Simple Line Chart
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig1.update_layout(title=symbol + " Closing Price", xaxis_title="Date", yaxis_title="Price")

                st.plotly_chart(fig1)

                # RSI
                rsi = ta.RSI(df["Adj Close"], timeperiod=14)
                rsi = rsi.dropna()

                # Bollinger Bands
                df["20 Day MA"] = df["Adj Close"].rolling(window=20).mean()
                df["20 Day STD"] = df["Adj Close"].rolling(window=20).std()
                df["Upper Band"] = df["20 Day MA"] + (df["20 Day STD"] * 2)
                df["Lower Band"] = df["20 Day MA"] - (df["20 Day STD"] * 2)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig2.add_trace(go.Scatter(x=df.index, y=df["20 Day MA"], mode='lines', name='20 Day MA'))
                fig2.add_trace(go.Scatter(x=df.index, y=df["Upper Band"], mode='lines', name='Upper Band'))
                fig2.add_trace(go.Scatter(x=df.index, y=df["Lower Band"], mode='lines', name='Lower Band'))
                fig2.update_layout(title=f"30 Day Bollinger Band for {symbol.upper()}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig2)

                dfc = df.reset_index()
                dfc["Date"] = pd.to_datetime(dfc["Date"])

                # Candlestick with Bollinger Bands
                fig3 = go.Figure()
                fig3.add_trace(go.Candlestick(x=dfc['Date'],
                                open=dfc['Open'],
                                high=dfc['High'],
                                low=dfc['Low'],
                                close=dfc['Adj Close'], name='Candlestick'))

                fig3.add_trace(go.Scatter(x=dfc['Date'], y=df["20 Day MA"], mode='lines', name='20 Day MA'))
                fig3.add_trace(go.Scatter(x=dfc['Date'], y=df["Upper Band"], mode='lines', name='Upper Band'))
                fig3.add_trace(go.Scatter(x=dfc['Date'], y=df["Lower Band"], mode='lines', name='Lower Band'))

                fig3.update_layout(title=f"{symbol.upper()} Candlestick Chart with Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig3)

                # Combine RSI and Bollinger Bands
                fig4 = go.Figure()

                fig4.add_trace(go.Scatter(x=df.index, y=df["20 Day MA"], mode='lines', name='20 Day MA'))
                fig4.add_trace(go.Scatter(x=df.index, y=df["Upper Band"], mode='lines', name='Upper Band'))
                fig4.add_trace(go.Scatter(x=df.index, y=df["Lower Band"], mode='lines', name='Lower Band'))

                fig4.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI'))

                fig4.update_layout(title=f"Bollinger Bands & RSI for {symbol.upper()}", xaxis_title="Date", yaxis_title="Price/RSI")
                st.plotly_chart(fig4)
            
        if pred_option_Technical_Indicators == "Volume Weighted Average Price (VWAP)":
            st.success("This program allows you to view the VWAP over time of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
            
                # Read data
                df = yf.download(symbol, start, end)

                def VWAP(df):
                    return (df["Adj Close"] * df["Volume"]).sum() / df["Volume"].sum()

                n = 14
                df["VWAP"] = pd.concat(
                    [
                        (pd.Series(VWAP(df.iloc[i : i + n]), index=[df.index[i + n]]))
                        for i in range(len(df) - n)
                    ]
                )
            
                vwap_series = pd.concat([(pd.Series(VWAP(df.iloc[i : i + n]), index=[df.index[i + n]])) for i in range(len(df) - n)])
                vwap_series = vwap_series.dropna()


                # Simple Line Chart
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig1.add_trace(go.Scatter(x=vwap_series.index, y=vwap_series, mode='lines', name='VWAP'))
                fig1.update_layout(title="Volume Weighted Average Price for Stock", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig1)

                # Candlestick with VWAP
                df.loc[:, "VolumePositive"] = df["Open"] < df["Adj Close"]
            
                df = df.dropna()
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"])
                df["Date"] = df["Date"].apply(mdates.date2num)

                fig2 = go.Figure()

                fig2.add_trace(go.Candlestick(x=df['Date'],
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'], name='Candlestick'))

                fig2.add_trace(go.Scatter(x=df['Date'], y=df["VWAP"], mode='lines', name='VWAP'))

                fig2.update_layout(title=f"Stock {symbol} Closing Price with VWAP", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig2)

        if pred_option_Technical_Indicators == "Weighted Moving Average (WMA)":
            st.success("This program allows you to view the WMA over time of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
            
                # Read data
                df = yf.download(symbol, start, end)

                def WMA(data, n):
                    ws = np.zeros(data.shape[0])
                    t_sum = sum(range(1, n + 1))
                    for i in range(n - 1, data.shape[0]):
                        ws[i] = sum(data[i - n + 1 : i + 1] * np.linspace(1, n, n)) / t_sum
                    return ws

                df["WMA"] = WMA(df["Adj Close"], 5)

                # Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.add_trace(go.Scatter(x=df.index[4:], y=df["WMA"][4:], mode='lines', name='WMA'))
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='#0079a3', opacity=0.4))

                fig.update_layout(title=f'Stock {symbol} Closing Price', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

                # Candlestick with WMA
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.add_trace(go.Scatter(x=df.index[4:], y=df["WMA"][4:], mode='lines', name='WMA'))

                fig.update_layout(title=f'Stock {symbol} Candlestick Chart with WMA', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Weighted Smoothing Moving Average (WSMA)":
            st.success("This program allows you to view the WMA over time of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
            
                # Read data
                df = yf.download(symbol, start, end)
                
                def WSMA(df, column="Adj Close", n=14):
                    ema = df[column].ewm(span=n, min_periods=n - 1).mean()
                    K = 1 / n
                    wsma = df[column] * K + ema * (1 - K)
                    return wsma

                df["WSMA"] = WSMA(df, column="Adj Close", n=14)
                df = df.dropna()

                # Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.add_trace(go.Scatter(x=df.index, y=df["WSMA"], mode='lines', name="WSMA"))
                fig.update_layout(title="Wilder's Smoothing Moving Average for Stock",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick with WSMA
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.add_trace(go.Scatter(x=df.index, y=df["WSMA"], mode='lines', name="WSMA"))

                fig.update_layout(title=f"Stock {symbol} Candlestick Chart with WSMA",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)       
        if pred_option_Technical_Indicators == "Z Score Indicator (Z Score)":
            st.success("This program allows you to view the WMA over time of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
            
                # Read data
                df = yf.download(symbol, start, end)
            
                # Read data
                df = yf.download(symbol, start, end)

                from scipy.stats import zscore

                df["z_score"] = zscore(df["Adj Close"])

                # Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                # Z-Score Chart
                fig.add_trace(go.Scatter(x=df.index, y=df["z_score"], mode='lines', name="Z-Score"))
                fig.update_layout(title="Z-Score for " + symbol,
                                xaxis_title="Date",
                                yaxis_title="Z-Score",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick with Z-Score
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.add_trace(go.Scatter(x=df.index, y=df["z_score"], mode='lines', name="Z-Score"))
                fig.update_layout(title="Stock " + symbol + " Candlestick Chart with Z-Score",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                
                st.plotly_chart(fig)
        if pred_option_Technical_Indicators == "Absolute Price Oscillator (APO)":
            st.success("This program allows you to view the WMA over time of a ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                df["HL"] = (df["High"] + df["Low"]) / 2
                df["HLC"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3
                df["HLCC"] = (df["High"] + df["Low"] + df["Adj Close"] + df["Adj Close"]) / 4
                df["OHLC"] = (df["Open"] + df["High"] + df["Low"] + df["Adj Close"]) / 4

                df["Long_Cycle"] = df["Adj Close"].rolling(20).mean()
                df["Short_Cycle"] = df["Adj Close"].rolling(5).mean()
                df["APO"] = df["Long_Cycle"] - df["Short_Cycle"]

                # Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                # Absolute Price Oscillator (APO) Chart
                fig.add_trace(go.Scatter(x=df.index, y=df["APO"], mode='lines', name="Absolute Price Oscillator", line=dict(color='green')))
                fig.update_layout(title="Absolute Price Oscillator (APO) for " + symbol,
                                xaxis_title="Date",
                                yaxis_title="APO",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick with Absolute Price Oscillator (APO)
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.add_trace(go.Scatter(x=df.index, y=df["APO"], mode='lines', name="Absolute Price Oscillator", line=dict(color='green')))
                fig.update_layout(title="Stock " + symbol + " Candlestick Chart with APO",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Acceleration Bands":
            st.success("This program allows you to view the Acclerations bands of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)
            

                n = 7
                UBB = df["High"] * (1 + 4 * (df["High"] - df["Low"]) / (df["High"] + df["Low"]))
                df["Upper_Band"] = UBB.rolling(n, center=False).mean()
                df["Middle_Band"] = df["Adj Close"].rolling(n).mean()
                LBB = df["Low"] * (1 - 4 * (df["High"] - df["Low"]) / (df["High"] + df["Low"]))
                df["Lower_Band"] = LBB.rolling(n, center=False).mean()

                # Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.add_trace(go.Scatter(x=df.index, y=df["Upper_Band"], mode='lines', name='Upper Band'))
                fig.add_trace(go.Scatter(x=df.index, y=df["Middle_Band"], mode='lines', name='Middle Band'))
                fig.add_trace(go.Scatter(x=df.index, y=df["Lower_Band"], mode='lines', name='Lower Band'))
                fig.update_layout(title="Stock Closing Price of " + str(n) + "-Day Acceleration Bands",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick Chart
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.add_trace(go.Scatter(x=df.index, y=df["Upper_Band"], mode='lines', name='Upper Band'))
                fig.add_trace(go.Scatter(x=df.index, y=df["Middle_Band"], mode='lines', name='Middle Band'))
                fig.add_trace(go.Scatter(x=df.index, y=df["Lower_Band"], mode='lines', name='Lower Band'))
                fig.update_layout(title="Stock " + symbol + " Closing Price with Acceleration Bands",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                
                st.plotly_chart(fig)


        if pred_option_Technical_Indicators == "Accumulation Distribution Line":
            st.success("This program allows you to view the ADL of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)
                
                df["MF Multiplier"] = (2 * df["Adj Close"] - df["Low"] - df["High"]) / (
                    df["High"] - df["Low"]
                )
                df["MF Volume"] = df["MF Multiplier"] * df["Volume"]
                df["ADL"] = df["MF Volume"].cumsum()
                df = df.drop(["MF Multiplier", "MF Volume"], axis=1)

                # Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=df.index, y=df["ADL"], mode='lines', name='Accumulation Distribution Line'))
                fig.update_layout(yaxis2=dict(title="Accumulation Distribution Line"))

                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color='rgba(0, 0, 255, 0.4)'))
                fig.update_layout(yaxis3=dict(title="Volume"))

                st.plotly_chart(fig)

                # Candlestick Chart
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=df.index, y=df["ADL"], mode='lines', name='Accumulation Distribution Line'))
                fig.update_layout(yaxis2=dict(title="Accumulation Distribution Line"))

                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color='rgba(0, 0, 255, 0.4)'))
                fig.update_layout(yaxis3=dict(title="Volume"))

                st.plotly_chart(fig)
            
        if pred_option_Technical_Indicators == "Aroon":
            st.success("This program allows you to view the Aroon of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                df = yf.download(symbol, start, end)

                n = 25
                high_max = lambda xs: np.argmax(xs[::-1])
                low_min = lambda xs: np.argmin(xs[::-1])

                df["Days since last High"] = (
                    df["High"]
                    .rolling(center=False, min_periods=0, window=n)
                    .apply(func=high_max)
                    .astype(int)
                )

                df["Days since last Low"] = (
                    df["Low"]
                    .rolling(center=False, min_periods=0, window=n)
                    .apply(func=low_min)
                    .astype(int)
                )

                df["Aroon_Up"] = ((25 - df["Days since last High"]) / 25) * 100
                df["Aroon_Down"] = ((25 - df["Days since last Low"]) / 25) * 100

                df = df.drop(["Days since last High", "Days since last Low"], axis=1)

                # Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Up"], mode='lines', name='Aroon UP', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Down"], mode='lines', name='Aroon DOWN', line=dict(color='red')))
                fig.update_layout(yaxis2=dict(title="Aroon"))

                st.plotly_chart(fig)

                # Candlestick Chart
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Up"], mode='lines', name='Aroon UP', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Down"], mode='lines', name='Aroon DOWN', line=dict(color='red')))
                fig.update_layout(yaxis2=dict(title="Aroon"))

                st.plotly_chart(fig)


        if pred_option_Technical_Indicators == "Aroon Oscillator":
            st.success("This program allows you to view the Aroon Oscillator bands of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                n = 25
                high_max = lambda xs: np.argmax(xs[::-1])
                low_min = lambda xs: np.argmin(xs[::-1])

                df["Days since last High"] = (
                    df["High"]
                    .rolling(center=False, min_periods=0, window=n)
                    .apply(func=high_max)
                    .astype(int)
                )

                df["Days since last Low"] = (
                    df["Low"]
                    .rolling(center=False, min_periods=0, window=n)
                    .apply(func=low_min)
                    .astype(int)
                )

                df["Aroon_Up"] = ((25 - df["Days since last High"]) / 25) * 100
                df["Aroon_Down"] = ((25 - df["Days since last Low"]) / 25) * 100

                df["Aroon_Oscillator"] = df["Aroon_Up"] - df["Aroon_Down"]

                df = df.drop(
                    ["Days since last High", "Days since last Low", "Aroon_Up", "Aroon_Down"], axis=1
                )

                # Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Oscillator"], mode='lines', name='Aroon Oscillator', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
                st.plotly_chart(fig)

                # Bar Chart
                fig = go.Figure()
                df["Positive"] = df["Aroon_Oscillator"] > 0
                fig.add_trace(go.Bar(x=df.index, y=df["Aroon_Oscillator"], marker_color=df.Positive.map({True: "green", False: "red"})))
                fig.add_shape(type="line", x0=df.index[0], y0=0, x1=df.index[-1], y1=0, line=dict(color="red", width=1, dash="dash"))

                fig.update_layout(title="Aroon Oscillator",
                                xaxis_title="Date",
                                yaxis_title="Aroon Oscillator",
                                legend=dict(x=0, y=1, traceorder="normal"))

                st.plotly_chart(fig)

                # Candlestick Chart
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=df.index, y=df["Aroon_Oscillator"], mode='lines', name='Aroon Oscillator', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Average Directional Index (ADX)":
            st.success("This program allows you to view the ADX  of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Simple Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                st.plotly_chart(fig)

                # ADX calculation
                adx = ta.ADX(df["High"], df["Low"], df["Adj Close"], timeperiod=14)
                adx = adx.dropna()

                # Line Chart with ADX
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=adx.index, y=adx, mode='lines', name='ADX'))
                fig.add_shape(type="line", x0=adx.index[0], y0=20, x1=adx.index[-1], y1=20, line=dict(color="red", width=1, dash="dash"))
                fig.add_shape(type="line", x0=adx.index[0], y0=50, x1=adx.index[-1], y1=50, line=dict(color="red", width=1, dash="dash"))
                st.plotly_chart(fig)

                # Candlestick Chart with ADX
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.update_layout(title="Stock " + symbol + " Candlestick Chart with ADX",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=adx.index, y=adx, mode='lines', name='ADX'))
                fig.add_shape(type="line", x0=adx.index[0], y0=20, x1=adx.index[-1], y1=20, line=dict(color="red", width=1, dash="dash"))
                fig.add_shape(type="line", x0=adx.index[0], y0=50, x1=adx.index[-1], y1=50, line=dict(color="red", width=1, dash="dash"))
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Average True Range (ATR)":
            st.success("This program allows you to view the ATR of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)
    
                n = 14
                df["HL"] = df["High"] - df["Low"]
                df["HC"] = abs(df["High"] - df["Adj Close"].shift())
                df["LC"] = abs(df["Low"] - df["Adj Close"].shift())
                df["TR"] = df[["HL", "HC", "LC"]].max(axis=1)
                df["ATR"] = df["TR"].rolling(n).mean()
                df = df.drop(["HL", "HC", "LC", "TR"], axis=1)

                # Simple Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # ATR Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], mode='lines', name='ATR'))
                fig.add_shape(type="line", x0=df.index[0], y0=1, x1=df.index[-1], y1=1, line=dict(color="black", width=1, dash="dash"))
                fig.update_layout(title="Average True Range (ATR)",
                                xaxis_title="Date",
                                yaxis_title="ATR",
                                legend=dict(x=0, y=1, traceorder="normal"))
                
                st.plotly_chart(fig)

                # Candlestick Chart with ATR
                fig = go.Figure()

                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.update_layout(title="Stock " + symbol + " Candlestick Chart with ATR",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], mode='lines', name='ATR'))
                fig.add_shape(type="line", x0=df.index[0], y0=1, x1=df.index[-1], y1=1, line=dict(color="black", width=1, dash="dash"))
                
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Balance of Power":
            st.success("This program allows you to view the BoP of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                df["BOP"] = (df["Adj Close"] - df["Open"]) / (df["High"] - df["Low"])

                # Simple Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.add_trace(go.Scatter(x=df.index, y=[df["Adj Close"].mean()] * len(df), mode='lines', name='Mean', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df.index, y=df["Low"], mode='lines', name='Low', line=dict(color='blue', dash='dash')))
                fig.add_trace(go.Scatter(x=df.index, y=df["High"], mode='lines', name='High', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name='Volume', marker_color='#0079a3', opacity=0.4))

                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # BOP Bar Chart
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df.index, y=df["BOP"], name='Balance of Power', marker_color=df["BOP"].apply(lambda x: 'green' if x >= 0 else 'red')))
                fig.update_layout(title="Balance of Power",
                                xaxis_title="Date",
                                yaxis_title="BOP",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick Chart with BOP
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df["BOP"], mode='lines', name='Balance of Power', line=dict(color='black')))
                fig.update_layout(title="Stock " + symbol + " Candlestick Chart with BOP",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Beta Indicator":
            st.success("This program allows you to view the Beta Indicator of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                market = "^GSPC"
                # Read data
                df = yf.download(symbol, start, end)
                mk = yf.download(market, start, end)

                df["Returns"] = df["Adj Close"].pct_change().dropna()
                mk["Returns"] = mk["Adj Close"].pct_change().dropna()

                n = 5
                covar = df["Returns"].rolling(n).cov(mk["Returns"])
                variance = mk["Returns"].rolling(n).var()
                df["Beta"] = covar / variance

                # Stock Closing Price Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.update_layout(title="Stock " + symbol + " Closing Price",
                                xaxis_title="Date",
                                yaxis_title="Price")
                st.plotly_chart(fig)

                # Beta Line Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Beta"], mode='lines', name='Beta', line=dict(color='red')))
                fig.update_layout(title="Beta",
                                xaxis_title="Date",
                                yaxis_title="Beta")
                st.plotly_chart(fig)

                # Candlestick Chart with Beta
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))
                fig.update_layout(title="Stock " + symbol + " Candlestick Chart with Beta",
                                xaxis_title="Date",
                                yaxis_title="Price")
                fig.add_trace(go.Scatter(x=df.index, y=df["Beta"], mode='lines', name='Beta', line=dict(color='red')))
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Bollinger Bands":
            st.success("This program allows you to view the Acclerations bands of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
            
                # Read data
                df = yf.download(symbol, start, end)
                df["VolumePositive"] = df["Open"] < df["Adj Close"]

                n = 20
                MA = pd.Series(df["Adj Close"].rolling(n).mean())
                STD = pd.Series(df["Adj Close"].rolling(n).std())
                bb1 = MA + 2 * STD
                df["Upper Bollinger Band"] = pd.Series(bb1)
                bb2 = MA - 2 * STD
                df["Lower Bollinger Band"] = pd.Series(bb2)

                # Bollinger Bands Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.add_trace(go.Scatter(x=df.index, y=df["Upper Bollinger Band"], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df.index, y=df["Lower Bollinger Band"], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
                fig.update_layout(title=f"{symbol} Bollinger Bands",
                                xaxis_title="Date",
                                yaxis_title="Price")
                st.plotly_chart(fig)

                # Candlestick Chart with Bollinger Bands
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick')])
                fig.add_trace(go.Scatter(x=df.index, y=df["Upper Bollinger Band"], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df.index, y=df["Lower Bollinger Band"], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
                fig.update_layout(title="Candlestick Chart with Bollinger Bands",
                                xaxis_title="Date",
                                yaxis_title="Price")
                st.plotly_chart(fig)

                # Volume Chart
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name='Volume', marker_color=df.VolumePositive.map({True: "green", False: "red"})))
                fig.update_layout(title="Volume",
                                xaxis_title="Date",
                                yaxis_title="Volume")
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Bollinger Bandwidth":
            st.success("This program allows you to view the Bollinger bandwidth of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                n = 20
                MA = pd.Series(df["Adj Close"].rolling(n).mean())
                STD = pd.Series(df["Adj Close"].rolling(n).std())
                bb1 = MA + 2 * STD
                df["Upper Bollinger Band"] = pd.Series(bb1)
                bb2 = MA - 2 * STD
                df["Lower Bollinger Band"] = pd.Series(bb2)
                df["SMA"] = df["Adj Close"].rolling(n).mean()

                df["BBWidth"] = ((df["Upper Bollinger Band"] - df["Lower Bollinger Band"]) / df["SMA"] * 100)

                # Bollinger Bands Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig.add_trace(go.Scatter(x=df.index, y=df["Upper Bollinger Band"], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df.index, y=df["Lower Bollinger Band"], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"].rolling(20).mean(), mode='lines', name='Mean Average', line=dict(color='orange', dash='dash')))
                fig.update_layout(title=f"{symbol} Bollinger Bands",
                                xaxis_title="Date",
                                yaxis_title="Price")
                st.plotly_chart(fig)

                # BB Width Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["BBWidth"], mode='lines', name='BB Width', line=dict(color='black')))
                fig.add_trace(go.Scatter(x=df.index, y=df["BBWidth"].rolling(20).mean(), mode='lines', name='200 Moving Average', line=dict(color='darkblue')))
                fig.update_layout(title="Bollinger Bands Width",
                                xaxis_title="Date",
                                yaxis_title="BB Width")
                st.plotly_chart(fig)



        if pred_option_Technical_Indicators == "Breadth Indicator":
            st.success("This program allows you to view the Breadth Indicator of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
        
                # Read data
                df = yf.download(symbol, start, end)

                #df["Adj Close"][1:]

                # ## On Balance Volume
                OBV = ta.OBV(df["Adj Close"], df["Volume"])

                Advances = q.get("URC/NYSE_ADV", start_date="2017-07-27")["Numbers of Stocks"]
                Declines = q.get("URC/NYSE_DEC", start_date="2017-07-27")["Numbers of Stocks"]

                adv_vol = q.get("URC/NYSE_ADV_VOL", start_date="2017-07-27")["Numbers of Stocks"]
                dec_vol = q.get("URC/NYSE_DEC_VOL", start_date="2017-07-27")["Numbers of Stocks"]

                data = pd.DataFrame()
                data["Advances"] = Advances
                data["Declines"] = Declines
                data["adv_vol"] = adv_vol
                data["dec_vol"] = dec_vol

                data["Net_Advances"] = data["Advances"] - data["Declines"]
                data["Ratio_Adjusted"] = (
                    data["Net_Advances"] / (data["Advances"] + data["Declines"])
                ) * 1000
                data["19_EMA"] = ta.EMA(data["Ratio_Adjusted"], timeperiod=19)
                data["39_EMA"] = ta.EMA(data["Ratio_Adjusted"], timeperiod=39)
                data["RANA"] = (
                    (data["Advances"] - data["Declines"]) / (data["Advances"] + data["Declines"]) * 1000
                )

                # Finding the TRIN Value
                data["ad_ratio"] = data["Advances"].divide(data["Declines"])  # AD Ratio
                data["ad_vol"] = data["adv_vol"].divide(data["dec_vol"])  # AD Volume Ratio
                data["TRIN"] = data["ad_ratio"].divide(data["adv_vol"])  # TRIN Value

                # Function to calculate Force Index
                def ForceIndex(data, n):
                    ForceIndex = pd.Series(df["Adj Close"].diff(n) * df["Volume"], name="ForceIndex")
                    data = data.join(ForceIndex)
                    return data

                # Calculate Force Index
                n = 10
                ForceIndex = ForceIndex(df, n)
                ForceIndex = ForceIndex["ForceIndex"]

                # Market Price Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Close Price'))
                fig.update_layout(title="Market Price Chart",
                                xaxis_title="Date",
                                yaxis_title="Close Price")
                st.plotly_chart(fig)

                # Force Index
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=ForceIndex, mode='lines', name='Force Index'))
                fig.update_layout(title="Force Index",
                                xaxis_title="Date",
                                yaxis_title="Force Index")
                st.plotly_chart(fig)


                # Function to calculate Chaikin Oscillator
                def Chaikin(data):
                    money_flow_volume = (
                        (2 * data["Adj Close"] - data["High"] - data["Low"])
                        / (data["High"] - data["Low"])
                        * data["Volume"]
                    )
                    ad = money_flow_volume.cumsum()
                    Chaikin = pd.Series(
                        ad.ewm(com=(3 - 1) / 2).mean() - ad.ewm(com=(10 - 1) / 2).mean(), name="Chaikin"
                    )
                    #data["Chaikin"] = data.join(Chaikin)
                    data["Chaikin"] = Chaikin
                    return data


                # Calculate Chaikin Oscillator
                Chaikin(df)
            

                # Chaikin Oscillator Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Chaikin"], mode='lines', name='Chaikin Oscillator'))
                fig.update_layout(title="Chaikin Oscillator",
                                xaxis_title="Date",
                                yaxis_title="Chaikin Oscillator")
                
                st.plotly_chart(fig)


                # Calculate Cumulative Volume Index
                data["CVI"] = data["Net_Advances"].shift(1) + (data["Advances"] - data["Declines"])

                # Cumulative Volume Index Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data["CVI"], mode='lines', name='Cumulative Volume Index'))
                fig.update_layout(title="Cumulative Volume Index",
                                xaxis_title="Date",
                                yaxis_title="CVI")
                
                st.plotly_chart(fig)


        if pred_option_Technical_Indicators == "Candle Absolute Returns":
            st.success("This program allows you to view the Candle absolute returns  of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
        
                # Read data
                df = yf.download(symbol, start, end)
                df["Absolute_Return"] = (
                    100 * (df["Adj Close"] - df["Adj Close"].shift(1)) / df["Adj Close"].shift(1)
                )

                # Plot the closing price
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name="Closing Price"))
                fig.update_layout(title="Stock " + symbol + " Closing Price", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)

                # Plot the Absolute Return
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Absolute_Return"], mode="lines", name="Absolute Return", line=dict(color="red")))
                fig.update_layout(title="Absolute Return", xaxis_title="Date", yaxis_title="Absolute Return")
                
                st.plotly_chart(fig)
        if pred_option_Technical_Indicators == "Central Pivot Range (CPR)":
            st.success("This program allows you to view the CPR of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate CPR
                df["Pivot"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3.0
                df["BC"] = (df["High"] + df["Low"]) / 2.0
                df["TC"] = (df["Pivot"] - df["BC"]) + df["Pivot"]

                # Plot candlestick with CPR
                candlestick = go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'],
                                            name='Candlestick')

                pivot = go.Scatter(x=df.index, y=df["Pivot"], mode='lines', name='Pivot')
                bc = go.Scatter(x=df.index, y=df["BC"], mode='lines', name='BC')
                tc = go.Scatter(x=df.index, y=df["TC"], mode='lines', name='TC')

                data = [candlestick, pivot, bc, tc]

                layout = go.Layout(title=f'Stock {symbol} Closing Price with Central Pivot Range (CPR)',
                                xaxis=dict(title='Date'),
                                yaxis=dict(title='Price'),
                                showlegend=True)

                fig = go.Figure(data=data, layout=layout)
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Chaikin Money Flow":
            st.success("This program allows you to view the Chaikin Money Flow (CMF) of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate CMF
                n = 20
                df["MF_Multiplier"] = (2 * df["Adj Close"] - df["Low"] - df["High"]) / (df["High"] - df["Low"])
                df["MF_Volume"] = df["MF_Multiplier"] * df["Volume"]
                df["CMF"] = df["MF_Volume"].rolling(n).sum() / df["Volume"].rolling(n).sum()
                df = df.drop(["MF_Multiplier", "MF_Volume"], axis=1)

                # Plot CMF
                cmf_chart = go.Scatter(x=df.index, y=df["CMF"], mode='lines', name='Chaikin Money Flow')

                layout = go.Layout(title=f'Chaikin Money Flow for Stock {symbol}',
                                xaxis=dict(title='Date'),
                                yaxis=dict(title='CMF'),
                                showlegend=True)

                fig = go.Figure(data=[cmf_chart], layout=layout)
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Chaikin Oscillator":
            st.success("This program allows you to view the Chaikin Oscillator of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Chaikin Oscillator
                df["MF_Multiplier"] = (2 * df["Adj Close"] - df["Low"] - df["High"]) / (df["High"] - df["Low"])
                df["MF_Volume"] = df["MF_Multiplier"] * df["Volume"]
                df["ADL"] = df["MF_Volume"].cumsum()
                df["ADL_3_EMA"] = df["ADL"].ewm(ignore_na=False, span=3, min_periods=2, adjust=True).mean()
                df["ADL_10_EMA"] = df["ADL"].ewm(ignore_na=False, span=10, min_periods=9, adjust=True).mean()
                df["Chaikin_Oscillator"] = df["ADL_3_EMA"] - df["ADL_10_EMA"]
                df = df.drop(["MF_Multiplier", "MF_Volume", "ADL", "ADL_3_EMA", "ADL_10_EMA"], axis=1)

                # Plot Chaikin Oscillator
                co_chart = go.Scatter(x=df.index, y=df["Chaikin_Oscillator"], mode='lines', name='Chaikin Oscillator')

                layout = go.Layout(title=f'Chaikin Oscillator for Stock {symbol}',
                                xaxis=dict(title='Date'),
                                yaxis=dict(title='Chaikin Oscillator'),
                                showlegend=True)

                fig = go.Figure(data=[co_chart], layout=layout)
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Commodity Channel Index (CCI)":
            st.success("This program allows you to view the Commodity Channel Index (CCI) of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate CCI
                n = 20
                df["TP"] = (df["High"] + df["Low"] + df["Adj Close"]) / 3
                df["SMA_TP"] = df["TP"].rolling(n).mean()
                df["SMA_STD"] = df["TP"].rolling(n).std()
                df["CCI"] = (df["TP"] - df["SMA_TP"]) / (0.015 * df["SMA_STD"])
                df = df.drop(["TP", "SMA_TP", "SMA_STD"], axis=1)

                # Plot CCI
                cci_chart = go.Scatter(x=df.index, y=df["CCI"], mode='lines', name='Commodity Channel Index (CCI)')

                layout = go.Layout(title=f'Commodity Channel Index (CCI) for Stock {symbol}',
                                xaxis=dict(title='Date'),
                                yaxis=dict(title='CCI'),
                                showlegend=True)

                fig = go.Figure(data=[cci_chart], layout=layout)
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Correlation Coefficient":
            st.success("This program allows you to view the Correlation Coefficient between two tickers over time")
            symbol1 = st.text_input("Enter the first ticker")
            symbol2 = st.text_input("Enter the second ticker")
            if symbol1 and symbol2:
                message = (f"Tickers captured : {symbol1}, {symbol2}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                start = start_date
                end = end_date

                # Read data
                df1 = yf.download(symbol1, start, end)
                df2 = yf.download(symbol2, start, end)

                # Calculate correlation coefficient
                cc = df1["Adj Close"].corr(df2["Adj Close"])

                # Plot correlation coefficient
                cc_chart = go.Scatter(x=df1.index, y=cc, mode='lines', name='Correlation Coefficient')

                layout = go.Layout(title=f'Correlation Coefficient between {symbol1} and {symbol2}',
                                xaxis=dict(title='Date'),
                                yaxis=dict(title='Correlation Coefficient'),
                                showlegend=True)

                fig = go.Figure(data=[cc_chart], layout=layout)
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Covariance":
            st.success("This program allows you to view the Covariance between two tickers over time")
            symbol1 = st.text_input("Enter the first ticker")
            symbol2 = st.text_input("Enter the second ticker")
            if symbol1 and symbol2:
                message = (f"Tickers captured : {symbol1}, {symbol2}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                start = start_date
                end = end_date

                # Read data
                df1 = yf.download(symbol1, start, end)
                df2 = yf.download(symbol2, start, end)

                # Calculate covariance
                c = df1["Adj Close"].cov(df2["Adj Close"])

                # Plot covariance
                cov_chart = go.Scatter(x=df1.index, y=c, mode='lines', name='Covariance')

                layout = go.Layout(title=f'Covariance between {symbol1} and {symbol2}',
                                xaxis=dict(title='Date'),
                                yaxis=dict(title='Covariance'),
                                showlegend=True)

                fig = go.Figure(data=[cov_chart], layout=layout)
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Detrended Price Oscillator (DPO)":
            st.success("This program allows you to view the Detrended Price Oscillator (DPO) of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                n = 15
                df["DPO"] = (df["Adj Close"].shift(int((0.5 * n) + 1)) - df["Adj Close"].rolling(n).mean())

                # Plot DPO
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                    open=df['Open'],
                                                    high=df['High'],
                                                    low=df['Low'],
                                                    close=df['Adj Close'],
                                                    name='Candlestick'),
                                    go.Scatter(x=df.index,
                                                y=df['DPO'],
                                                mode='lines',
                                                name='DPO')])
                fig.update_layout(title=f"Detrended Price Oscillator (DPO) for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Donchian Channel":
            st.success("This program allows you to view the Donchian Channel of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                df["Upper_Channel_Line"] = pd.Series.rolling(df["High"], window=20).max()
                df["Lower_Channel_Line"] = pd.Series.rolling(df["Low"], window=20).min()
                df["Middle_Channel_Line"] = (df["Upper_Channel_Line"] + df["Lower_Channel_Line"]) / 2
                df = df.dropna()

                # Plot Donchian Channel
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                    open=df['Open'],
                                                    high=df['High'],
                                                    low=df['Low'],
                                                    close=df['Adj Close'],
                                                    name='Candlestick'),
                                    go.Scatter(x=df.index,
                                                y=df['Upper_Channel_Line'],
                                                mode='lines',
                                                name='Upper Channel Line'),
                                    go.Scatter(x=df.index,
                                                y=df['Lower_Channel_Line'],
                                                mode='lines',
                                                name='Lower Channel Line'),
                                    go.Scatter(x=df.index,
                                                y=df['Middle_Channel_Line'],
                                                mode='lines',
                                                name='Middle Channel Line')])
                fig.update_layout(title=f"Donchian Channel for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Double Exponential Moving Average (DEMA)":
            st.success("This program allows you to view the Double Exponential Moving Average (DEMA) of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                df["EMA"] = ta.EMA(df["Adj Close"], timeperiod=5)
                df["EMA_S"] = ta.EMA(df["EMA"], timeperiod=5)
                df["DEMA"] = (2 * df["EMA"]) - df["EMA_S"]

                # Plot DEMA
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                    open=df['Open'],
                                                    high=df['High'],
                                                    low=df['Low'],
                                                    close=df['Adj Close'],
                                                    name='Candlestick'),
                                    go.Scatter(x=df.index,
                                                y=df['DEMA'],
                                                mode='lines',
                                                name='DEMA')])
                fig.update_layout(title=f"Double Exponential Moving Average (DEMA) for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Dynamic Momentum Index":
            st.success("This program allows you to view the Dynamic Momentum Index (DMI) of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                df["sd"] = df["Adj Close"].rolling(5).std()
                df["asd"] = df["sd"].rolling(10).mean()
                df["DMI"] = 14 / (df["sd"] / df["asd"])
                df = df.drop(["sd", "asd"], axis=1)

                # Plot DMI
                fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                    open=df['Open'],
                                                    high=df['High'],
                                                    low=df['Low'],
                                                    close=df['Adj Close'],
                                                    name='Candlestick'),
                                    go.Scatter(x=df.index,
                                                y=df['DMI'],
                                                mode='lines',
                                                name='Dynamic Momentum Index')])
                fig.update_layout(title=f"Dynamic Momentum Index (DMI) for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Ease of Movement":
            st.success("This program allows you to view the Ease of Movement (EVM) indicator of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)
                # Create a function for Ease of Movement
                def EVM(data, ndays):
                    dm = ((data["High"] + data["Low"]) / 2) - (
                        (data["High"].shift(1) + data["Low"].shift(1)) / 2
                    )
                    br = (data["Volume"] / 100000000) / ((data["High"] - data["Low"]))
                    EVM = dm / br
                    EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name="EVM")
                    data = data.join(EVM_MA)
                    return data

                # Compute the 14-day Ease of Movement for stock
                n = 14
                Stock_EVM = EVM(df, n)
                EVM = Stock_EVM["EVM"]

                # Compute EVM
                df = EVM(df, 14)

                # Plot EVM
                fig = go.Figure(data=[
                    go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'),
                    go.Scatter(x=df.index,
                            y=df['EVM'],
                            mode='lines',
                            name='Ease of Movement')
                ])
                fig.update_layout(title=f"Ease of Movement (EVM) for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Force Index":
            st.success("This program allows you to view the Force Index of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                n = 13
                df["FI_1"] = (df["Adj Close"] - df["Adj Close"].shift()) * df["Volume"]
                df["FI_13"] = df["FI_1"].ewm(ignore_na=False, span=n, min_periods=n, adjust=True).mean()

                # Plot Force Index
                fig = go.Figure(data=[
                    go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Adj Close'],
                                name='Candlestick'),
                    go.Scatter(x=df.index,
                            y=df['FI_13'],
                            mode='lines',
                            name='Force Index')
                ])
                fig.update_layout(title=f"Force Index for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Geometric Return Indicator":
            st.success("This program allows you to view the Geometric Return Indicator of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Compute Geometric Return Indicator
                n = 10
                df["Geometric_Return"] = pd.Series(df["Adj Close"]).rolling(n).apply(gmean)

                # Plot Geometric Return Indicator with Candlestick graph
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Adj Close'],
                                            name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Geometric_Return'], mode='lines', name='Geometric Return Indicator'))
                fig.update_layout(title=f"Geometric Return Indicator for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Golden/Death Cross":
            st.success("This program allows you to view the Golden/Death Cross of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Compute Golden/Death Cross
                df["MA_50"] = df["Adj Close"].rolling(center=False, window=50).mean()
                df["MA_200"] = df["Adj Close"].rolling(center=False, window=200).mean()
                df["diff"] = df["MA_50"] - df["MA_200"]

                # Plot Golden/Death Cross with Candlestick graph
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Adj Close'],
                                            name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], mode='lines', name='MA_50'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], mode='lines', name='MA_200'))
                fig.update_layout(title=f"Golden/Death Cross for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "High Minus Low":
            st.success("This program allows you to view the High Minus Low of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Compute High Minus Low
                df["H-L"] = df["High"] - df["Low"]

                # Plot High Minus Low with Candlestick graph
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Adj Close'],
                                            name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df['H-L'], mode='lines', name='High Minus Low'))
                fig.update_layout(title=f"High Minus Low for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Hull Moving Average":
            st.success("This program allows you to view the Hull Moving Average (HMA) of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Compute Hull Moving Average
                period = 20
                df['WMA'] = df['Adj Close'].rolling(window=period).mean()
                half_period = int(period / 2)
                sqrt_period = int(np.sqrt(period))
                df['Weighted_MA'] = df['Adj Close'].rolling(window=half_period).mean() * 2 - df['Adj Close'].rolling(window=period).mean()
                df['HMA'] = df['Weighted_MA'].rolling(window=sqrt_period).mean()

                # Plot Hull Moving Average with Candlestick graph
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Adj Close'],
                                            name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], mode='lines', name='Hull Moving Average'))
                fig.update_layout(title=f"Hull Moving Average (HMA) for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Keltner Channels":
            st.success("This program allows you to visualize Keltner Channels for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Keltner Channels
                n = 20
                df["EMA"] = ta.EMA(df["Adj Close"], timeperiod=n)
                df["ATR"] = ta.ATR(df["High"], df["Low"], df["Adj Close"], timeperiod=10)
                df["Upper Line"] = df["EMA"] + 2 * df["ATR"]
                df["Lower Line"] = df["EMA"] - 2 * df["ATR"]

                # Plot Keltner Channels
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Adj Close'],
                                            name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode='lines', name='EMA'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper Line'], mode='lines', name='Upper Line'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower Line'], mode='lines', name='Lower Line'))
                fig.update_layout(title=f"Keltner Channels for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Linear Regression":
            st.success("This program allows you to visualize Linear Regression for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Linear Regression
                avg = df["Adj Close"].mean()
                df["Linear_Regression"] = avg - (df["Adj Close"].mean() - df["Adj Close"]) / 20

                # Plot Linear Regression
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Adj Close'],
                                            name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Regression'], mode='lines', name='Linear Regression'))
                fig.update_layout(title=f"Linear Regression for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Linear Regression Slope":
            st.success("This program allows you to visualize Linear Regression Slope for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Linear Regression Slope
                avg1 = df["Adj Close"].mean()
                avg2 = df["Adj Close"].mean()
                df["AVGS1_S1"] = avg1 - df["Adj Close"]
                df["AVGS2_S2"] = avg2 - df["Adj Close"]
                df["Average_SQ"] = df["AVGS1_S1"] ** 2
                df["AVG_AVG"] = df["AVGS1_S1"] * df["AVGS2_S2"]
                sum_sq = df["Average_SQ"].sum()
                sum_avg = df["AVG_AVG"].sum()
                slope = sum_avg / sum_sq
                intercept = avg2 - (slope * avg1)
                df["Linear_Regression"] = intercept + slope * (df["Adj Close"])
                df = df.drop(["AVGS1_S1", "AVGS2_S2", "Average_SQ", "AVG_AVG"], axis=1)

                # Plot Linear Regression Slope
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Adj Close'],
                                            name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Regression'], mode='lines', name='Linear Regression Slope'))
                fig.update_layout(title=f"Linear Regression Slope for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Linear Weighted Moving Average (LWMA)":
            st.success("This program allows you to visualize Linearly Weighted Moving Average (LWMA) for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Linear Weighted Moving Average (LWMA)
                def linear_weight_moving_average(close, n):
                    lwma = [np.nan] * n
                    for i in range(n, len(close)):
                        lwma.append(
                            (close[i - n : i] * (np.arange(n) + 1)).sum() / (np.arange(n) + 1).sum()
                        )
                    return lwma

                period = 14
                df["LWMA"] = linear_weight_moving_average(df["Adj Close"], period)

                # Plot Linear Weighted Moving Average (LWMA)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Adj Close'],
                                            name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df['LWMA'], mode='lines', name='LWMA'))
                fig.update_layout(title=f"Linear Weighted Moving Average (LWMA) for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "McClellan Oscillator":
            st.success("This program allows you to visualize McClellan Oscillator for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate McClellan Oscillator
                ema_19 = ta.EMA(df["Advancing Issues"] - df["Declining Issues"], timeperiod=19)
                ema_39 = ta.EMA(df["Advancing Issues"] - df["Declining Issues"], timeperiod=39)
                mcclellan_oscillator = ema_19 - ema_39

                # Plot McClellan Oscillator
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=mcclellan_oscillator, mode='lines', name='McClellan Oscillator'))
                fig.update_layout(title=f"McClellan Oscillator for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Value',
                                template='plotly_dark')
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Momentum":
            st.success("This program allows you to visualize Momentum for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Momentum
                period = 14
                df['Momentum'] = df['Adj Close'].diff(period)

                # Plot Momentum
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Adj Close'],
                                            name='Candlestick'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Momentum'], mode='lines', name='Momentum'))
                fig.update_layout(title=f"Momentum for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price',
                                template='plotly_dark')
                st.plotly_chart(fig)        

        if pred_option_Technical_Indicators == "Moving Average Envelopes":
            st.success("This program allows you to visualize Moving Average Envelopes for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                df["20SMA"] = ta.SMA(df["Adj Close"], timeperiod=20)
                df["Upper_Envelope"] = df["20SMA"] + (df["20SMA"] * 0.025)
                df["Lower_Envelope"] = df["20SMA"] - (df["20SMA"] * 0.025)

                # Plot Line Chart with Moving Average Envelopes
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Upper_Envelope"], mode='lines', name='Upper Envelope', line=dict(color='blue')))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Lower_Envelope"], mode='lines', name='Lower Envelope', line=dict(color='red')))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"].rolling(20).mean(), mode='lines', name='Moving Average', line=dict(color='orange', dash='dash')))
                fig_line.update_layout(title=f"Stock of Moving Average Envelopes for {symbol}",
                                xaxis_title='Date',
                                yaxis_title='Price')
                
                st.plotly_chart(fig_line)

                # Candlestick with Moving Average Envelopes
                fig_candlestick = go.Figure(data=[go.Candlestick(x=df.index,
                                                                open=df['Open'],
                                                                high=df['High'],
                                                                low=df['Low'],
                                                                close=df['Adj Close'],
                                                                name='Candlestick')])
                fig_candlestick.add_trace(go.Scatter(x=df.index, y=df["Upper_Envelope"], mode='lines', name='Upper Envelope', line=dict(color='blue')))
                fig_candlestick.add_trace(go.Scatter(x=df.index, y=df["Lower_Envelope"], mode='lines', name='Lower Envelope', line=dict(color='red')))
                fig_candlestick.add_trace(go.Scatter(x=df.index, y=df["Adj Close"].rolling(20).mean(), mode='lines', name='Moving Average', line=dict(color='orange', dash='dash')))
                fig_candlestick.update_layout(title=f"Stock {symbol} Candlestick with Moving Average Envelopes",
                                            xaxis_title='Date',
                                            yaxis_title='Price')
        
                st.plotly_chart(fig_candlestick)

        if pred_option_Technical_Indicators == "Moving Average High/Low":
            st.success("This program allows you to visualize Moving Average High/Low for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                n = 14  # number of periods
                df["MA_High"] = df["High"].rolling(n).mean()
                df["MA_Low"] = df["Low"].rolling(n).mean()

                # Plot Line Chart with Moving Average High/Low
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["MA_High"], mode='lines', name='Moving Average of High'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["MA_Low"], mode='lines', name='Moving Average of Low'))
                fig_line.update_layout(title=f"Stock {symbol} Moving Average High/Low",
                                xaxis_title='Date',
                                yaxis_title='Price')
                
                st.plotly_chart(fig_line)

        if pred_option_Technical_Indicators == "Moving Average Ribbon":
            st.success("This program allows you to visualize Moving Average Ribbon for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                df["MA10"] = df["Adj Close"].rolling(10).mean()
                df["MA20"] = df["Adj Close"].rolling(20).mean()
                df["MA30"] = df["Adj Close"].rolling(30).mean()
                df["MA40"] = df["Adj Close"].rolling(40).mean()
                df["MA50"] = df["Adj Close"].rolling(50).mean()
                df["MA60"] = df["Adj Close"].rolling(60).mean()

                # Plot Line Chart with Moving Average Ribbon
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["MA10"], mode='lines', name='MA10'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode='lines', name='MA20'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["MA30"], mode='lines', name='MA30'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["MA40"], mode='lines', name='MA40'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode='lines', name='MA50'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["MA60"], mode='lines', name='MA60'))
                fig_line.update_layout(title=f"Stock {symbol} Moving Average Ribbon",
                                xaxis_title='Date',
                                yaxis_title='Price')
                
                st.plotly_chart(fig_line)

        if pred_option_Technical_Indicators == "Moving Average Envelopes (MMA)":
            st.success("This program allows you to visualize Moving Average Envelopes (MMA) for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                df["20SMA"] = ta.SMA(df["Adj Close"], timeperiod=20)
                df["Upper_Envelope"] = df["20SMA"] + (df["20SMA"] * 0.025)
                df["Lower_Envelope"] = df["20SMA"] - (df["20SMA"] * 0.025)

                # Plot Line Chart with Moving Average Envelopes (MMA)
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Upper_Envelope"], mode='lines', name='Upper Envelope', line=dict(color='blue')))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Lower_Envelope"], mode='lines', name='Lower Envelope', line=dict(color='red')))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"].rolling(20).mean(), mode='lines', name='Moving Average', line=dict(color='orange', dash='dash')))
                fig_line.update_layout(title=f"Stock {symbol} Moving Average Envelopes (MMA)",
                                xaxis_title='Date',
                                yaxis_title='Price')
                
                st.plotly_chart(fig_line)

        if pred_option_Technical_Indicators == "Moving Linear Regression":
            st.success("This program allows you to visualize Moving Linear Regression for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                df['Slope'] = ta.LINEARREG_SLOPE(df['Adj Close'], timeperiod=14)

                # Plot Line Chart with Moving Linear Regression
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Slope"], mode='lines', name='Slope', line=dict(color='red')))
                fig_line.update_layout(title=f"Stock {symbol} Moving Linear Regression",
                                xaxis_title='Date',
                                yaxis_title='Price')
                
                st.plotly_chart(fig_line)

        if pred_option_Technical_Indicators == "New Highs/New Lows":
            st.success("This program allows you to visualize New Highs/New Lows for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                df = yf.download(symbol, start, end)

                df['52wHigh'] = df['Adj Close'].rolling(window=252).max()
                df['52wLow'] = df['Adj Close'].rolling(window=252).min()

                # Plot Line Chart with New Highs/New Lows
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Adj Close'))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["52wHigh"], mode='lines', name='52 Weeks High', line=dict(color='green')))
                fig_line.add_trace(go.Scatter(x=df.index, y=df["52wLow"], mode='lines', name='52 Weeks Low', line=dict(color='red')))
                fig_line.update_layout(title=f"Stock {symbol} New Highs/New Lows",
                                xaxis_title='Date',
                                yaxis_title='Price')
                
                st.plotly_chart(fig_line)

        if pred_option_Technical_Indicators == "Pivot Point":
            st.success("This program allows you to visualize Pivot Points for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                # Calculate Pivot Points
                PP = (dataset["High"] + dataset["Low"] + dataset["Close"]) / 3
                R1 = 2 * PP - dataset["Low"]
                S1 = 2 * PP - dataset["High"]
                R2 = PP + dataset["High"] - dataset["Low"]
                S2 = PP - dataset["High"] + dataset["Low"]
                R3 = dataset["High"] + 2 * (PP - dataset["Low"])
                S3 = dataset["Low"] - 2 * (dataset["High"] - PP)

                # Plot Pivot Points
                fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                                    open=dataset['Open'],
                                                    high=dataset['High'],
                                                    low=dataset['Low'],
                                                    close=dataset['Close'],
                                                    name='Candlesticks'),
                                    go.Scatter(x=dataset.index, y=PP, mode='lines', name='Pivot Point'),
                                    go.Scatter(x=dataset.index, y=R1, mode='lines', name='R1'),
                                    go.Scatter(x=dataset.index, y=S1, mode='lines', name='S1'),
                                    go.Scatter(x=dataset.index, y=R2, mode='lines', name='R2'),
                                    go.Scatter(x=dataset.index, y=S2, mode='lines', name='S2'),
                                    go.Scatter(x=dataset.index, y=R3, mode='lines', name='R3'),
                                    go.Scatter(x=dataset.index, y=S3, mode='lines', name='S3')])
                fig.update_layout(title=f"{symbol} Pivot Points",
                                xaxis_title='Date',
                                yaxis_title='Price')
                
                st.plotly_chart(fig)
    
        if pred_option_Technical_Indicators == "Price Channels":
            st.success("This program allows you to visualize Price Channels for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                # Calculate Price Channels
                rolling_high = dataset['High'].rolling(window=20).max()
                rolling_low = dataset['Low'].rolling(window=20).min()
                midline = (rolling_high + rolling_low) / 2

                # Plot Price Channels
                fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                                    open=dataset['Open'],
                                                    high=dataset['High'],
                                                    low=dataset['Low'],
                                                    close=dataset['Close'],
                                                    name='Candlesticks'),
                                    go.Scatter(x=dataset.index, y=rolling_high, mode='lines', name='Upper Channel'),
                                    go.Scatter(x=dataset.index, y=rolling_low, mode='lines', name='Lower Channel'),
                                    go.Scatter(x=dataset.index, y=midline, mode='lines', name='Midline')])
                fig.update_layout(title=f"{symbol} Price Channels",
                                xaxis_title='Date',
                                yaxis_title='Price')
                
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Price Relative":
            st.success("This program allows you to visualize Price Relative for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            benchmark_ticker = st.text_input("Enter the benchmark ticker")
            if ticker and benchmark_ticker:
                message = (f"Ticker captured : {ticker}, Benchmark Ticker : {benchmark_ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                benchmark = benchmark_ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)
                benchmark_data = yf.download(benchmark, start, end)

                # Calculate Price Relative
                price_relative = dataset['Adj Close'] / benchmark_data['Adj Close']

                # Plot Price Relative
                fig = go.Figure(data=[go.Scatter(x=price_relative.index, y=price_relative, mode='lines', name='Price Relative')])
                fig.update_layout(title=f"{symbol} Price Relative to {benchmark}",
                                xaxis_title='Date',
                                yaxis_title='Price Relative')
                
                st.plotly_chart(fig)


        if pred_option_Technical_Indicators == "Realized Volatility":
            st.success("This program allows you to visualize Realized Volatility for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = f"Ticker captured: {ticker}"
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                # Calculate Realized Volatility
                returns = dataset["Adj Close"].pct_change().dropna()
                realized_volatility = returns.std() * np.sqrt(252)  # Annualized volatility assuming 252 trading days

                # Plot Realized Volatility
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=returns.index, y=returns, mode='lines', name='Daily Returns'))
                fig.add_trace(go.Scatter(x=returns.index, y=np.ones(len(returns)) * realized_volatility, mode='lines', name='Realized Volatility', line=dict(color='red', dash='dash')))
                fig.update_layout(title=f"{symbol} Daily Returns and Realized Volatility",
                                xaxis_title="Date",
                                yaxis_title="Returns / Realized Volatility")
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Relative Volatility Index":
            st.success("This program allows you to visualize Relative Volatility Index for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = f"Ticker captured: {ticker}"
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                # Calculate Relative Volatility Index (RVI)
                n = 14  # Number of period
                change = dataset["Adj Close"].diff(1)
                gain = change.mask(change < 0, 0)
                loss = abs(change.mask(change > 0, 0))
                avg_gain = gain.rolling(n).std()
                avg_loss = loss.rolling(n).std()
                RS = avg_gain / avg_loss
                RVI = 100 - (100 / (1 + RS))

                # Plot RVI
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=RVI.index, y=RVI, mode='lines', name='Relative Volatility Index', line=dict(color='blue')))
                fig.add_shape(type="line", x0=RVI.index[0], y0=60, x1=RVI.index[-1], y1=60, line=dict(color="red", width=1, dash="dash"), name="Overbought")
                fig.add_shape(type="line", x0=RVI.index[0], y0=40, x1=RVI.index[-1], y1=40, line=dict(color="green", width=1, dash="dash"), name="Oversold")
                fig.update_layout(title=f"{symbol} Relative Volatility Index",
                                xaxis_title="Date",
                                yaxis_title="RVI")
                st.plotly_chart(fig)


        if pred_option_Technical_Indicators == "Smoothed Moving Average":
            st.success("This program allows you to visualize Smoothed Moving Average for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = f"Ticker captured: {ticker}"
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                n = 7
                dataset["SMMA"] = dataset["Adj Close"].ewm(alpha=1 / float(n)).mean()

                # Plot Smoothed Moving Average
                fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                                    open=dataset['Open'],
                                                    high=dataset['High'],
                                                    low=dataset['Low'],
                                                    close=dataset['Close'],
                                                    name='Candlesticks'),
                                    go.Scatter(x=dataset.index, y=dataset["SMMA"], mode='lines', name='Smoothed Moving Average', line=dict(color='red'))])
                fig.update_layout(title=f"{symbol} Smoothed Moving Average",
                                xaxis_title="Date",
                                yaxis_title="Price")
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Speed Resistance Lines":
            st.success("This program allows you to visualize Speed Resistance Lines for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = f"Ticker captured: {ticker}"
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                dataset["Middle_Line"] = dataset["Low"] + (dataset["High"] - dataset["Low"]) * 0.667
                dataset["Lower_Line"] = dataset["Low"] + (dataset["High"] - dataset["Low"]) * 0.333

                # Plot Speed Resistance Lines
                fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                                    open=dataset['Open'],
                                                    high=dataset['High'],
                                                    low=dataset['Low'],
                                                    close=dataset['Close'],
                                                    name='Candlesticks'),
                                    go.Scatter(x=dataset.index, y=dataset["Middle_Line"], mode='lines', name='Middle Line', line=dict(color='red')),
                                    go.Scatter(x=dataset.index, y=dataset["Lower_Line"], mode='lines', name='Lower Line', line=dict(color='green'))])
                fig.update_layout(title=f"{symbol} Speed Resistance Lines",
                                xaxis_title="Date",
                                yaxis_title="Price")
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Standard Deviation Volatility":
            st.success("This program allows you to visualize Standard Deviation Volatility for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = f"Ticker captured: {ticker}"
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                dataset["STD"] = dataset["Adj Close"].rolling(10).std()

                # Plot Standard Deviation Volatility
                fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                                    open=dataset['Open'],
                                                    high=dataset['High'],
                                                    low=dataset['Low'],
                                                    close=dataset['Close'],
                                                    name='Candlesticks'),
                                    go.Scatter(x=dataset.index, y=dataset["STD"], mode='lines', name='Standard Deviation Volatility', line=dict(color='red'))])
                fig.update_layout(title=f"{symbol} Standard Deviation Volatility",
                                xaxis_title="Date",
                                yaxis_title="Price")
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Stochastic RSI":
            st.success("This program allows you to visualize Stochastic RSI for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = f"Ticker captured: {ticker}"
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                # Calculate RSI
                n = 14
                change = dataset["Adj Close"].diff(1)
                gain = change.mask(change < 0, 0)
                loss = abs(change.mask(change > 0, 0))
                avg_gain = gain.rolling(n).mean()
                avg_loss = loss.rolling(n).mean()
                RS = avg_gain / avg_loss
                RSI = 100 - (100 / (1 + RS))

                # Calculate Stochastic RSI
                LL_RSI = RSI.rolling(14).min()
                HH_RSI = RSI.rolling(14).max()
                dataset["Stoch_RSI"] = (RSI - LL_RSI) / (HH_RSI - LL_RSI)

                # Plot Stochastic RSI
                fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                                    open=dataset['Open'],
                                                    high=dataset['High'],
                                                    low=dataset['Low'],
                                                    close=dataset['Close'],
                                                    name='Candlesticks'),
                                    go.Scatter(x=dataset.index, y=dataset["Stoch_RSI"], mode='lines', name='Stochastic RSI', line=dict(color='red'))])
                fig.update_layout(title=f"{symbol} Stochastic RSI",
                                xaxis_title="Date",
                                yaxis_title="Stochastic RSI")
                st.plotly_chart(fig)
        if pred_option_Technical_Indicators == "Stochastic Fast":
            st.success("This program allows you to visualize Stochastic Fast for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = f"Ticker captured: {ticker}"
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                # Calculate Stochastic Fast
                low_min = dataset['Low'].rolling(window=14).min()
                high_max = dataset['High'].rolling(window=14).max()
                dataset['%K'] = 100 * (dataset['Close'] - low_min) / (high_max - low_min)
                dataset['%D'] = dataset['%K'].rolling(window=3).mean()

                # Plot Stochastic Fast
                fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                                    open=dataset['Open'],
                                                    high=dataset['High'],
                                                    low=dataset['Low'],
                                                    close=dataset['Close'],
                                                    name='Candlesticks'),
                                    go.Scatter(x=dataset.index, y=dataset["%K"], mode='lines', name='Stochastic Fast %K', line=dict(color='red')),
                                    go.Scatter(x=dataset.index, y=dataset["%D"], mode='lines', name='Stochastic Fast %D', line=dict(color='blue'))])
                fig.update_layout(title=f"{symbol} Stochastic Fast",
                                xaxis_title="Date",
                                yaxis_title="Stochastic Fast")
                st.plotly_chart(fig)
        if pred_option_Technical_Indicators == "Stochastic Full":
            st.success("This program allows you to visualize Stochastic Full for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = f"Ticker captured: {ticker}"
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                # Calculate Stochastic Full
                low_min = dataset['Low'].rolling(window=14).min()
                high_max = dataset['High'].rolling(window=14).max()
                dataset['%K'] = 100 * (dataset['Close'] - low_min) / (high_max - low_min)
                dataset['%D'] = dataset['%K'].rolling(window=3).mean()
                dataset['%D_full'] = dataset['%D'].rolling(window=3).mean()

                # Plot Stochastic Full
                fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                                    open=dataset['Open'],
                                                    high=dataset['High'],
                                                    low=dataset['Low'],
                                                    close=dataset['Close'],
                                                    name='Candlesticks'),
                                    go.Scatter(x=dataset.index, y=dataset["%K"], mode='lines', name='Stochastic Full %K', line=dict(color='red')),
                                    go.Scatter(x=dataset.index, y=dataset["%D_full"], mode='lines', name='Stochastic Full %D', line=dict(color='blue'))])
                fig.update_layout(title=f"{symbol} Stochastic Full",
                                xaxis_title="Date",
                                yaxis_title="Stochastic Full")
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "Stochastic Slow":
            st.success("This program allows you to visualize Stochastic Slow for a selected ticker")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = f"Ticker captured: {ticker}"
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date

                # Read data
                dataset = yf.download(symbol, start, end)

                # Calculate Stochastic Slow
                low_min = dataset['Low'].rolling(window=14).min()
                high_max = dataset['High'].rolling(window=14).max()
                dataset['%K'] = 100 * (dataset['Close'] - low_min) / (high_max - low_min)
                dataset['%D'] = dataset['%K'].rolling(window=3).mean()

                # Plot Stochastic Slow
                fig = go.Figure(data=[go.Candlestick(x=dataset.index,
                                                    open=dataset['Open'],
                                                    high=dataset['High'],
                                                    low=dataset['Low'],
                                                    close=dataset['Close'],
                                                    name='Candlesticks'),
                                    go.Scatter(x=dataset.index, y=dataset["%K"], mode='lines', name='Stochastic Slow %K', line=dict(color='red')),
                                    go.Scatter(x=dataset.index, y=dataset["%D"], mode='lines', name='Stochastic Slow %D', line=dict(color='blue'))])
                fig.update_layout(title=f"{symbol} Stochastic Slow",
                                xaxis_title="Date",
                                yaxis_title="Stochastic Slow")
                st.plotly_chart(fig)

        
        
        if pred_option_Technical_Indicators == "Super Trend":
            st.success("This program allows you to view the Super Trend of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                n = 7  # Number of periods
                df["H-L"] = abs(df["High"] - df["Low"])
                df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
                df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
                df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
                df["ATR"] = df["TR"].rolling(n).mean()

                df["Upper Basic"] = (df["High"] + df["Low"]) / 2 + (2 * df["ATR"])
                df["Lower Basic"] = (df["High"] + df["Low"]) / 2 - (2 * df["ATR"])

                df["Upper Band"] = df["Upper Basic"]
                df["Lower Band"] = df["Lower Basic"]

                for i in range(n, len(df)):
                    if df["Close"][i - 1] <= df["Upper Band"][i - 1]:
                        df["Upper Band"][i] = min(df["Upper Basic"][i], df["Upper Band"][i - 1])
                    else:
                        df["Upper Band"][i] = df["Upper Basic"][i]

                for i in range(n, len(df)):
                    if df["Close"][i - 1] >= df["Lower Band"][i - 1]:
                        df["Lower Band"][i] = max(df["Lower Basic"][i], df["Lower Band"][i - 1])
                    else:
                        df["Lower Band"][i] = df["Lower Basic"][i]

                df["SuperTrend"] = 0.00
                for i in range(n, len(df)):
                    if df["Close"][i] <= df["Upper Band"][i]:
                        df["SuperTrend"][i] = df["Upper Band"][i]
                    elif df["Close"][i] > df["Upper Band"][i]:
                        df["SuperTrend"][i] = df["Lower Band"][i]

                # Candlestick Chart with Super Trend
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'], name='Candlestick'))

                fig.update_layout(title="Stock " + symbol + " Candlestick Chart with Super Trend",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))

                fig.add_trace(go.Scatter(x=df.index, y=df["SuperTrend"], mode='lines', name='SuperTrend'))
                st.plotly_chart(fig)

        if pred_option_Technical_Indicators == "True Strength Index":
            st.success("This program allows you to view the True Strength Index (TSI) of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Calculate True Strength Index (TSI)
                df["PC"] = df["Adj Close"] - df["Adj Close"].shift()
                df["EMA_FS"] = df["PC"].ewm(span=25, min_periods=25, adjust=True).mean()
                df["EMA_SS"] = df["EMA_FS"].ewm(span=13, min_periods=13, adjust=True).mean()
                df["Absolute_PC"] = abs(df["Adj Close"] - df["Adj Close"].shift())
                df["Absolute_FS"] = df["Absolute_PC"].ewm(span=25, min_periods=25).mean()
                df["Absolute_SS"] = df["Absolute_FS"].ewm(span=13, min_periods=13).mean()
                df["TSI"] = 100 * df["EMA_SS"] / df["Absolute_SS"]
                df = df.drop(["PC", "EMA_FS", "EMA_SS", "Absolute_PC", "Absolute_FS", "Absolute_SS"], axis=1)

                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
                fig.add_trace(go.Scatter(x=df.index, y=df["TSI"], mode='lines', name='True Strength Index'))
                fig.update_layout(title="Stock " + symbol + " True Strength Index",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick Chart with True Strength Index
                fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'], name='Candlestick')])
                fig_candle.add_trace(go.Scatter(x=df.index, y=df["TSI"], mode='lines', name='True Strength Index'))
                fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with True Strength Index",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig_candle)
    
        if pred_option_Technical_Indicators == "Ultimate Oscillator":
            st.success("This program allows you to view the Ultimate Oscillator of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Ultimate Oscillator
                df["Prior Close"] = df["Adj Close"].shift()
                df["BP"] = df["Adj Close"] - df[["Low", "Prior Close"]].min(axis=1)
                df["TR"] = df[["High", "Prior Close"]].max(axis=1) - df[["Low", "Prior Close"]].min(axis=1)
                df["Average7"] = df["BP"].rolling(7).sum() / df["TR"].rolling(7).sum()
                df["Average14"] = df["BP"].rolling(14).sum() / df["TR"].rolling(14).sum()
                df["Average28"] = df["BP"].rolling(28).sum() / df["TR"].rolling(28).sum()
                df["UO"] = 100 * (4 * df["Average7"] + 2 * df["Average14"] + df["Average28"]) / (4 + 2 + 1)
                df = df.drop(["Prior Close", "BP", "TR", "Average7", "Average14", "Average28"], axis=1)

                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
                fig.add_trace(go.Scatter(x=df.index, y=df["UO"], mode='lines', name='Ultimate Oscillator'))
                fig.update_layout(title="Stock " + symbol + " Ultimate Oscillator",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick Chart with Ultimate Oscillator
                fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'], name='Candlestick')])
                fig_candle.add_trace(go.Scatter(x=df.index, y=df["UO"], mode='lines', name='Ultimate Oscillator'))
                fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Ultimate Oscillator",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig_candle)

        if pred_option_Technical_Indicators == "Variance Indicator":
            st.success("This program allows you to view the Variance Indicator of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Variance Indicator
                n = 14
                df["Variance"] = df["Adj Close"].rolling(n).var()

                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
                fig.add_trace(go.Scatter(x=df.index, y=df["Variance"], mode='lines', name='Variance Indicator'))
                fig.update_layout(title="Stock " + symbol + " Variance Indicator",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick Chart with Variance Indicator
                fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'], name='Candlestick')])
                fig_candle.add_trace(go.Scatter(x=df.index, y=df["Variance"], mode='lines', name='Variance Indicator'))
                fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Variance Indicator",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig_candle)

        if pred_option_Technical_Indicators == "Volume Price Confirmation Indicator":
            st.success("This program allows you to view the Volume Price Confirmation Indicator of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Volume Price Confirmation Indicator
                short_term = 5
                long_term = 20
                vwma_lt = ((df["Adj Close"] * df["Volume"]) + (df["Adj Close"].shift(1) * df["Volume"].shift(1)) + (df["Adj Close"].shift(2) * df["Volume"].shift(2))) / (df["Volume"].rolling(long_term).sum())
                vwma_st = ((df["Adj Close"] * df["Volume"]) + (df["Adj Close"].shift(1) * df["Volume"].shift(1)) + (df["Adj Close"].shift(2) * df["Volume"].shift(2))) / (df["Volume"].rolling(short_term).sum())
                vpc = vwma_lt - df["Adj Close"].rolling(long_term).mean()
                vpr = vwma_st / df["Adj Close"].rolling(short_term).mean()
                vm = df["Adj Close"].rolling(short_term).mean() / df["Adj Close"].rolling(long_term).mean()
                vpci = vpc * vpr * vm

                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Closing Price'))
                fig.add_trace(go.Scatter(x=df.index, y=vpci, mode='lines', name='Volume Price Confirmation Indicator'))
                fig.update_layout(title="Stock " + symbol + " Volume Price Confirmation Indicator",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick Chart with Volume Price Confirmation Indicator
                fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'], name='Candlestick')])
                fig_candle.add_trace(go.Scatter(x=df.index, y=vpci, mode='lines', name='Volume Price Confirmation Indicator'))
                fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Volume Price Confirmation Indicator",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig_candle)

        if pred_option_Technical_Indicators == "Volume Weighted Moving Average (VWMA)":
            st.success("This program allows you to view the Volume Weighted Moving Average (VWMA) of a ticker over time")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button("Check"):    
                symbol = ticker
                start = start_date
                end = end_date
                # Read data
                df = yf.download(symbol, start, end)

                # Calculate Volume Weighted Moving Average (VWMA)
                df["Volume_x_Close"] = df["Volume"] * df["Close"]
                df["VWMA"] = df["Volume_x_Close"].rolling(window=14).sum() / df["Volume"].rolling(window=14).sum()

                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Closing Price'))
                fig.add_trace(go.Scatter(x=df.index, y=df["VWMA"], mode='lines', name='Volume Weighted Moving Average (VWMA)'))
                fig.update_layout(title="Stock " + symbol + " Volume Weighted Moving Average (VWMA)",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig)

                # Candlestick Chart with Volume Weighted Moving Average (VWMA)
                fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'], name='Candlestick')])
                fig_candle.add_trace(go.Scatter(x=df.index, y=df["VWMA"], mode='lines', name='Volume Weighted Moving Average (VWMA)'))
                fig_candle.update_layout(title="Stock " + symbol + " Candlestick Chart with Volume Weighted Moving Average (VWMA)",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"))
                st.plotly_chart(fig_candle)
    elif option =='Portfolio Strategies':
        pred_option_portfolio_strategies = st.selectbox('Make a choice', [
                                                                'Astral Timing signals',
                                                                'Lumibot Backtesting strategy',
                                                                'Backtest Strategies',
                                                                'Backtrader Backtest',
                                                                'Best Moving Averages Analysis',
                                                                'EMA Crossover Strategy',
                                                                'Factor Analysis',
                                                                'Financial Signal Analysis',
                                                                'Geometric Brownian Motion',
                                                                'Long Hold Stats Analysis',
                                                                'LS DCA Analysis',
                                                                'Monte Carlo',
                                                                'Moving Average Crossover Signals',
                                                                'Moving Average Strategy',
                                                                'Optimal Portfolio',
                                                                'Optimized Bollinger Bands',
                                                                'Pairs Trading',
                                                                'Portfolio Analysis',
                                                                'Portfolio Optimization',
                                                                'Portfolio VAR Simulation',
                                                                'Risk Management'
                                                                'RSI Trendline Strategy',
                                                                'RWB Strategy',
                                                                'SMA Trading Strategy',
                                                                'Stock Spread Plotter',
                                                                'Support Resistance Finder'])

        if pred_option_portfolio_strategies == "Astral Timing signals":
            st.success("This program allows you to start backtesting using Astral Timing signals")
            ticker = st.text_input("Enter the ticker you want to monitor")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")

            if st.button("Check"):
                # Apply Astral Timing signals to stock data.
                def astral(data, completion, step, step_two, what, high, low, where_long, where_short):
                    data['long_signal'] = 0
                    data['short_signal'] = 0

                    # Iterate through the DataFrame
                    for i in range(len(data)):
                        # Long signal logic
                        if data.iloc[i][what] < data.iloc[i - step][what] and data.iloc[i][low] < data.iloc[i - step_two][low]:
                            data.at[data.index[i], 'long_signal'] = -1
                        elif data.iloc[i][what] >= data.iloc[i - step][what]:
                            data.at[data.index[i], 'long_signal'] = 0

                        # Short signal logic
                        if data.iloc[i][what] > data.iloc[i - step][what] and data.iloc[i][high] > data.iloc[i - step_two][high]:
                            data.at[data.index[i], 'short_signal'] = 1
                        elif data.iloc[i][what] <= data.iloc[i - step][what]:
                            data.at[data.index[i], 'short_signal'] = 0

                    return data

                # Define stock ticker and date range
            
                start = start_date
                end = end_date

                # Fetch stock data
                data = pdr.get_data_yahoo(ticker, start, end)

                # Apply Astral Timing signals
                astral_data = astral(data, 8, 1, 5, 'Close', 'High', 'Low', 'long_signal', 'short_signal')

                # Display the results
                # Create candlestick chart with signals
                fig = go.Figure(data=[go.Candlestick(x=astral_data.index,
                                                    open=astral_data['Open'],
                                                    high=astral_data['High'],
                                                    low=astral_data['Low'],
                                                    close=astral_data['Close'])])

                # Add long and short signals to the plot
                fig.add_trace(go.Scatter(x=astral_data.index, y=astral_data['long_signal'],
                                        mode='markers', marker=dict(color='blue'), name='Long Signal'))
                fig.add_trace(go.Scatter(x=astral_data.index, y=astral_data['short_signal'],
                                        mode='markers', marker=dict(color='red'), name='Short Signal'))

                # Customize layout
                fig.update_layout(title=f"{ticker} Candlestick Chart with Signals",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                xaxis_rangeslider_visible=False)

                # Display the interactive plot
                st.plotly_chart(fig)
                st.write(astral_data[['long_signal', 'short_signal']])

        if pred_option_portfolio_strategies == "Backtest Strategies":
            st.success("This portion allows you backtest a ticker for a period using SMA Logic ")
            ticker = st.text_input("Enter the ticker for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):

                # Function to fetch stock data
                def get_stock_data(stock, start, end):
                    """
                    Fetches stock data from Yahoo Finance.
                    """
                    return pdr.get_data_yahoo(stock, start, end)

                # Trading statistics calculation
                def calculate_trading_statistics(df, buy_sell_logic, additional_logic=None):
                        """
                        Calculates trading statistics based on buy/sell logic.
                        """
                        position = 0
                        percentChange = []
                        buyP = sellP = 0  # Initialize buyP and sellP
                        for i in df.index:
                            close = df.loc[i, "Adj Close"]
                            if buy_sell_logic(df, i, position):
                                position = 0 if position == 1 else 1
                                buyP = close if position == 1 else buyP
                                sellP = close if position == 0 else sellP
                                if position == 0:
                                    perc = (sellP / buyP - 1) * 100
                                    percentChange.append(perc)
                            if additional_logic: additional_logic(df, i)
                        return calculate_statistics_from_percent_change(percentChange)

                # Compute statistics from percent change
                def calculate_statistics_from_percent_change(percentChange):
                    """
                    Computes statistics from percentage change in stock prices.
                    """
                    gains = sum(p for p in percentChange if p > 0)
                    losses = sum(p for p in percentChange if p < 0)
                    numGains = sum(1 for p in percentChange if p > 0)
                    numLosses = sum(1 for p in percentChange if p < 0)
                    totReturn = round(np.prod([((p / 100) + 1) for p in percentChange]) * 100 - 100, 2)
                    avgGain = gains / numGains if numGains > 0 else 0
                    avgLoss = losses / numLosses if numLosses > 0 else 0
                    maxReturn = max(percentChange) if numGains > 0 else 0
                    maxLoss = min(percentChange) if numLosses > 0 else 0
                    ratioRR = -avgGain / avgLoss if numLosses > 0 else "inf"
                    batting_avg = numGains / (numGains + numLosses) if numGains + numLosses > 0 else 0
                    return {
                        "total_return": totReturn,
                        "avg_gain": avgGain,
                        "avg_loss": avgLoss,
                        "max_return": maxReturn,
                        "max_loss": maxLoss,
                        "gain_loss_ratio": ratioRR,
                        "num_trades": numGains + numLosses,
                        "batting_avg": batting_avg
                    }

                # SMA strategy logic
                def sma_strategy_logic(df, i, position):
                    """
                    Logic for Simple Moving Average (SMA) trading strategy.
                    """
                    SMA_short, SMA_long = df["SMA_20"], df["SMA_50"]
                    return (SMA_short[i] > SMA_long[i] and position == 0) or (SMA_short[i] < SMA_long[i] and position == 1)

            
                stock = ticker
                num_of_years = years
                start = start_date
                end = end_date
                df = get_stock_data(stock, start, end)

                # Implementing SMA strategy
                df["SMA_20"] = df["Adj Close"].rolling(window=20).mean()
                df["SMA_50"] = df["Adj Close"].rolling(window=50).mean()
                sma_stats = calculate_trading_statistics(df, sma_strategy_logic)
                st.write("Simple Moving Average Strategy Stats:", sma_stats)

        if pred_option_portfolio_strategies == "Lumibot Backtesting strategy":
            st.success("This portion allows you backtest a ticker for a period using Lumibot YahooBacktesting strategy ")
            ticker = st.text_input("Enter the ticker for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            quantities = st.number_input("Enter the quantities of the ticker to buy")
            if quantities:
                st.write(f"The noted quantities are : {quantities}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                start = start_date
                end = end_date
                class MyStrategy(Strategy):
                    def on_trading_iteration(self):
                        if self.first_iteration:
                            order = self.create_order(ticker, quantity=quantities, side="buy")
                            self.submit_order(order)

                # Create a backtest
                backtesting_start = start
                backtesting_end = end

                # The benchmark asset to use for the backtest to compare to
                benchmark_asset = Asset(symbol="QQQ", asset_type="stock")

                backtest = MyStrategy.backtest(
                    datasource_class=YahooDataBacktesting,
                    backtesting_start=backtesting_start,
                    backtesting_end=backtesting_end,
                    benchmark_asset=benchmark_asset,
                )


            
        if pred_option_portfolio_strategies == "Backtrader Backtest":
            ticker = st.text_input("Enter the ticker for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):

                # Setting chart resolution
                plt.rcParams['figure.dpi'] = 140

                # API credentials
                API_KEY = API_KEY_ALPACA
                SECRET_KEY = SECRET_KEY_ALPACA

                # Initialize REST API
                rest_api = REST(API_KEY, SECRET_KEY, 'https://paper-api.alpaca.markets/v2')

                def run_backtest(strategy, symbol, start, end, timeframe=TimeFrame.Day, cash=10000):
                    cerebro = bt.Cerebro(stdstats=True)
                    cerebro.broker.setcash(cash)
                    cerebro.addstrategy(strategy)

                    # Adding analyzers
                    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
                    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')

                    # Loading data
                    data = rest_api.get_bars(symbol, timeframe, start, end, adjustment='all').df
                    bt_data = bt.feeds.PandasData(dataname=data, name=symbol)
                    cerebro.adddata(bt_data)

                    # Running the backtest
                    initial_value = cerebro.broker.getvalue()
                    st.write(f'Starting Portfolio Value: {initial_value}')
                    results = cerebro.run()
                    final_value = cerebro.broker.getvalue()
                    st.write(f'Final Portfolio Value: {round(final_value, 2)}')

                    strategy_return = 100 * (final_value - initial_value)/initial_value
                    st.write(f'Strategy Return: {round(strategy_return, 2)}%')

                    # Display results
                    strategy_statistics(results, initial_value, data)

                    # Plotting the results
                    cerebro.plot(iplot=False)
                
                def strategy_statistics(results, initial_value, data):
                    # Analyzing the results
                    strat = results[0]
                    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()

                    # Sharpe Ratio
                    sharpe_ratio = round(strat.analyzers.sharpe_ratio.get_analysis()['sharperatio'], 2)

                    # Total number of trades
                    total_trades = trade_analysis.total.closed

                    # Win Rate
                    win_rate = round((trade_analysis.won.total / total_trades) * 100, 2) if total_trades > 0 else 0

                    # Average Percent Gain and Loss
                    avg_percent_gain = round(trade_analysis.won.pnl.average / initial_value * 100, 2) if trade_analysis.won.total > 0 else 0
                    avg_percent_loss = round(trade_analysis.lost.pnl.average / initial_value * 100, 2) if trade_analysis.lost.total > 0 else 0

                    # Profit Factor
                    profit_factor = round((avg_percent_gain * win_rate) / (avg_percent_loss * (1 - win_rate)), 2) if avg_percent_loss != 0 else float('inf')

                    # Gain/Loss Ratio
                    gain_loss_ratio = round(avg_percent_gain / -avg_percent_loss, 2) if avg_percent_loss != 0 else float('inf')

                    # Max Return and Max Loss as Percentages
                    max_return = round(trade_analysis.won.pnl.max / initial_value * 100, 2) if trade_analysis.won.total > 0 else 0
                    max_loss = round(trade_analysis.lost.pnl.max / initial_value * 100, 2) if trade_analysis.lost.total > 0 else 0

                    # Buy and Hold Return
                    buy_and_hold_return = round((data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100, 2)

                    # Displaying results
                    st.write(f'Buy and Hold Return: {buy_and_hold_return}%')
                    st.write('Sharpe Ratio:', sharpe_ratio)
                    st.write('Total Trades:', total_trades)
                    st.write('Win Rate (%):', win_rate)
                    st.write('Average % Gain per Trade:', avg_percent_gain)
                    st.write('Average % Loss per Trade:', avg_percent_loss)
                    st.write('Profit Factor:', profit_factor)
                    st.write('Gain/Loss Ratio:', gain_loss_ratio)
                    st.write('Max % Return on a Trade:', max_return)
                    st.write('Max % Loss on a Trade:', max_loss)

                # Class for SMA Crossover strategy
                class SmaCross(bt.Strategy):
                    params = dict(pfast=13, pslow=25)

                    # Define trading strategy
                    def __init__(self):
                        sma1 = bt.ind.SMA(period=self.p.pfast)
                        sma2 = bt.ind.SMA(period=self.p.pslow)
                        self.crossover = bt.ind.CrossOver(sma1, sma2)

                        # Custom trade tracking
                        self.trade_data = []

                    # Execute trades
                    def next(self):
                        # Trading the entire portfolio
                        size = int(self.broker.get_cash() / self.data.close[0])

                        if not self.position:
                            if self.crossover > 0:
                                self.buy(size=size)
                                self.entry_bar = len(self)  # Record entry bar index
                        elif self.crossover < 0:
                            self.close()

                    # Record trade details
                    def notify_trade(self, trade):
                        if trade.isclosed:
                            exit_bar = len(self)
                            holding_period = exit_bar - self.entry_bar
                            trade_record = {
                                'entry': self.entry_bar,
                                'exit': exit_bar,
                                'duration': holding_period,
                                'profit': trade.pnl
                            }
                            self.trade_data.append(trade_record)

                    # Caclulating holding periods
                    def stop(self):
                        # Calculate and st.write average holding periods
                        total_holding = sum([trade['duration'] for trade in self.trade_data])
                        total_trades = len(self.trade_data)
                        avg_holding_period = round(total_holding / total_trades) if total_trades > 0 else 0

                        # Calculating for winners and losers separately
                        winners = [trade for trade in self.trade_data if trade['profit'] > 0]
                        losers = [trade for trade in self.trade_data if trade['profit'] < 0]
                        avg_winner_holding = round(sum(trade['duration'] for trade in winners) / len(winners))if winners else 0
                        avg_loser_holding = round(sum(trade['duration'] for trade in losers) / len(losers)) if losers else 0

                        # Display average holding period statistics
                        st.write('Average Holding Period:', avg_holding_period)
                        st.write('Average Winner Holding Period:', avg_winner_holding)
                        st.write('Average Loser Holding Period:', avg_loser_holding)

                    # Run backtest
                run_backtest(SmaCross, ticker, start_date, end_date, TimeFrame.Day, portfolio)
        
        if pred_option_portfolio_strategies == "Best Moving Averages Analysis":
            st.success("This portion allows you backtest a ticker for a period using the best moving averages Logic ")
            ticker = st.text_input("Enter the ticker for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                # Define stock symbol and historical data range
                symbol = ticker
                num_of_years = years
                start_date = start_date
                end_date = end_date

                # Fetch stock data using yfinance
                data = yf.download(symbol, start=start_date, end=end_date)

                # Calculate Simple Moving Averages (SMAs) for different periods
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['SMA_200'] = data['Close'].rolling(window=200).mean()

                # Create interactive plot for stock price and its SMAs
                fig_stock = go.Figure()
                fig_stock.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                fig_stock.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='20-period SMA'))
                fig_stock.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-period SMA'))
                fig_stock.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='200-period SMA'))

                fig_stock.update_layout(title=f'{symbol} Stock Price and SMAs',
                                        xaxis_title='Date',
                                        yaxis_title='Price')

                # Display the interactive plot
                st.plotly_chart(fig_stock)

                # Analysis to find the best SMA for predicting future returns
                days_forward = 10
                results = []

                # Testing different SMA lengths
                for sma_length in range(20, 500):
                    data['SMA'] = data['Close'].rolling(sma_length).mean()
                    data['Position'] = data['Close'] > data['SMA']
                    data['Forward Close'] = data['Close'].shift(-days_forward)
                    data['Forward Return'] = (data['Forward Close'] - data['Close']) / data['Close']
                    
                    # Splitting into training and test datasets
                    train_data = data[:int(0.6 * len(data))]
                    test_data = data[int(0.6 * len(data)):]
                    
                    # Calculating average forward returns
                    train_return = train_data[train_data['Position']]['Forward Return'].mean()
                    test_return = test_data[test_data['Position']]['Forward Return'].mean()
                    
                    # Statistical test
                    p_value = ttest_ind(train_data[train_data['Position']]['Forward Return'],
                                        test_data[test_data['Position']]['Forward Return'],
                                        equal_var=False)[1]
                    
                    results.append({'SMA Length': sma_length, 
                                    'Train Return': train_return, 
                                    'Test Return': test_return, 
                                    'p-value': p_value})

                # Sorting results and printing the best SMA
                best_result = sorted(results, key=lambda x: x['Train Return'], reverse=True)[0]
                st.write(f"Best SMA Length: {best_result['SMA Length']}")
                st.write(f"Train Return: {best_result['Train Return']:.4f}")
                st.write(f"Test Return: {best_result['Test Return']:.4f}")
                st.write(f"p-value: {best_result['p-value']:.4f}")

                # Create interactive plot for the best SMA
                data['Best SMA'] = data['Close'].rolling(best_result['SMA Length']).mean()
                fig_best_sma = go.Figure()
                fig_best_sma.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                fig_best_sma.add_trace(go.Scatter(x=data.index, y=data['Best SMA'], mode='lines', name=f"{best_result['SMA Length']} periods SMA"))

                fig_best_sma.update_layout(title=f'{symbol} Stock Price and Best SMA',
                                            xaxis_title='Date',
                                            yaxis_title='Price')

                # Display the interactive plot for the best SMA
                st.plotly_chart(fig_best_sma)

        if pred_option_portfolio_strategies == "EMA Crossover Strategy":
            tickers = []
            ticker = st.text_input("Enter the ticker for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
                tickers.append(ticker)
            more_input = st.selectbox("Please Add one/more ticker(s) for comparison", ("","Yes", "No"))
            if more_input == "Yes":
                ticker_2 =  st.text_input("Enter another ticker to continue the investigation")
                tickers.append(ticker_2)
                portfolio = st.number_input("Enter the portfolio size in USD")
                if portfolio:
                    st.write(f"The portfolio size in USD Captured is : {portfolio}")
                min_date = datetime(1980, 1, 1)
                # Date input widget with custom minimum date
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:", min_value=min_date)
                with col2:
                    end_date = st.date_input("End Date:")
                years = end_date.year - start_date.year
                st.success(f"years captured : {years}")
                if st.button("Check"):
                    # Define the analysis period
                    num_of_years = years
                    start =  start_date
                    end = end_date

                    # Define tickers for analysis

                    # Fetch stock data using Yahoo Finance
                    df = pdr.get_data_yahoo(tickers, start, end)['Close']

                    # Calculate moving averages
                    short_rolling = df.rolling(window=20).mean()
                    long_rolling = df.rolling(window=100).mean()
                    ema_short = df.ewm(span=20, adjust=False).mean()

                    # Determine trading position based on EMA
                    trade_positions_raw = df - ema_short
                    trade_positions = trade_positions_raw.apply(np.sign) / 3  # Equal weighting
                    trade_positions_final = trade_positions.shift(1)  # Shift to simulate next-day trading

                    # Calculate asset and portfolio returns
                    asset_log_returns = np.log(df).diff()
                    portfolio_log_returns = trade_positions_final * asset_log_returns
                    cumulative_portfolio_log_returns = portfolio_log_returns.cumsum()
                    cumulative_portfolio_relative_returns = np.exp(cumulative_portfolio_log_returns) - 1

                    # Plot cumulative returns
                    cumulative_fig = go.Figure()
                    for ticker in asset_log_returns:
                        cumulative_fig.add_trace(go.Scatter(x=cumulative_portfolio_relative_returns.index,
                                                            y=100 * cumulative_portfolio_relative_returns[ticker],
                                                            mode='lines',
                                                            name=ticker))

                    cumulative_fig.update_layout(title='Cumulative Log Returns (%)',
                                                xaxis_title='Date',
                                                yaxis_title='Cumulative Log Returns (%)',
                                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(cumulative_fig)

                    # Comparing exact and approximate cumulative returns
                    cumulative_return_exact = cumulative_portfolio_relative_returns.sum(axis=1)
                    cumulative_log_return = cumulative_portfolio_log_returns.sum(axis=1)
                    cumulative_return_approx = np.exp(cumulative_log_return) - 1

                    # Plot exact vs approximate returns
                    approx_fig = go.Figure()
                    approx_fig.add_trace(go.Scatter(x=cumulative_return_exact.index,
                                                    y=100 * cumulative_return_exact,
                                                    mode='lines',
                                                    name='Exact'))
                    approx_fig.add_trace(go.Scatter(x=cumulative_return_approx.index,
                                                    y=100 * cumulative_return_approx,
                                                    mode='lines',
                                                    name='Approx'))

                    approx_fig.update_layout(title='Total Cumulative Relative Returns (%)',
                                            xaxis_title='Date',
                                            yaxis_title='Total Cumulative Relative Returns (%)',
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(approx_fig)

                    # Function to print portfolio statistics
                    def print_portfolio_statistics(portfolio_returns, num_of_years):
                        total_return = portfolio_returns[-1]
                        avg_yearly_return = (1 + total_return) ** (1 / num_of_years) - 1
                        st.write(f'Total Portfolio Return: {total_return * 100:.2f}%')
                        st.write(f'Average Yearly Return: {avg_yearly_return * 100:.2f}%')

                    # Printing statistics for EMA crossover strategy
                    print_portfolio_statistics(cumulative_return_exact, num_of_years)
            if more_input == "No":
                st.error("The EMA crossover cannot proceed without a comparison")

        if pred_option_portfolio_strategies == "Factor Analysis":
            tickers = []
            ticker = st.text_input("Enter the ticker for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
                tickers.append(ticker)
            more_input = st.selectbox("Please Add one/more ticker(s) for comparison", ("","Yes", "No"))
            if more_input == "Yes":
                ticker_2 =  st.text_input("Enter another ticker to continue the investigation")
                tickers.append(ticker_2)
                portfolio = st.number_input("Enter the portfolio size in USD")
                if portfolio:
                    st.write(f"The portfolio size in USD Captured is : {portfolio}")
                min_date = datetime(1980, 1, 1)
                # Date input widget with custom minimum date
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:", min_value=min_date)
                with col2:
                    end_date = st.date_input("End Date:")
                years = end_date.year - start_date.year
                st.success(f"years captured : {years}")
                if st.button("Check"):

                    # Setting plot aesthetics
                    sns.set(style='darkgrid', context='talk', palette='Dark2')

                    # Defining the time frame for data collection
                    end_date = dt.datetime.now()
                    start_date = end_date - dt.timedelta(days=365 * 7)

                    # List of stock symbols for factor analysis
                    symbols = tickers

                    # Fetching adjusted close prices for the specified symbols
                    df = pd.DataFrame({symbol: yf.download(symbol, start_date, end_date)['Adj Close']
                                    for symbol in symbols})

                    # Initializing FactorAnalyzer and fitting it to our data
                    fa = FactorAnalyzer(rotation=None, n_factors=df.shape[1])
                    fa.fit(df.dropna())

                    # Extracting communalities, eigenvalues, and factor loadings
                    communalities = fa.get_communalities()
                    eigenvalues, _ = fa.get_eigenvalues()
                    loadings = fa.loadings_

                    # Plotting the Scree plot to assess the number of factors
                    # Plotting the Scree plot to assess the number of factors
                    scree_fig = go.Figure()
                    scree_fig.add_trace(go.Scatter(x=list(range(1, df.shape[1] + 1)),
                                                y=eigenvalues,
                                                mode='markers+lines',
                                                name='Eigenvalues',
                                                marker=dict(color='blue')))
                    scree_fig.update_layout(title='Scree Plot',
                                            xaxis_title='Number of Factors',
                                            yaxis_title='Eigenvalue',
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(scree_fig)

                    # Bartlett's test of sphericity
                    chi_square_value, p_value = calculate_bartlett_sphericity(df.dropna())
                    st.write('Bartlett sphericity test:\nChi-square value:', chi_square_value, '\nP-value:', p_value)

                    # Kaiser-Meyer-Olkin (KMO) test
                    kmo_all, kmo_model = calculate_kmo(df.dropna())
                    st.write('Kaiser-Meyer-Olkin (KMO) Test:\nOverall KMO:', kmo_all, '\nKMO per variable:', kmo_model)

                    # Printing results
                    st.write("\nFactor Analysis Results:")
                    st.write("\nCommunalities:\n", communalities)
                    st.write("\nFactor Loadings:\n", loadings)
            if more_input == "No":
                st.error("The EMA crossover cannot proceed without a comparison")


        if pred_option_portfolio_strategies == "Financial Signal Analysis":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
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
                st.plotly_chart(fig1)

                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=stock_data, name=f'{ticker.upper()} Price Distribution'))
                fig2.update_layout(title=f'{ticker.upper()} Price Distribution', xaxis_title='Price', yaxis_title='Frequency')
                st.plotly_chart(fig2)

                # Calculating and plotting stock returns
                stock_returns = stock_data.apply(np.log).diff(1)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=stock_returns.index, y=stock_returns.values, mode='lines', name=f'{ticker.upper()} Returns'))
                fig3.update_layout(title=f'{ticker.upper()} Returns', xaxis_title='Date', yaxis_title='Returns')
                st.plotly_chart(fig3)

                fig4 = go.Figure()
                fig4.add_trace(go.Histogram(x=stock_returns, name=f'{ticker.upper()} Returns Distribution'))
                fig4.update_layout(title=f'{ticker.upper()} Returns Distribution', xaxis_title='Returns', yaxis_title='Frequency')
                st.plotly_chart(fig4)

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
                st.plotly_chart(fig5)

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
                st.plotly_chart(fig6)
        if pred_option_portfolio_strategies == "Geometric Brownian Motion":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                # Constants for analysis
                num_of_years = years  # Number of years for historical data
                start = start_date
                stock = ticker
                index = '^GSPC'

                # Fetching historical data from Yahoo Finance
                stock_data = pdr.get_data_yahoo(stock, start_date, end_date)
                index_data = pdr.get_data_yahoo(index, start_date, end_date)

                # Resampling data to monthly frequency and calculating returns
                stock_monthly = stock_data.resample('M').last()
                index_monthly = index_data.resample('M').last()
                combined_data = pd.DataFrame({'Stock': stock_monthly['Adj Close'], 
                                            'Index': index_monthly['Adj Close']})
                combined_returns = combined_data.pct_change().dropna()

                # Calculating covariance matrix for the returns
                cov_matrix = np.cov(combined_returns['Stock'], combined_returns['Index'])

                # Class for Geometric Brownian Motion simulation
                class GBM:
                    def __init__(self, initial_price, drift, volatility, time_period, total_time):
                        self.initial_price = initial_price
                        self.drift = drift
                        self.volatility = volatility
                        self.time_period = time_period
                        self.total_time = total_time
                        self.simulate()

                    def simulate(self):
                        self.prices = [self.initial_price]
                        while self.total_time > 0:
                            dS = self.prices[-1] * (self.drift * self.time_period + 
                                                    self.volatility * np.random.normal(0, math.sqrt(self.time_period)))
                            self.prices.append(self.prices[-1] + dS)
                            self.total_time -= self.time_period

                # Parameters for GBM simulation
                num_simulations = 20
                initial_price = stock_data['Adj Close'][-1]
                drift = 0.24
                volatility = math.sqrt(cov_matrix[0, 0])
                time_period = 1 / 365
                total_time = 1

                # Running multiple GBM simulations
                simulations = [GBM(initial_price, drift, volatility, time_period, total_time) for _ in range(num_simulations)]

                # Plotting the simulations
                fig = go.Figure()
                for i, sim in enumerate(simulations):
                    fig.add_trace(go.Scatter(x=np.arange(len(sim.prices)), y=sim.prices, mode='lines', name=f'Simulation {i+1}'))

                fig.add_trace(go.Scatter(x=np.arange(len(sim.prices)), y=[initial_price] * len(sim.prices),
                                        mode='lines', name='Initial Price', line=dict(color='red', dash='dash')))
                fig.update_layout(title=f'Geometric Brownian Motion for {stock.upper()}',
                                xaxis_title='Time Steps',
                                yaxis_title='Price')
                st.plotly_chart(fig)

        
        if pred_option_portfolio_strategies == "Long Hold Stats Analysis":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                # Constants for analysis         
                # Function to download stock data
                def download_stock_data(symbol, start, end):
                    return yf.download(symbol, start, end)['Adj Close']

                # Function to calculate investment statistics
                def calculate_investment_stats(df, investment_amount, symbol):
                    # Calculate number of shares bought and investment values
                    shares = int(investment_amount / df.iloc[0])
                    begin_value = round(shares * df.iloc[0], 2)
                    current_value = round(shares * df.iloc[-1], 2)

                    # Calculate daily returns and various statistics
                    returns = df.pct_change().dropna()
                    stats = {
                        'mean': round(returns.mean() * 100, 2),
                        'std_dev': round(returns.std() * 100, 2),
                        'skew': round(returns.skew(), 2),
                        'kurt': round(returns.kurtosis(), 2),
                        'total_return': round((1 + returns).cumprod().iloc[-1], 4) * 100
                    }
                    return shares, begin_value, current_value, stats

                # User inputs
                symbol = ticker
                num_of_years = years
                investment_amount = portfolio

                # Calculate date range
                start = dt.datetime.now() - dt.timedelta(days=int(365.25 * num_of_years))
                end = dt.datetime.now()

                # Download and process stock data
                df = download_stock_data(ticker, start_date, end_date)
                shares, begin_value, current_value, stats = calculate_investment_stats(df, investment_amount, symbol)

                # Print statistics
                st.write(f'\nNumber of Shares for {symbol}: {shares}')
                st.write(f'Beginning Value: ${begin_value}')
                st.write(f'Current Value: ${current_value}')
                st.write(f"\nStatistics:\nMean: {stats['mean']}%\nStd. Dev: {stats['std_dev']}%\nSkew: {stats['skew']}\nKurt: {stats['kurt']}\nTotal Return: {stats['total_return']}%")

                # Plotting returns and other statistics
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df.pct_change(), mode='lines', name='Daily Returns'))
                fig.update_layout(title=f'{symbol} Daily Returns', xaxis_title='Date', yaxis_title='Returns')
                st.plotly_chart(fig)

        if pred_option_portfolio_strategies == "LS DCA Analysis":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                def fetch_stock_data(symbol, start, end):
                    return yf.download(symbol, start, end)

                # Lump Sum Investment Function
                def lump_sum_investment(df, invest_date, principal):
                    invest_price = df.loc[invest_date]['Adj Close']
                    current_price = df['Adj Close'][-1]
                    return principal * ((current_price / invest_price) - 1)

                # Dollar-Cost Averaging Function
                def dca_investment(df, invest_date, periods, freq, principal):
                    dca_dates = pd.date_range(invest_date, periods=periods, freq=freq)
                    dca_dates = dca_dates[dca_dates < df.index[-1]]
                    cut_off_count = 12 - len(dca_dates)
                    cut_off_value = cut_off_count * (principal / periods)
                    dca_value = cut_off_value
                    for date in dca_dates:
                        trading_date = df.index[df.index.searchsorted(date)]
                        dca_value += lump_sum_investment(df, trading_date, principal / periods)
                    return dca_value

                # User Input
                symbol = ticker
                years = years
                principal = portfolio

                # Set dates for data retrieval
                start_date = start_date
                end_date = end_date

                # Fetch Data
                stock_data = fetch_stock_data(symbol, start_date, end_date)

                # Analysis for Lump Sum and DCA
                lump_sum_values = [lump_sum_investment(stock_data, date, principal) for date in stock_data.index]
                dca_values = [dca_investment(stock_data, date, 12, '30D', principal) for date in stock_data.index]
            
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name=f'{symbol} Price'))
                fig1.update_layout(title=f'{symbol} Stock Price', yaxis_title='Price', yaxis_tickprefix='$', yaxis_tickformat=',.0f')

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=stock_data.index, y=lump_sum_values, mode='lines', name='Lump Sum Investment'))
                fig2.add_trace(go.Scatter(x=stock_data.index, y=dca_values, mode='lines', name='DCA Investment'))
                fig2.update_layout(title='Lump Sum vs. DCA Investment Value', yaxis_title='Investment Value', yaxis_tickprefix='$', yaxis_tickformat=',.0f')

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=stock_data.index, y=np.array(lump_sum_values) - np.array(dca_values), mode='lines', name='Difference in Investment Values'))
                fig3.update_layout(title='Difference Between Lump Sum and DCA', yaxis_title='Difference', yaxis_tickprefix='$', yaxis_tickformat=',.0f')

                # Display plots in Streamlit
                st.plotly_chart(fig1)
                st.plotly_chart(fig2)
                st.plotly_chart(fig3)


        if pred_option_portfolio_strategies == "Monte Carlo":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                # Function to download stock data
                symbol = ticker
                df = yf.download(symbol, start_date, end_date)

                # Function to calculate annual volatility
                def annual_volatility(df):
                    quote = df['Close']
                    returns = quote.pct_change()
                    return returns.std() * np.sqrt(252)

                # Function to calculate CAGR
                def cagr(df):
                    quote = df['Close']
                    days = (quote.index[-1] - quote.index[0]).days
                    return ((((quote[-1]) / quote[1])) ** (365.0/days)) - 1

                # Monte Carlo Simulation Function
                def monte_carlo_simulation(simulations, days_predicted):
                    mu = cagr(df)
                    vol = annual_volatility(df)
                    start_price = df['Close'][-1]

                    results = []
                    
                    # Run simulations
                    for _ in range(simulations):
                        prices = [start_price]
                        for _ in range(days_predicted):
                            shock = np.random.normal(mu / days_predicted, vol / math.sqrt(days_predicted))
                            prices.append(prices[-1] * (1 + shock))
                        results.append(prices[-1])

                    return pd.DataFrame({
                        "Results": results,
                        "Percentile 5%": np.percentile(results, 5),
                        "Percentile 95%": np.percentile(results, 95)
                    })

            
                
                symbol = ticker
                start_date = start_date
                end_date = end_date
                simulations = 1000
                days_predicted = 252

                # Perform Monte Carlo Simulation
                simulation_results = monte_carlo_simulation(simulations, days_predicted)

                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=simulation_results['Results'], histnorm='probability'))
                fig.update_layout(title=f"{symbol} Monte Carlo Simulation Histogram", xaxis_title="Price", yaxis_title="Probability Density")
                st.plotly_chart(fig)

        if pred_option_portfolio_strategies == "Moving Average Crossover Signals":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                # Function to retrieve stock data
                def get_stock_data(ticker, start_date, end_date):
                    return yf.download(ticker, start_date, end_date)

                # Function to calculate Simple Moving Averages (SMA)
                def calculate_sma(data, window):
                    return data['Close'].rolling(window=window).mean()

                # Function to generate buy and sell signals
                def generate_signals(data):
                    signal_buy = []
                    signal_sell = []
                    flag = -1

                    for i in range(len(data)):
                        if data['SMA 50'][i] > data['SMA 200'][i] and flag != 1:
                            signal_buy.append(data['Close'][i])
                            signal_sell.append(np.nan)
                            flag = 1
                        elif data['SMA 50'][i] < data['SMA 200'][i] and flag != 0:
                            signal_buy.append(np.nan)
                            signal_sell.append(data['Close'][i])
                            flag = 0
                        else:
                            signal_buy.append(np.nan)
                            signal_sell.append(np.nan)

                    return signal_buy, signal_sell

                # Main function to run the analysis
                def main():
                    st.title("Stock Data Analysis with Moving Averages and Signals")

                    ticker = st.text_input("Enter a ticker:")
                    if not ticker:
                        st.warning("Please enter a valid ticker.")
                        return

                    num_of_years = 6
                    start_date = dt.datetime.now() - dt.timedelta(int(365.25 * num_of_years))
                    end_date = dt.datetime.now()

                    # Retrieve and process stock data
                    stock_data = get_stock_data(ticker, start_date, end_date)
                    stock_data['SMA 50'] = calculate_sma(stock_data, 50)
                    stock_data['SMA 200'] = calculate_sma(stock_data, 200)

                    # Generate buy and sell signals
                    buy_signals, sell_signals = generate_signals(stock_data)

                    # Plotting
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA 50'], mode='lines', name='SMA 50'))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA 200'], mode='lines', name='SMA 200'))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green')))
                    fig.add_trace(go.Scatter(x=stock_data.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red')))

                    fig.update_layout(title=f'{ticker.upper()} Close Price History with Buy & Sell Signals',
                                    xaxis_title='Date',
                                    yaxis_title='Close Price')

                    st.plotly_chart(fig)
        
        if pred_option_portfolio_strategies == "Moving Average Strategy":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
            
                # Function to download stock data
                def download_stock_data(ticker, start_date, end_date):
                    return yf.download(ticker, start_date, end_date)

                # Function to calculate moving averages
                def calculate_moving_averages(data, windows):
                    for window in windows:
                        data[f'SMA_{window}'] = data['Adj Close'].rolling(window).mean()
                    return data

                # Main function
                def main():
                    st.title("Stock Data Analysis with Moving Averages")

                    ticker = st.text_input("Enter a ticker:")
                    if not ticker:
                        st.warning("Please enter a valid ticker.")
                        return

                    num_of_years = 6
                    start_date = dt.datetime.now() - dt.timedelta(int(365.25 * num_of_years))
                    end_date = dt.datetime.now()

                    # Download and process stock data
                    stock_data = download_stock_data(ticker, start_date, end_date)
                    stock_data = calculate_moving_averages(stock_data, [20, 40, 80])

                    # Plotting
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name='Close Price'))
                    for window in [20, 40, 80]:
                        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[f'SMA_{window}'], mode='lines', name=f'{window}-days SMA'))
                    
                    fig.update_layout(title=f'{ticker.upper()} Close Price with Moving Averages',
                                    xaxis_title='Date',
                                    yaxis_title='Close Price')

                    st.plotly_chart(fig)

        if pred_option_portfolio_strategies == "Optimal Portfolio":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                # Function to download stock data and calculate annual returns
                def annual_returns(symbols, start_date, end_date):
                    df = yf.download(symbols, start_date, end_date)['Adj Close']
                    log_rets = np.log(df / df.shift(1))
                    return np.exp(log_rets.groupby(log_rets.index.year).sum()) - 1

                # Function to calculate portfolio variance
                def portfolio_var(returns, weights):
                    cov_matrix = np.cov(returns.T)
                    return np.dot(weights.T, np.dot(cov_matrix, weights))

                # Function to calculate Sharpe ratio
                def sharpe_ratio(returns, weights, rf_rate):
                    portfolio_return = np.dot(returns.mean(), weights)
                    portfolio_volatility = np.sqrt(portfolio_var(returns, weights))
                    return (portfolio_return - rf_rate) / portfolio_volatility

                # Function to optimize portfolio for maximum Sharpe ratio
                def optimize_portfolio(returns, initial_weights, rf_rate):
                    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bounds = tuple((0, 1) for _ in range(len(initial_weights)))
                    optimized = fmin(lambda x: -sharpe_ratio(returns, x, rf_rate), initial_weights, disp=False)
                    return optimized

                # Main function to execute the script
                def main():
                    st.title("Optimal Portfolio Optimization")

                    symbols = st.text_input("Enter stock symbols (comma-separated):")
                    start_date = st.date_input("Enter start date:")
                    end_date = st.date_input("Enter end date:")
                    rf_rate = st.number_input("Enter risk-free rate (in decimal):", min_value=0.0, value=0.003)

                    if not symbols or not start_date or not end_date:
                        st.warning("Please provide all required inputs.")
                        return

                    symbols = symbols.split(',')

                    # Calculate annual returns
                    returns = annual_returns(symbols, start_date, end_date)

                    # Initialize equal weights
                    initial_weights = np.ones(len(symbols)) / len(symbols)

                    # Calculate equal weighted portfolio Sharpe ratio
                    equal_weighted_sharpe = sharpe_ratio(returns, initial_weights, rf_rate)
                    
                    # Optimize portfolio
                    optimal_weights = optimize_portfolio(returns, initial_weights, rf_rate)
                    optimal_sharpe = sharpe_ratio(returns, optimal_weights, rf_rate)

                    # Display results
                    st.write(f"Equal Weighted Portfolio Sharpe Ratio: {equal_weighted_sharpe}")
                    st.write(f"Optimal Portfolio Weights: {optimal_weights}")
                    st.write(f"Optimal Portfolio Sharpe Ratio: {optimal_sharpe}")

    
        if pred_option_portfolio_strategies == "Optimized Bollinger Bands":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
            # Function to download stock data from Yahoo Finance
                def get_stock_data(ticker):
                    df = yf.download(ticker)
                    df = df[['Adj Close']]
                    return df

                # Function to add Bollinger Bands to DataFrame
                def add_bollinger_bands(df, window_size=20, num_std_dev=2):
                    df['SMA'] = df['Adj Close'].rolling(window=window_size).mean()
                    df['Upper Band'] = df['SMA'] + (df['Adj Close'].rolling(window=window_size).std() * num_std_dev)
                    df['Lower Band'] = df['SMA'] - (df['Adj Close'].rolling(window=window_size).std() * num_std_dev)
                    return df

                # Function to plot stock prices with Bollinger Bands
                def plot_with_bollinger_bands(df, ticker):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name=f'{ticker} Adjusted Close'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], mode='lines', name='20 Day SMA'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper Band'], mode='lines', name='Upper Bollinger Band'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower Band'], mode='lines', name='Lower Bollinger Band'))
                    fig.update_layout(title=f'{ticker} Stock Price with Bollinger Bands',
                                    xaxis_title='Date',
                                    yaxis_title='Price')
                    st.plotly_chart(fig)

                # Main function to execute the script
                st.title("Stock Price with Bollinger Bands")

                ticker = st.text_input("Enter stock ticker:")
                if not ticker:
                    st.warning("Please enter a valid ticker.")
                    return

                df = get_stock_data(ticker)
                df = add_bollinger_bands(df)

                plot_with_bollinger_bands(df, ticker)

        if pred_option_portfolio_strategies == "Pairs Trading":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
    
                # Function to download stock data from Yahoo Finance
                def download_stock_data(symbols, start_date, end_date):
                    """Download historical stock data for given symbols from Yahoo Finance."""
                    stock_data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
                    return stock_data.dropna()

                # Function to identify cointegrated pairs of stocks
                def find_cointegrated_pairs(data):
                    """Identify cointegrated pairs of stocks."""
                    n = data.shape[1]
                    score_matrix, pvalue_matrix = np.zeros((n, n)), np.ones((n, n))
                    pairs = []
                    for i in range(n):
                        for j in range(i+1, n):
                            S1, S2 = data[data.columns[i]], data[data.columns[j]]
                            _, pvalue, _ = coint(S1, S2)
                            score_matrix[i, j], pvalue_matrix[i, j] = _, pvalue
                            if pvalue < 0.05:  # Using a p-value threshold of 0.05
                                pairs.append((data.columns[i], data.columns[j]))
                    return score_matrix, pvalue_matrix, pairs

                # Function to plot heatmap of p-values for cointegration test using Plotly
                def plot_cointegration_heatmap(pvalues, tickers):
                    """Plot heatmap of p-values for cointegration test."""
                    fig = go.Figure(data=go.Heatmap(
                        z=pvalues,
                        x=tickers,
                        y=tickers,
                        colorscale='Viridis',
                        zmin=0,
                        zmax=0.05
                    ))
                    fig.update_layout(title="P-Values for Pairs Cointegration Test")
                    return fig

                # Pairs Trading Section
                def pairs_trading():
                    st.title("Pairs Trading")

                    # Inputs
                    ticker = st.text_input("Please enter the ticker needed for investigation")
                    portfolio = st.number_input("Enter the portfolio size in USD")
                    min_date = datetime.datetime(1980, 1, 1)
                    start_date = st.date_input("Start date:", min_value=min_date)
                    end_date = st.date_input("End Date:")
                    years = end_date.year - start_date.year

                    if st.button("Check"):
                        # Download and process data
                        data = download_stock_data([ticker], start_date, end_date)

                        # Find cointegrated pairs
                        _, pvalues, pairs = find_cointegrated_pairs(data)

                        # Plot heatmap of p-values
                        fig = plot_cointegration_heatmap(pvalues, [ticker])
                        st.plotly_chart(fig)

                        # Display the found pairs
                        st.write("Cointegrated Pairs:", pairs)
                
        if pred_option_portfolio_strategies == "Portfolio Analysis":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"): 
                # Override yfinance with Pandas Datareader's Yahoo Finance API
                yf.pdr_override()

                def get_historical_prices(symbols, start_date, end_date):
                    """Retrieve historical stock prices for specified symbols."""
                    return yf.download(symbols, start=start_date, end=end_date)['Adj Close']

                def calculate_daily_returns(prices):
                    """Calculate daily returns from stock prices."""
                    return np.log(prices / prices.shift(1))

                def calculate_monthly_returns(daily_returns):
                    """Calculate monthly returns from daily returns."""
                    return np.exp(daily_returns.groupby(lambda date: date.month).sum()) - 1

                def calculate_annual_returns(daily_returns):
                    """Calculate annual returns from daily returns."""
                    return np.exp(daily_returns.groupby(lambda date: date.year).sum()) - 1

                def portfolio_variance(returns, weights=None):
                    """Calculate the variance of a portfolio."""
                    if weights is None:
                        weights = np.ones(len(returns.columns)) / len(returns.columns)
                    covariance_matrix = np.cov(returns.T)
                    return np.dot(weights, np.dot(covariance_matrix, weights))

                def sharpe_ratio(returns, weights=None, risk_free_rate=0.001):
                    """Calculate the Sharpe ratio of a portfolio."""
                    if weights is None:
                        weights = np.ones(len(returns.columns)) / len(returns.columns)
                    port_var = portfolio_variance(returns, weights)
                    port_return = np.dot(returns.mean(), weights)
                    return (port_return - risk_free_rate) / np.sqrt(port_var)

                # Example usage
                symbols = ['AAPL', 'MSFT', 'GOOGL']
                start_date = dt.datetime.now() - dt.timedelta(days=365*5)
                end_date = dt.datetime.now()

                # Fetch historical data
                historical_prices = get_historical_prices(symbols, start_date, end_date)

                # Calculate returns
                daily_returns = calculate_daily_returns(historical_prices)
                monthly_returns = calculate_monthly_returns(daily_returns)
                annual_returns = calculate_annual_returns(daily_returns)

                # Calculate portfolio metrics
                portfolio_variance = portfolio_variance(annual_returns)
                portfolio_sharpe_ratio = sharpe_ratio(daily_returns)

                # Display results
                st.write(f"Portfolio Variance: {portfolio_variance}")
                st.write(f"Portfolio Sharpe Ratio: {portfolio_sharpe_ratio}")

                # Plot historical prices using Plotly
                fig = go.Figure()
                for symbol in symbols:
                    fig.add_trace(go.Scatter(x=historical_prices.index, y=historical_prices[symbol], mode='lines', name=symbol))

                fig.update_layout(title="Historical Prices",
                                xaxis_title="Date",
                                yaxis_title="Adjusted Closing Price",
                                legend_title="Symbols")
                st.plotly_chart(fig)


        if pred_option_portfolio_strategies == "Portfolio Optimization":
            tickers = []
            ticker = st.text_input("Enter the ticker for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
                tickers.append(ticker)
            more_input = st.selectbox("Please Add one/more ticker(s) for comparison", ("","Yes", "No"))
            if more_input == "Yes":
                ticker_2 =  st.text_input("Enter another ticker to continue the investigation")
                tickers.append(ticker_2)
                portfolio = st.number_input("Enter the portfolio size in USD")
                if portfolio:
                    st.write(f"The portfolio size in USD Captured is : {portfolio}")
                min_date = datetime(1980, 1, 1)
                # Date input widget with custom minimum date
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:", min_value=min_date)
                with col2:
                    end_date = st.date_input("End Date:")
                years = end_date.year - start_date.year
                st.success(f"years captured : {years}")
                if st.button("Check"):


                    # Registering converters for using matplotlib's plot_date() function.
                    register_matplotlib_converters()

                    # Setting display options for pandas
                    pd.set_option("display.max_columns", None)
                    pd.set_option("display.max_rows", None)

                    # Defining stocks to include in the portfolio
                    stocks = tickers

                    # Getting historical data from Yahoo Finance
                    start = start_date
                    end = end_date
                    df = yf.download(stocks, start=start, end=end)["Close"]

                    # Calculating daily returns of each stock
                    returns = df.pct_change()

                    # Define a function to generate random portfolios
                    def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
                        results = np.zeros((3, num_portfolios))
                        weights_record = []
                        for i in range(num_portfolios):
                            weights = np.random.random(n)
                            weights /= np.sum(weights)
                            weights_record.append(weights)
                            portfolio_std_dev, portfolio_return = portfolio_performance(
                                weights, mean_returns, cov_matrix
                            )
                            results[0, i] = portfolio_std_dev
                            results[1, i] = portfolio_return
                            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
                        return results, weights_record

                    # Calculating mean returns and covariance matrix of returns
                    mean_returns = returns.mean()
                    cov_matrix = returns.cov()

                    # Setting the number of random portfolios to generate and the risk-free rate
                    num_portfolios = 50000
                    risk_free_rate = 0.021

                    # Define a function to calculate the negative Sharpe ratio
                    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
                        p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
                        return -(p_ret - risk_free_rate) / p_var

                    # Define a function to find the portfolio with maximum Sharpe ratio
                    def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
                        num_assets = len(mean_returns)
                        args = (mean_returns, cov_matrix, risk_free_rate)
                        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
                        bound = (0.0, 1.0)
                        bounds = tuple(bound for asset in range(num_assets))
                        result = sco.minimize(
                            neg_sharpe_ratio,
                            num_assets
                            * [
                                1.0 / num_assets,
                            ],
                            args=args,
                            method="SLSQP",
                            bounds=bounds,
                            constraints=constraints,
                        )
                        return result

                    # Helper function to calculate portfolio performance
                    def portfolio_performance(weights, mean_returns, cov_matrix):
                        returns = np.sum(mean_returns * weights) * 252
                        std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                        return std_dev, returns

                    # Calculate portfolio volatility
                    def portfolio_volatility(weights, mean_returns, cov_matrix):
                        return portfolio_performance(weights, mean_returns, cov_matrix)[0]

                    # Function to find portfolio with minimum variance
                    def min_variance(mean_returns, cov_matrix):
                        num_assets = len(mean_returns)
                        args = (mean_returns, cov_matrix)
                        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
                        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

                        result = sco.minimize(portfolio_volatility, [1./num_assets]*num_assets,
                                            args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                        return result

                    # Function to calculate efficient return
                    def efficient_return(mean_returns, cov_matrix, target_return):
                        num_assets = len(mean_returns)
                        args = (mean_returns, cov_matrix)
                        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
                        constraints = [{'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1] - target_return},
                                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

                        result = sco.minimize(portfolio_volatility, [1./num_assets]*num_assets,
                                            args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                        return result

                    # Function to construct efficient frontier
                    def efficient_frontier(mean_returns, cov_matrix, returns_range):
                        efficient_portfolios = []
                        for ret in returns_range:
                            efficient_portfolios.append(efficient_return(mean_returns, cov_matrix, ret))
                        return efficient_portfolios

                    # Calculating efficient frontier
                    target_returns = np.linspace(0.0, 0.5, 100)
                    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target_returns)

                    # Extracting volatility and return for each portfolio
                    volatility = [p['fun'] for p in efficient_portfolios]
                    returns = [portfolio_performance(p['x'], mean_returns, cov_matrix)[1] for p in efficient_portfolios]

                    # Plotting the efficient frontier
                    fig_efficient_frontier = go.Figure()
                    fig_efficient_frontier.add_trace(go.Scatter(x=volatility, y=returns, mode='lines', name='Efficient Frontier'))
                    fig_efficient_frontier.add_trace(go.Scatter(x=[p['fun'] for p in min_variance(mean_returns, cov_matrix)], y=[p['x'].mean() for p in min_variance(mean_returns, cov_matrix)], mode='markers', marker=dict(size=10, color='red'), name='Minimum Variance Portfolio'))
                    fig_efficient_frontier.update_layout(title='Efficient Frontier', xaxis_title='Volatility', yaxis_title='Return')
                    st.plotly_chart(fig_efficient_frontier)

            if pred_option_portfolio_strategies == "Portfolio VAR Simulation":
                ticker = st.text_input("Please enter the ticker needed for investigation")
                if ticker:
                    message = (f"Ticker captured : {ticker}")
                    st.success(message)
                portfolio = st.number_input("Enter the portfolio size in USD")
                if portfolio:
                    st.write(f"The portfolio size in USD Captured is : {portfolio}")
                min_date = datetime(1980, 1, 1)
                # Date input widget with custom minimum date
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:", min_value=min_date)
                with col2:
                    end_date = st.date_input("End Date:")
                years = end_date.year - start_date.year
                st.success(f"years captured : {years}")
                if st.button("Check"):

                    # Define the tickers and time parameters
                    tickers = ['GOOGL', 'FB', 'AAPL', 'NFLX', 'AMZN']
                    Time = 1440  # Number of trading days in minutes
                    pvalue = 1000  # Portfolio value in dollars
                    num_of_years = 3
                    start_date = dt.datetime.now() - dt.timedelta(days=365.25 * num_of_years)
                    end_date = dt.datetime.now()

                    # Fetching and preparing stock data
                    price_data = [web.DataReader(ticker, start=start_date, end=end_date, data_source='yahoo')['Adj Close'] for ticker in tickers]
                    df_stocks = pd.concat(price_data, axis=1)
                    df_stocks.columns = tickers

                    # Calculating expected returns and covariance matrix
                    mu = expected_returns.mean_historical_return(df_stocks)
                    Sigma = risk_models.sample_cov(df_stocks)

                    # Portfolio Optimization using Efficient Frontier
                    ef = EfficientFrontier(mu, Sigma, weight_bounds=(0,1))
                    sharpe_pwt = ef.max_sharpe()
                    cleaned_weights = ef.clean_weights()

                    # Plotting Cumulative Returns of All Stocks
                    cum_returns = ((df_stocks.pct_change() + 1).cumprod() - 1)
                    fig_cum_returns = go.Figure()
                    for column in cum_returns.columns:
                        fig_cum_returns.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[column], mode='lines', name=column))
                    fig_cum_returns.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')
                    st.plotly_chart(fig_cum_returns)

                    # Portfolio VaR Simulation
                    ticker_returns = cum_returns.pct_change().dropna()
                    weighted_returns = ticker_returns.dot(np.array(list(cleaned_weights.values())))
                    portfolio_return = weighted_returns.mean()
                    portfolio_vol = weighted_returns.std()

                    # Simulating daily returns for VAR calculation
                    simulated_daily_returns = [np.random.normal(portfolio_return / Time, portfolio_vol / np.sqrt(Time), Time) for _ in range(10000)]

                    # Plotting Range of Returns in a Day
                    fig_returns_range = go.Figure()
                    for i in range(10000):
                        fig_returns_range.add_trace(go.Scatter(y=simulated_daily_returns[i], mode='lines', name=f'Simulation {i+1}'))
                    fig_returns_range.add_trace(go.Scatter(y=np.percentile(simulated_daily_returns, 5), mode='lines', name='5th Percentile', line=dict(color='red', dash='dash')))
                    fig_returns_range.add_trace(go.Scatter(y=np.percentile(simulated_daily_returns, 95), mode='lines', name='95th Percentile', line=dict(color='green', dash='dash')))
                    fig_returns_range.add_trace(go.Scatter(y=np.mean(simulated_daily_returns), mode='lines', name='Mean', line=dict(color='blue')))
                    fig_returns_range.update_layout(title=f'Range of Returns in a Day of {Time} Minutes', xaxis_title='Minute', yaxis_title='Returns')
                    st.plotly_chart(fig_returns_range)

                    # Histogram of Daily Returns
                    fig_hist_returns = go.Figure()
                    fig_hist_returns.add_trace(go.Histogram(x=simulated_daily_returns.flatten(), nbinsx=15))
                    fig_hist_returns.add_trace(go.Scatter(x=[np.percentile(simulated_daily_returns, 5), np.percentile(simulated_daily_returns, 5)], y=[0, 1500], mode='lines', name='5th Percentile', line=dict(color='red', dash='dash')))
                    fig_hist_returns.add_trace(go.Scatter(x=[np.percentile(simulated_daily_returns, 95), np.percentile(simulated_daily_returns, 95)], y=[0, 1500], mode='lines', name='95th Percentile', line=dict(color='green', dash='dash')))
                    fig_hist_returns.update_layout(title='Histogram of Daily Returns', xaxis_title='Returns', yaxis_title='Frequency')
                    st.plotly_chart(fig_hist_returns)

                    # Printing VaR results
                    st.write(f"5th Percentile: {np.percentile(simulated_daily_returns, 5)}")
                    st.write(f"95th Percentile: {np.percentile(simulated_daily_returns, 95)}")
                    st.write(f"Amount required to cover minimum losses for one day: ${pvalue * -np.percentile(simulated_daily_returns, 5)}")


        if pred_option_portfolio_strategies == "Risk Management":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                            
                # Define the start and end dates for data retrieval
                start = dt.datetime(2019, 1, 1)
                now = dt.datetime.now()

                # Define the moving averages and exponential moving averages to be used
                smaUsed = [50, 200]
                emaUsed = [21]

                # User inputs for stock ticker and position
                stock = st.text_input("Enter a ticker: ")
                position = st.selectbox("Buy or Short?", ["Buy", "Short"]).lower()
                AvgGain = st.number_input("Enter Your Average Gain (%)", value=0.0, step=0.1)
                AvgLoss = st.number_input("Enter Your Average Loss (%)", value=0.0, step=0.1)

                # Fetch historical data from Yahoo Finance
                df = yf.download(stock, start=start, end=now)

                # Calculate the maximum stop value and target returns based on user's position
                if position == "buy":
                    close = df["Adj Close"][-1]
                    maxStop = close * (1 - AvgLoss / 100)
                    targets = [round(close * (1 + (i * AvgGain / 100)), 2) for i in range(1, 4)]
                elif position == "short":
                    close = df["Adj Close"][-1]
                    maxStop = close * (1 + AvgLoss / 100)
                    targets = [round(close * (1 - (i * AvgGain / 100)), 2) for i in range(1, 4)]

                # Calculate SMA and EMA for the stock
                for x in smaUsed:
                    df[f"SMA_{x}"] = df["Adj Close"].rolling(window=x).mean()
                for x in emaUsed:
                    df[f"EMA_{x}"] = df["Adj Close"].ewm(span=x, adjust=False).mean()

                # Fetching the latest values of SMA, EMA, and 5 day low
                sma_values = {f"SMA_{x}": round(df[f"SMA_{x}"][-1], 2) for x in smaUsed}
                ema_values = {f"EMA_{x}": round(df[f"EMA_{x}"][-1], 2) for x in emaUsed}
                low5 = round(min(df["Low"].tail(5)), 2)

                # Calculate the performance metrics and checks
                performance_checks = {}
                for key, value in {**sma_values, **ema_values, "Low_5": low5}.items():
                    pf = round(((close / value) - 1) * 100, 2)
                    check = value > maxStop if position == "buy" else value < maxStop
                    performance_checks[key] = {"Performance": pf, "Check": check}

                # Displaying the results
                st.write(f"Current Stock: {stock} | Price: {round(close, 2)}")
                st.write(" | ".join([f"{key}: {value}" for key, value in {**sma_values, **ema_values, 'Low_5': low5}.items()]))
                st.write("-------------------------------------------------")
                st.write(f"Max Stop: {round(maxStop, 2)}")
                st.write(f"Price Targets: 1R: {targets[0]} | 2R: {targets[1]} | 3R: {targets[2]}")
                for key, value in performance_checks.items():
                    st.write(f"From {key} {value['Performance']}% - {'Within' if value['Check'] else 'Outside'} Max Stop")


        if pred_option_portfolio_strategies == "RSI Trendline Strategy":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):

                # Override the yfinance module
                yf.pdr_override()

                # Define the date range for data retrieval
                num_of_years = 10
                start = datetime.datetime.now() - datetime.timedelta(days=365.25 * num_of_years)
                end = datetime.datetime.now()

                # Load stock symbols
                @st.cache
                def load_tickers():
                    stocklist = ti.tickers_sp500()
                    return [stock.replace(".", "-") for stock in stocklist]  # Adjusting ticker format for Yahoo Finance

                stocklist = load_tickers()

                # Initialize the DataFrame for exporting results
                exportList = pd.DataFrame(columns=['Stock', "RSI", "200 Day MA"])

                # Process a limited number of stocks for demonstration
                for stock in stocklist[:5]:
                    time.sleep(1.5)  # To avoid hitting API rate limits
                    st.write(f"\npulling {stock}")

                    # Fetch stock data
                    df = pdr.get_data_yahoo(stock, start=start, end=end)

                    try:
                        # Calculate indicators: 200-day MA, RSI
                        df["SMA_200"] = df.iloc[:, 4].rolling(window=200).mean()
                        df["rsi"] = ta.RSI(df["Close"])
                        currentClose, moving_average_200, RSI = df["Adj Close"][-1], df["SMA_200"][-1], df["rsi"].tail(14).mean()
                        two_day_rsi_avg = (df.rsi[-1] + df.rsi[-2]) / 2

                        # Define entry criteria
                        if currentClose > moving_average_200 and two_day_rsi_avg < 33:
                            exportList = exportList.append({'Stock': stock, "RSI": RSI, "200 Day MA": moving_average_200}, ignore_index=True)
                            st.write(f"{stock} made the requirements")

                    except Exception as e:
                        st.write(e)  # Handling exceptions

                # Displaying the exported list
                st.write(exportList)


        if pred_option_portfolio_strategies == "RWB Strategy":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
        
                yf.pdr_override()

                emas_used = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]

                def get_stock_data(ticker, num_of_years):
                    start_date = dt.date.today() - dt.timedelta(days=365.25 * num_of_years)
                    end_date = dt.datetime.now()
                    df = pdr.get_data_yahoo(ticker, start_date, end_date).dropna()
                    for ema in emas_used:
                        df[f"Ema_{ema}"] = df.iloc[:, 4].ewm(span=ema, adjust=False).mean()
                    return df.iloc[60:]

                def rwb_strategy(df):
                    pos, num, percent_change = 0, 0, []
                    for i in df.index:
                        cmin = min(df[f"Ema_{ema}"][i] for ema in emas_used[:6])
                        cmax = max(df[f"Ema_{ema}"][i] for ema in emas_used[6:])
                        close = df["Adj Close"][i]
                        if cmin > cmax and pos == 0:
                            bp, pos = close, 1
                            st.write(f"Buying now at {bp}")
                        elif cmin < cmax and pos == 1:
                            pos, sp = 0, close
                            st.write(f"Selling now at {sp}")
                            percent_change.append((sp / bp - 1) * 100)
                        if num == df["Adj Close"].count() - 1 and pos == 1:
                            pos, sp = 0, close
                            st.write(f"Selling now at {sp}")
                            percent_change.append((sp / bp - 1) * 100)
                        num += 1
                    return percent_change

                st.title("RWB Strategy Visualization")

                stock = st.text_input("Enter a ticker:", "AAPL")
                num_of_years = st.number_input("Enter number of years:", min_value=1, max_value=10, step=1, value=5)

                df = get_stock_data(stock, num_of_years)
                percent_change = rwb_strategy(df)

                gains = sum(i for i in percent_change if i > 0)
                losses = sum(i for i in percent_change if i < 0)
                total_trades = len(percent_change)
                total_return = round((np.prod([1 + i/100 for i in percent_change]) - 1) * 100, 2)

                st.write(f"Results for {stock.upper()} going back to {num_of_years} years:")
                st.write(f"Number of Trades: {total_trades}")
                st.write(f"Total return: {total_return}%")

                fig = go.Figure()
                for ema in emas_used:
                    fig.add_trace(go.Scatter(x=df.index, y=df[f"Ema_{ema}"], mode='lines', name=f"Ema_{ema}"))
                fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name="Adj Close", line=dict(color='green')))
                fig.update_layout(title=f"RWB Strategy for {stock.upper()}", xaxis_title="Date", yaxis_title="Price", template='plotly_dark')
                st.plotly_chart(fig)


        
        if pred_option_portfolio_strategies == "SMA Trading Strategy":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):
                # Define the function to get historical data
                def get_stock_data(stock, num_of_years):
                    start = dt.date.today() - dt.timedelta(days=365 * num_of_years)
                    end = dt.datetime.now()
                    return yf.download(stock, start, end, interval='1d')

                # Define the function for SMA Trading Strategy
                def sma_trading_strategy(df, short_sma, long_sma):
                    df[f"SMA_{short_sma}"] = df['Adj Close'].rolling(window=short_sma).mean()
                    df[f"SMA_{long_sma}"] = df['Adj Close'].rolling(window=long_sma).mean()

                    position = 0
                    percent_change = []
                    for i in df.index:
                        close = df['Adj Close'][i]
                        SMA_short = df[f"SMA_{short_sma}"][i]
                        SMA_long = df[f"SMA_{long_sma}"][i]

                        if SMA_short > SMA_long and position == 0:
                            buyP, position = close, 1
                            st.write("Buy at the price:", buyP)
                        elif SMA_short < SMA_long and position == 1:
                            sellP, position = close, 0
                            st.write("Sell at the price:", sellP)
                            percent_change.append((sellP / buyP - 1) * 100)

                    if position == 1:
                        position = 0
                        sellP = df['Adj Close'][-1]
                        st.write("Sell at the price:", sellP)
                        percent_change.append((sellP / buyP - 1) * 100)

                    return percent_change

                # Main script
                st.title("SMA Trading Strategy Visualization")

                stock = st.text_input("Enter a ticker:", "NFLX")
                num_of_years = st.number_input("Enter number of years:", min_value=1, max_value=10, step=1, value=5)
                short_sma = st.number_input("Enter short SMA:", min_value=1, value=20)
                long_sma = st.number_input("Enter long SMA:", min_value=1, value=50)

                df = get_stock_data(stock, num_of_years)
                percent_change = sma_trading_strategy(df, short_sma, long_sma)
                current_price = round(df['Adj Close'][-1], 2)

                # Calculate strategy statistics
                gains = 0
                numGains = 0
                losses = 0
                numLosses = 0
                totReturn = 1
                for i in percent_change:
                    if i > 0:
                        gains += i
                        numGains += 1
                    else:
                        losses += i
                        numLosses += 1
                    totReturn = totReturn * ((i / 100) + 1)
                totReturn = round((totReturn - 1) * 100, 2)
                # Plot SMA and Adj Close
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{short_sma}"], mode='lines', name=f"SMA_{short_sma}"))
                fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{long_sma}"], mode='lines', name=f"SMA_{long_sma}"))
                fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name="Adj Close", line=dict(color='green')))
                fig.update_layout(title=f"SMA Trading Strategy for {stock.upper()}", xaxis_title="Date", yaxis_title="Price", template='plotly_dark')
                st.plotly_chart(fig)

                # Display strategy statistics
                st.write(f"Results for {stock.upper()} going back to {num_of_years} years:")
                st.write(f"Number of Trades: {numGains + numLosses}")
                st.write(f"Total return: {totReturn}%")


        
    
        if pred_option_portfolio_strategies == "Stock Spread Plotter":
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")

            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"Years captured: {years}")
            # Initialize the stocks list in session state if it doesn't exist
            if 'stocks' not in st.session_state:
                st.session_state.stocks = []

            # Function to add stock
            def addStock():
                tsk = st.text_input('Enter your tickers', key='ticker_input', placeholder='Enter a ticker')
                if tsk:
                    st.session_state.stocks.append(tsk)
                    st.success(f"Ticker {tsk} added.")
                    st.experimental_rerun()  # Refresh the app to show the updated list

            # Display the add stock button and call the function if clicked
            if st.button('Add Ticker'):
                addStock()

            # Display the list of stocks
            st.write("Current list of tickers:")
            st.write(st.session_state.stocks)
            threshold = st.slider("Threshold", min_value=0.1, max_value=5.0, value=0.5)
            stop_loss = st.slider("Stop Loss", min_value=0.1, max_value=5.0, value=1.0)
            if st.button("Check"):
                # Function to fetch stock data
                def fetch_stock_data(tickers, start_date, end_date):
                    data = pdr.get_data_yahoo(tickers, start=start_date, end=end_date)['Adj Close']
                    data.index = pd.to_datetime(data.index)
                    return data

                def plot_stock_spread(df, ticker1, ticker2, threshold=0.5, stop_loss=1):
                    spread = df[ticker1] - df[ticker2]
                    mean_spread = spread.mean()
                    sell_threshold = mean_spread + threshold
                    buy_threshold = mean_spread - threshold
                    sell_stop = mean_spread + stop_loss
                    buy_stop = mean_spread - stop_loss

                    fig = go.Figure()

                    # Add the individual stock prices
                    fig.add_trace(go.Scatter(x=df.index, y=df[ticker1], mode='lines', name=ticker1))
                    fig.add_trace(go.Scatter(x=df.index, y=df[ticker2], mode='lines', name=ticker2))

                    # Add the spread
                    fig.add_trace(go.Scatter(x=df.index, y=spread, mode='lines', name='Spread', line=dict(color='#85929E')))

                    # Add threshold and stop lines
                    fig.add_hline(y=sell_threshold, line=dict(color='blue', dash='dash'), name='Sell Threshold')
                    fig.add_hline(y=buy_threshold, line=dict(color='red', dash='dash'), name='Buy Threshold')
                    fig.add_hline(y=sell_stop, line=dict(color='green', dash='dash'), name='Sell Stop')
                    fig.add_hline(y=buy_stop, line=dict(color='yellow', dash='dash'), name='Buy Stop')

                    # Update layout for better presentation
                    fig.update_layout(
                        title=f'Stock Spread between {ticker1} and {ticker2}',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        legend_title='Legend',
                        template='plotly_white'
                    )

                    st.plotly_chart(fig)
                # Main
                if len(st.session_state.stocks) >= 2:
                    df = fetch_stock_data(st.session_state.stocks, start_date, end_date)
                    for i in range(len(st.session_state.stocks) - 1):
                        plot_stock_spread(df, st.session_state.stocks[i], st.session_state.stocks[i + 1], threshold, stop_loss)
                else:
                    st.error("Please enter at least two tickers for the analysis.")

        if pred_option_portfolio_strategies == "Support Resistance Finder":
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Check"):

                # Function to retrieve stock data
                def fetch_stock_data(ticker, start_date, end_date):
                    df = yf.download(ticker, start=start_date, end=end_date)
                    df["Date"] = df.index
                    return df.reset_index(drop=True)

                # Function to identify support and resistance levels
                def identify_levels(df):
                    levels = []
                    for i in range(2, df.shape[0] - 2):
                        if is_support(df, i):
                            levels.append((i, df["Low"][i], "Support"))
                        elif is_resistance(df, i):
                            levels.append((i, df["High"][i], "Resistance"))
                    return levels

                # Define support and resistance checks
                def is_support(df, i):
                    return df["Low"][i] < min(df["Low"][i - 1], df["Low"][i + 1])

                def is_resistance(df, i):
                    return df["High"][i] > max(df["High"][i - 1], df["High"][i + 1])

                # Function to plot support and resistance levels
                def plot_support_resistance(df, levels):
                    fig, ax = plt.subplots()
                    candlestick_ohlc(ax, zip(mpl_dates.date2num(df['Date']), df['Open'], df['High'], df['Low'], df['Close']), width=0.6, colorup='green', colordown='red', alpha=0.8)
                    ax.xaxis.set_major_formatter(mpl_dates.DateFormatter('%d-%m-%Y'))

                    for level in levels:
                        plt.hlines(level[1], xmin=df["Date"][level[0]], xmax=max(df["Date"]), colors="blue")
                    plt.title(f"Support and Resistance for {ticker.upper()}")
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    st.pyplot(fig)

                # Main
                st.title("Support and Resistance Levels Visualization")

                ticker = st.text_input("Enter a ticker:")
                num_of_years = st.slider("Number of years:", min_value=0.1, max_value=10.0, value=0.2, step=0.1)

                start_date = pd.Timestamp.now() - pd.Timedelta(days=int(365.25 * num_of_years))
                end_date = pd.Timestamp.now()

                df = fetch_stock_data(ticker, start_date, end_date)
                levels = identify_levels(df)
                plot_support_resistance(df, levels)

    elif option == "Algorithmic Trading":
        AI_option_trading = st.selectbox('Make a choice', ["Lumibots : Diversified Leverage", "Lumibots : Stock Bracket Strategy","Lumibots : Hold to Expiry","Lumibots : Important functions (Crypto)", "Lumibots : Stock Limit & Trailing Stops","Lumibots : Momentum Strategy","Lumibots : Stock OCO Strategy","Lumibots : CCXT Backtesting Strategy","Lumibots : Buy & Hold Strategy", "Lumibots : GLD signal", "Lumibots : Trend Strategy", "Lumibots : Swing High Strategy"])
        if AI_option_trading == 'Lumibots : Buy & Hold Strategy':
            st.write("Lumibots buy hold strategy for Long term investors")
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):
                    
                def initialize(self):
                    self.sleeptime = "1D"

                def on_trading_iteration(self):
                    if self.first_iteration:
                        stocks_and_quantities = [
                            {"symbol": ticker_input, "quantity": quantities_input},
                        ]
                        for stock_info in stocks_and_quantities:
                            symbol = stock_info["symbol"]
                            quantity = stock_info["quantity"]
                            price = self.get_last_price(symbol)
                            cost = price * quantity
                            self.cash = portfolio_size
                            if self.cash >= cost:
                                order = self.create_order(symbol, quantity, "buy")
                                self.submit_order(order)
            
                broker = Alpaca(ALPACA_CONFIG)
                strategy = BuyHold(broker=broker)
                trader = Trader()
                trader.add_strategy(strategy)
                trader.run_all()

        if AI_option_trading == 'Lumibots : GLD Signal Strategy':
            st.write("Lumibots buy hold strategy for Long term investors")
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):
                gld = pd.DataFrame(yf.download(ticker_input, start_date)['Close'])
                gld['9-day'] = gld['Close'].rolling(9).mean()
                gld['21-day'] = gld['Close'].rolling(21).mean()
                gld['Signal'] = np.where(np.logical_and(gld['9-day'] > gld['21-day'],
                                        gld['9-day'].shift(1) < gld['21-day'].shift(1)),
                                        "BUY", None)
                gld['Signal'] = np.where(np.logical_and(gld['9-day'] < gld['21-day'],
                                        gld['9-day'].shift(1) > gld['21-day'].shift(1)),
                                        "SELL", gld['Signal'])

                def signal(df, start=start_date, end=end_date):
                    df = pd.DataFrame(yf.download(ticker_input, start, end)['Close'])
                    df['9-day'] = df['Close'].rolling(9).mean()
                    df['21-day'] = df['Close'].rolling(21).mean()
                    df['Signal'] = np.where(np.logical_and(df['9-day'] > df['21-day'],
                                            df['9-day'].shift(1) < df['21-day'].shift(1)),
                                            "BUY", None)
                    df['Signal'] = np.where(np.logical_and(df['9-day'] < df['21-day'],
                                            df['9-day'].shift(1) > df['21-day'].shift(1)),
                                            "SELL", df['Signal'])
                    return df, df.iloc[-1].Signal

            
        if AI_option_trading == 'Lumibots : Swing High Strategy':
            st.write("Lumibots buy hold strategy for Long term investors")
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):
                class SwingHigh(Strategy):
                    data = {}  # Dictionary to store last price data for each symbol
                    order_numbers = {}  # Dictionary to store order numbers for each symbol
                    shares_per_ticker = {}  # Dictionary to specify the number of shares per ticker

                    def initialize(self, ticker_input, quantities_input):
                        self.symbols = ticker_input# Add other symbols as needed
                        self.shares_per_ticker = {ticker_input: quantities_input}   # Specify the number of shares for each symbol
                        self.sleeptime = "10S"
                    
                    def on_trading_iteration(self):
                        for symbol in self.symbols:
                            if symbol not in self.data:
                                self.data[symbol] = []

                            entry_price = self.get_last_price(symbol)
                            self.log_message(f"Position for {symbol}: {self.get_position(symbol)}")
                            self.data[symbol].append(entry_price)

                            if len(self.data[symbol]) > 3:
                                temp = self.data[symbol][-3:]
                                if temp[-1] > temp[1] > temp[0]:
                                    self.log_message(f"Last 3 prints for {symbol}: {temp}")
                                    order = self.create_order(symbol, quantity=self.shares_per_ticker[symbol], side="buy")
                                    self.submit_order(order)
                                    if symbol not in self.order_numbers:
                                        self.order_numbers[symbol] = 0
                                    self.order_numbers[symbol] += 1
                                    if self.order_numbers[symbol] == 1:
                                        self.log_message(f"Entry price for {symbol}: {temp[-1]}")
                                        entry_price = temp[-1]  # filled price
                                if self.get_position(symbol) and self.data[symbol][-1] < entry_price * 0.995:
                                    self.sell_all(symbol)
                                    self.order_numbers[symbol] = 0
                                elif self.get_position(symbol) and self.data[symbol][-1] >= entry_price * 1.015:
                                    self.sell_all(symbol)
                                    self.order_numbers[symbol] = 0

                    def before_market_closes(self):
                        for symbol in self.symbols:
                            self.sell_all(symbol)

        if AI_option_trading == 'Lumibots : Trend Strategy':
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):
                class Trend(Strategy):

                    def initialize(self):
                        signal = None
                        start = start_date

                        self.signal = signal
                        self.start = start
                        self.sleeptime = "1D"
                    # minute bars, make functions    

                    def on_trading_iteration(self):
                        bars = self.get_historical_prices("GLD", 22, "day")
                        gld = bars.df
                        #gld = pd.DataFrame(yf.download("GLD", self.start)['Close'])
                        gld['9-day'] = gld['close'].rolling(9).mean()
                        gld['21-day'] = gld['close'].rolling(21).mean()
                        gld['Signal'] = np.where(np.logical_and(gld['9-day'] > gld['21-day'],
                                                                gld['9-day'].shift(1) < gld['21-day'].shift(1)),
                                                "BUY", None)
                        gld['Signal'] = np.where(np.logical_and(gld['9-day'] < gld['21-day'],
                                                                gld['9-day'].shift(1) > gld['21-day'].shift(1)),
                                                "SELL", gld['Signal'])
                        self.signal = gld.iloc[-1].Signal
                        
                        symbol = "GLD"
                        quantity = 200
                        if self.signal == 'BUY':
                            pos = self.get_position(symbol)
                            if pos is not None:
                                self.sell_all()
                                
                            order = self.create_order(symbol, quantity, "buy")
                            self.submit_order(order)

                        elif self.signal == 'SELL':
                            pos = self.get_position(symbol)
                            if pos is not None:
                                self.sell_all()
                                
                            order = self.create_order(symbol, quantity, "sell")
                            self.submit_order(order)

                    
                if __name__ == "__main__":
                    trade = False
                    #Reactivate after code rebase
                    if trade:
                        broker = Alpaca(ALPACA_CONFIG)
                        strategy = Trend(broker=broker)
                        bot = Trader()
                        bot.add_strategy(strategy)
                        bot.run_all()
                    else:
                        start = start_date
                        end = end_date
                        Trend.backtest(
                            YahooDataBacktesting,
                            start,
                            end
                        )
        if AI_option_trading == 'Lumibots : CCXT Backtesting Strategy':
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):
                class CcxtBacktestingExampleStrategy(Strategy):
                    def initialize(self, asset:tuple[Asset,Asset] = None,
                                cash_at_risk:float=.25,window:int=21):
                        if asset is None:
                            raise ValueError("You must provide a valid asset pair")
                        # for crypto, market is 24/7
                        self.set_market("24/7")
                        self.sleeptime = "1D"
                        self.asset = asset
                        self.base, self.quote = asset
                        self.window = window
                        self.symbol = f"{self.base.symbol}/{self.quote.symbol}"
                        self.last_trade = None
                        self.order_quantity = 0.0
                        self.cash_at_risk = cash_at_risk

                    def _position_sizing(self):
                        cash = self.get_cash()
                        last_price = self.get_last_price(asset=self.asset,quote=self.quote)
                        quantity = round(cash * self.cash_at_risk / last_price,0)
                        return cash, last_price, quantity

                    def _get_historical_prices(self):
                        return self.get_historical_prices(asset=self.asset,length=None,
                                                    timestep="day",quote=self.quote).df

                    def _get_bbands(self,history_df:DataFrame):
                        # BBL (Lower Bollinger Band): Can act as a support level based on price volatility, and can indicate an 'oversold' condition if the price falls below this line.
                        # BBM (Breaking Bollinger Bands): This is essentially a moving average over a selected period of time, used as a reference point for price trends.
                        # BBU (Upper Bollinger Band): Can act as a resistance level based on price volatility, and can indicate an 'overbought' condition if the price moves above this line.
                        # BBB (Bollinger Band Width): Indicates the distance between the upper and lower bands, with a higher value indicating a more volatile market.
                        # BBP (Bollinger Band Percentage): This shows where the current price is located within the Bollinger Bands as a percentage, where a value close to 0 means that the price is close to the lower band, and a value close to 1 means that the price is close to the upper band.
                        # return bbands
                        num_std_dev = 2.0
                        close = 'close'

                        df = DataFrame(index=history_df.index)
                        df[close] = history_df[close]
                        df['bbm'] = df[close].rolling(window=self.window).mean()
                        df['bbu'] = df['bbm'] + df[close].rolling(window=self.window).std() * num_std_dev
                        df['bbl'] = df['bbm'] - df[close].rolling(window=self.window).std() * num_std_dev
                        df['bbb'] = (df['bbu'] - df['bbl']) / df['bbm']
                        df['bbp'] = (df[close] - df['bbl']) / (df['bbu'] - df['bbl'])
                        return df

                    def on_trading_iteration(self):
                        # During the backtest, we get the current time with self.get_datetime().
                        # The time interval is self.sleeptime.
                        current_dt = self.get_datetime()
                        cash, last_price, quantity = self._position_sizing()
                        history_df = self._get_historical_prices()
                        bbands = self._get_bbands(history_df)
                        prev_bbp = bbands[bbands.index < current_dt].tail(1).bbp.values[0]

                        if prev_bbp < -0.13 and cash > 0 and self.last_trade != Order.OrderSide.BUY and quantity > 0.0:
                            order = self.create_order(self.base,
                                                    quantity,
                                                    side = Order.OrderSide.BUY,
                                                    type = Order.OrderType.MARKET,
                                                    quote=self.quote)
                            self.submit_order(order)
                            self.last_trade = Order.OrderSide.BUY
                            self.order_quantity = quantity
                            self.log_message(f"Last buy trade was at {current_dt}")
                        elif prev_bbp > 1.2 and self.last_trade != Order.OrderSide.SELL and self.order_quantity > 0.0:
                            order = self.create_order(self.base,
                                                    self.order_quantity,
                                                    side = Order.OrderSide.SELL,
                                                    type = Order.OrderType.MARKET,
                                                    quote=self.quote)
                            self.submit_order(order)
                            self.last_trade = Order.OrderSide.SELL
                            self.order_quantity = 0.0
                            self.log_message(f"Last sell trade was at {current_dt}")

                    base_symbol = "ETH" # TODO-Adjust for either SPY/ETH for crypto
                    quote_symbol = ticker
                    start_date = datetime(2023,2,11)
                    end_date = datetime(2024,2,12)
                    asset = (Asset(symbol=base_symbol, asset_type="crypto"),
                            Asset(symbol=quote_symbol, asset_type="crypto"))

                    exchange_id = "kraken"  #"kucoin" #"bybit" #"okx" #"bitmex" # "binance"


                    # CcxtBacktesting default data download limit is 50,000
                    # If you want to change the maximum data download limit, you can do so by using 'max_data_download_limit'.
                    kwargs = {
                        # "max_data_download_limit":10000, # optional
                        "exchange_id":exchange_id,
                    }
                    CcxtBacktesting.MIN_TIMESTEP = "day"
                    results, strat_obj = CcxtBacktestingExampleStrategy.run_backtest(
                        CcxtBacktesting,
                        start_date,
                        end_date,
                        benchmark_asset=f"{base_symbol}/{quote_symbol}",
                        quote_asset=Asset(symbol=quote_symbol, asset_type="crypto"),
                        parameters={
                                "asset":asset,
                                "cash_at_risk":.25,
                                "window":21,},
                        **kwargs,
                    )
        if AI_option_trading == 'Lumibots : Important functions (Crypto)':
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):
                class ImportantFunctions(Strategy):
                    def initialize(self):
                        # Set the time between trading iterations
                        self.sleeptime = "30S"

                        # Set the market to 24/7 since those are the hours for the crypto market
                        self.set_market("24/7")

                    def on_trading_iteration(self):
                        ###########################
                        # Placing an Order
                        ###########################

                        # Define the base and quote assets for our transactions
                        base = Asset(symbol="BTC", asset_type="crypto")
                        quote = self.quote_asset

                        # Market Order for 0.1 BTC
                        mkt_order = self.create_order(base, 0.1, "buy", quote=quote)
                        self.submit_order(mkt_order)

                        # Limit Order for 0.1 BTC at a limit price of $10,000
                        lmt_order = self.create_order(base, 0.1, "buy", quote=quote, limit_price=10000)
                        self.submit_order(lmt_order)

                        ###########################
                        # Getting Historical Data
                        ###########################

                        # Get the historical prices for our base/quote pair for the last 100 minutes
                        bars = self.get_historical_prices(base, 100, "minute", quote=quote)
                        if bars is not None:
                            df = bars.df
                            max_price = df["close"].max()
                            self.log_message(f"Max price for {base} was {max_price}")

                            ############################
                            # TECHNICAL ANALYSIS
                            ############################

                            # Use pandas_ta to calculate the 20 period RSI
                            rsi = df.ta.rsi(length=20)
                            current_rsi = rsi.iloc[-1]
                            self.log_message(f"RSI for {base} was {current_rsi}")

                            # Use pandas_ta to calculate the MACD
                            macd = df.ta.macd()
                            current_macd = macd.iloc[-1]
                            self.log_message(f"MACD for {base} was {current_macd}")

                            # Use pandas_ta to calculate the 55 EMA
                            ema = df.ta.ema(length=55)
                            current_ema = ema.iloc[-1]
                            self.log_message(f"EMA for {base} was {current_ema}")

                        ###########################
                        # Positions and Orders
                        ###########################

                        # Get all the positions that we own, including cash
                        positions = self.get_positions()
                        for position in positions:
                            self.log_message(f"Position: {position}")

                            # Get the asset of the position
                            asset = position.asset

                            # Get the quantity of the position
                            quantity = position.quantity

                            # Get the symbol from the asset
                            symbol = asset.symbol

                            self.log_message(f"we own {quantity} shares of {symbol}")

                        # Get one specific position
                        asset_to_get = Asset(symbol="BTC", asset_type="crypto")
                        position = self.get_position(asset_to_get)

                        # Get all of the outstanding orders
                        orders = self.get_orders()
                        for order in orders:
                            self.log_message(f"Order: {order}")
                            # Do whatever you need to do with the order

                        # Get one specific order
                        order = self.get_order(mkt_order.identifier)

                        ###########################
                        # Other Useful Functions
                        ###########################

                        # Get the current (last) price for the base/quote pair
                        last_price = self.get_last_price(base, quote=quote)
                        self.log_message(
                            f"Last price for {base}/{quote} was {last_price}", color="green"
                        )

                        dt = self.get_datetime()
                        self.log_message(f"The current datetime is {dt}")
                        self.log_message(f"The current time is {dt.time()}")

                        # If you want to check if it's after a certain time, you can do this (eg. trading only after 9:30am)
                        if dt.time() > datetime.time(hour=9, minute=30):
                            self.log_message("It's after 9:30am")

                        # Get the value of the entire portfolio, including positions and cash
                        portfolio_value = self.portfolio_value
                        # Get the amount of cash in the account (the amount in the quote_asset)
                        cash = self.cash

                        self.log_message(f"The current value of your account is {portfolio_value}")
                        # Note: Cash is based on the quote asset
                        self.log_message(f"The current amount of cash in your account is {cash}")


                if __name__ == "__main__":
                    trader = Trader()

                
                    broker = Ccxt(KRAKEN_CONFIG)

                    strategy = ImportantFunctions(
                        broker=broker,
                    )

                    trader.add_strategy(strategy)
                    strategy_executors = trader.run_all()

        if AI_option_trading == 'Lumibots : Hold to Expiry':
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):
                """
                Strategy Description
                An example strategy for buying an option and holding it to expiry.
                """


                class OptionsHoldToExpiry(Strategy):
                    parameters = {
                        "buy_symbol": "SPY",
                        "expiry": datetime(2023, 10, 20),
                    }

                    # =====Overloading lifecycle methods=============

                    def initialize(self):
                        # Set the initial variables or constants

                        # Built in Variables
                        self.sleeptime = "1D"

                    def on_trading_iteration(self):
                        """Buys the self.buy_symbol once, then never again"""

                        buy_symbol = self.parameters["buy_symbol"]
                        expiry = self.parameters["expiry"]

                        # What to do each iteration
                        underlying_price = self.get_last_price(buy_symbol)
                        self.log_message(f"The value of {buy_symbol} is {underlying_price}")

                        if self.first_iteration:
                            # Calculate the strike price (round to nearest 1)
                            strike = round(underlying_price)

                            # Create options asset
                            asset = Asset(
                                symbol=buy_symbol,
                                asset_type="option",
                                expiration=expiry,
                                strike=strike,
                                right="call",
                            )

                            # Create order
                            order = self.create_order(
                                asset,
                                10,
                                "buy_to_open",
                            )
                            
                            # Submit order
                            self.submit_order(order)

                            # Log a message
                            self.log_message(f"Bought {order.quantity} of {asset}")


                if __name__ == "__main__":
                    is_live = False

                    if is_live:

                        from lumibot.brokers import InteractiveBrokers
                        from lumibot.traders import Trader

                        trader = Trader()

                        broker = InteractiveBrokers(INTERACTIVE_BROKERS_CONFIG)
                        strategy = OptionsHoldToExpiry(broker=broker)

                        trader.add_strategy(strategy)
                        strategy_executors = trader.run_all()

                    else:
                        from lumibot.backtesting import PolygonDataBacktesting

                        # Backtest this strategy
                        backtesting_start = datetime(2023, 10, 19)
                        backtesting_end = datetime(2023, 10, 24)

                        results = OptionsHoldToExpiry.backtest(
                            PolygonDataBacktesting,
                            backtesting_start,
                            backtesting_end,
                            benchmark_asset="SPY",
                            polygon_api_key="YOUR_POLYGON_API_KEY_HERE",  # Add your polygon API key here
                            polygon_has_paid_subscription=False,
                        )




        if AI_option_trading == 'Lumibots : Stock Bracket Strategy':
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):
                """
                Strategy Description

                An example strategy for how to use bracket orders.
                """
                class StockBracket(Strategy):
                    parameters = {
                        "buy_symbol": "SPY",
                        "take_profit_price": 405,
                        "stop_loss_price": 395,
                        "quantity": 10,
                    }

                    # =====Overloading lifecycle methods=============

                    def initialize(self):
                        # Set the initial variables or constants

                        # Built in Variables
                        self.sleeptime = "1D"

                        # Our Own Variables
                        self.counter = 0

                    def on_trading_iteration(self):
                        """Buys the self.buy_symbol once, then never again"""

                        buy_symbol = self.parameters["buy_symbol"]
                        take_profit_price = self.parameters["take_profit_price"]
                        stop_loss_price = self.parameters["stop_loss_price"]
                        quantity = self.parameters["quantity"]

                        # What to do each iteration
                        current_value = self.get_last_price(buy_symbol)
                        self.log_message(f"The value of {buy_symbol} is {current_value}")

                        if self.first_iteration:
                            # Bracket order
                            order = self.create_order(
                                buy_symbol,
                                quantity,
                                "buy",
                                take_profit_price=take_profit_price,
                                stop_loss_price=stop_loss_price,
                                type="bracket",
                            )
                            self.submit_order(order)


                if __name__ == "__main__":
                    is_live = False

                    if is_live:

                        from lumibot.brokers import Alpaca
                        from lumibot.traders import Trader

                        trader = Trader()

                        broker = Alpaca(ALPACA_CONFIG)

                        strategy = StockBracket(broker=broker)

                        trader.add_strategy(strategy)
                        strategy_executors = trader.run_all()

                    else:
                        from lumibot.backtesting import YahooDataBacktesting

                        # Backtest this strategy
                        backtesting_start = datetime(2023, 3, 3)
                        backtesting_end = datetime(2023, 3, 10)

                        results = StockBracket.backtest(
                            YahooDataBacktesting,
                            backtesting_start,
                            backtesting_end,
                            benchmark_asset="SPY",
                        )




        if AI_option_trading == 'Lumibots : Diversified Leverage':
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):
                """
                Strategy Description

                This strategy will buy a few symbols that have 2x or 3x returns (have leverage), but will 
                also diversify and rebalance the portfolio often.
                """


                class DiversifiedLeverage(Strategy):
                    # =====Overloading lifecycle methods=============

                    parameters = {
                        "portfolio": [
                            {
                                "symbol": "TQQQ",  # 3x Leveraged Nasdaq
                                "weight": 0.20,
                            },
                            {
                                "symbol": "UPRO",  # 3x Leveraged S&P 500
                                "weight": 0.20,
                            },
                            {
                                "symbol": "UDOW",  # 3x Leveraged Dow Jones
                                "weight": 0.10,
                            },
                            {
                                "symbol": "TMF",  # 3x Leveraged Treasury Bonds
                                "weight": 0.25,
                            },
                            {
                                "symbol": "UGL",  # 3x Leveraged Gold
                                "weight": 0.10,
                            },
                            {
                                "symbol": "DIG",  # 2x Leveraged Oil and Gas Companies (Commodities)
                                "weight": 0.15,
                            },
                        ],
                        "rebalance_period": 4,
                    }

                    def initialize(self):
                        # Setting the waiting period (in days) and the counter
                        self.counter = None

                        # There is only one trading operation per day
                        # no need to sleep between iterations
                        self.sleeptime = "1D"

                        # Initializing the portfolio variable with the assets and proportions we want to own
                        self.initialized = False

                        self.minutes_before_closing = 1

                    def on_trading_iteration(self):
                        rebalance_period = self.parameters["rebalance_period"]
                        # If the target number of days (period) has passed, rebalance the portfolio
                        if self.counter == rebalance_period or self.counter == None:
                            self.counter = 0
                            self.rebalance_portfolio()
                            self.log_message(
                                f"Next portfolio rebalancing will be in {rebalance_period} day(s)"
                            )

                        self.log_message("Sleeping until next trading day")
                        self.counter += 1

                    # =============Helper methods====================

                    def rebalance_portfolio(self):
                        """Rebalance the portfolio and create orders"""

                        orders = []
                        for asset in self.parameters["portfolio"]:
                            # Get all of our variables from portfolio
                            symbol = asset.get("symbol")
                            weight = asset.get("weight")
                            last_price = self.get_last_price(symbol)

                            # Get how many shares we already own
                            # (including orders that haven't been executed yet)
                            position = self.get_position(symbol)
                            quantity = 0
                            if position is not None:
                                quantity = float(position.quantity)

                            # Calculate how many shares we need to buy or sell
                            shares_value = self.portfolio_value * weight
                            self.log_message(
                                f"The current portfolio value is {self.portfolio_value} and the weight needed is {weight}, so we should buy {shares_value}"
                            )
                            new_quantity = shares_value // last_price
                            quantity_difference = new_quantity - quantity
                            self.log_message(
                                f"Currently own {quantity} shares of {symbol} but need {new_quantity}, so the difference is {quantity_difference}"
                            )

                            # If quantity is positive then buy, if it's negative then sell
                            side = ""
                            if quantity_difference > 0:
                                side = "buy"
                            elif quantity_difference < 0:
                                side = "sell"

                            # Execute the order if necessary
                            if side:
                                order = self.create_order(symbol, abs(quantity_difference), side)
                                orders.append(order)

                        self.submit_orders(orders)


                    if __name__ == "__main__":
                        is_live = False

                        if is_live:
                            ####
                            # Run the strategy live
                            ####

                            trader = Trader()
                            broker = Alpaca(ALPACA_CONFIG)
                            strategy = DiversifiedLeverage(broker=broker)
                            trader.add_strategy(strategy)
                            trader.run_all()

                        else:
                            ####
                            # Backtest the strategy
                            ####

                            # Choose the time from and to which you want to backtest
                            backtesting_start = datetime(2010, 6, 1)
                            backtesting_end = datetime(2023, 7, 31)

                            # 0.01% trading/slippage fee
                            trading_fee = TradingFee(percent_fee=0.005)

                            # Initialize the backtesting object
                            print("Starting Backtest...")
                            result = DiversifiedLeverage.backtest(
                                YahooDataBacktesting,
                                backtesting_start,
                                backtesting_end,
                                benchmark_asset="SPY",
                                parameters={},
                                buy_trading_fees=[trading_fee],
                                sell_trading_fees=[trading_fee],
                            )

                            print("Backtest result: ", result)



        if AI_option_trading == 'Lumibots : Stock Limit & Trailing Stops':
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):

                """
                Strategy Description

                An example of how to use limit orders and trailing stops to buy a stock and then sell it when it drops by a certain
                percentage. This is a very simple strategy that is meant to demonstrate how to use limit orders and trailing stops.
                """


                class LimitAndTrailingStop(Strategy):
                    parameters = {
                        "buy_symbol": "SPY",
                        "limit_buy_price": 403,
                        "limit_sell_price": 407,
                        "trail_percent": 0.02,
                        "trail_price": 7,
                    }

                    # =====Overloading lifecycle methods=============

                    def initialize(self):
                        # Set the initial variables or constants

                        # Built in Variables
                        self.sleeptime = "1D"

                        # Our Own Variables
                        self.counter = 0

                    def on_trading_iteration(self):
                        """Buys the self.buy_symbol once, then never again"""

                        buy_symbol = self.parameters["buy_symbol"]
                        limit_buy_price = self.parameters["limit_buy_price"]
                        limit_sell_price = self.parameters["limit_sell_price"]
                        trail_percent = self.parameters["trail_percent"]
                        trail_price = self.parameters["trail_price"]

                        # What to do each iteration
                        current_value = self.get_last_price(buy_symbol)
                        self.log_message(f"The value of {buy_symbol} is {current_value}")

                        if self.first_iteration:
                            # Create the limit buy order
                            purchase_order = self.create_order(buy_symbol, 100, "buy", limit_price=limit_buy_price)
                            self.submit_order(purchase_order)

                            # Create the limit sell order
                            sell_order = self.create_order(buy_symbol, 100, "sell", limit_price=limit_sell_price)
                            self.submit_order(sell_order)

                            # Place the trailing percent stop
                            trailing_pct_stop_order = self.create_order(buy_symbol, 100, "sell", trail_percent=trail_percent)
                            self.submit_order(trailing_pct_stop_order)

                            # Place the trailing price stop
                            trailing_price_stop_order = self.create_order(buy_symbol, 50, "sell", trail_price=trail_price)
                            self.submit_order(trailing_price_stop_order)


                if __name__ == "__main__":
                    is_live = False

                    if is_live:

                        from lumibot.brokers import Alpaca
                        from lumibot.traders import Trader

                        trader = Trader()

                        broker = Alpaca(ALPACA_CONFIG)

                        strategy = LimitAndTrailingStop(broker=broker)

                        trader.add_strategy(strategy)
                        strategy_executors = trader.run_all()

                    else:
                        from lumibot.backtesting import YahooDataBacktesting

                        # Backtest this strategy
                        backtesting_start = datetime(2023, 3, 3)
                        backtesting_end = datetime(2023, 3, 10)

                        results = LimitAndTrailingStop.backtest(
                            YahooDataBacktesting,
                            backtesting_start,
                            backtesting_end,
                            benchmark_asset="SPY",
                        )
                

        if AI_option_trading == 'Lumibots : Momentum Strategy':
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):

                """
                Strategy Description

                Buys the best performing asset from self.symbols over self.period number of days.
                For example, if SPY increased 2% yesterday, but VEU and AGG only increased 1% yesterday,
                then we will buy SPY.
                """


                class Momentum(Strategy):
                    # =====Overloading lifecycle methods=============

                    def initialize(self, symbols=None):
                        # Setting the waiting period (in days)
                        self.period = 2

                        # The counter for the number of days we have been holding the current asset
                        self.counter = 0

                        # There is only one trading operation per day
                        # No need to sleep between iterations
                        self.sleeptime = 0

                        # Set the symbols that we will be monitoring for momentum
                        if symbols:
                            self.symbols = symbols
                        else:
                            self.symbols = ["SPY", "VEU", "AGG"]

                        # The asset that we want to buy/currently own, and the quantity
                        self.asset = ""
                        self.quantity = 0

                    def on_trading_iteration(self):
                        # When the counter reaches the desired holding period,
                        # re-evaluate which asset we should be holding
                        momentums = []
                        if self.counter == self.period or self.counter == 0:
                            self.counter = 0
                            momentums = self.get_assets_momentums()

                            # Get the asset with the highest return in our period
                            # (aka the highest momentum)
                            momentums.sort(key=lambda x: x.get("return"))
                            best_asset_data = momentums[-1]
                            best_asset = best_asset_data["symbol"]
                            best_asset_return = best_asset_data["return"]

                            # Get the data for the currently held asset
                            if self.asset:
                                current_asset_data = [
                                    m for m in momentums if m["symbol"] == self.asset
                                ][0]
                                current_asset_return = current_asset_data["return"]

                                # If the returns are equals, keep the current asset
                                if current_asset_return >= best_asset_return:
                                    best_asset = self.asset
                                    best_asset_data = current_asset_data

                            self.log_message("%s best symbol." % best_asset)

                            # If the asset with the highest momentum has changed, buy the new asset
                            if best_asset != self.asset:
                                # Sell the current asset that we own
                                if self.asset:
                                    self.log_message("Swapping %s for %s." % (self.asset, best_asset))
                                    order = self.create_order(self.asset, self.quantity, "sell")
                                    self.submit_order(order)

                                # Calculate the quantity and send the buy order for the new asset
                                self.asset = best_asset
                                best_asset_price = best_asset_data["price"]
                                self.quantity = int(self.portfolio_value // best_asset_price)
                                order = self.create_order(self.asset, self.quantity, "buy")
                                self.submit_order(order)
                            else:
                                self.log_message("Keeping %d shares of %s" % (self.quantity, self.asset))

                        self.counter += 1

                        # Stop for the day, since we are looking at daily momentums
                        self.await_market_to_close()

                    def on_abrupt_closing(self):
                        # Sell all positions
                        self.sell_all()

                    def trace_stats(self, context, snapshot_before):
                        """
                        Add additional stats to the CSV logfile
                        """
                        # Get the values of all our variables from the last iteration
                        row = {
                            "old_best_asset": snapshot_before.get("asset"),
                            "old_asset_quantity": snapshot_before.get("quantity"),
                            "old_cash": snapshot_before.get("cash"),
                            "new_best_asset": self.asset,
                            "new_asset_quantity": self.quantity,
                        }

                        # Get the momentums of all the assets from the context of on_trading_iteration
                        # (notice that on_trading_iteration has a variable called momentums, this is what
                        # we are reading here)
                        momentums = context.get("momentums")
                        if len(momentums) != 0:
                            for item in momentums:
                                symbol = item.get("symbol")
                                for key in item:
                                    if key != "symbol":
                                        row[f"{symbol}_{key}"] = item[key]

                        # Add all of our values to the row in the CSV file. These automatically get
                        # added to portfolio_value, cash and return
                        return row

                    # =============Helper methods====================

                    def get_assets_momentums(self):
                        """
                        Gets the momentums (the percentage return) for all the assets we are tracking,
                        over the time period set in self.period
                        """

                        momentums = []
                        start_date = self.get_round_day(timeshift=self.period + 1)
                        end_date = self.get_round_day(timeshift=1)
                        data = self.get_bars(self.symbols, self.period + 2, timestep="day")
                        for asset, bars_set in data.items():
                            # Get the return for symbol over self.period days
                            # (from start_date to end_date)
                            symbol = asset.symbol
                            symbol_momentum = bars_set.get_momentum(start=start_date, end=end_date)
                            self.log_message(
                                "%s has a return value of %.2f%% over the last %d day(s)."
                                % (symbol, 100 * symbol_momentum, self.period)
                            )

                            momentums.append(
                                {
                                    "symbol": symbol,
                                    "price": bars_set.get_last_price(),
                                    "return": symbol_momentum,
                                }
                            )

                        return momentums


                if __name__ == "__main__":
                    is_live = False

                    if is_live:

                        from lumibot.brokers import Alpaca
                        from lumibot.traders import Trader

                        trader = Trader()

                        broker = Alpaca(ALPACA_CONFIG)

                        strategy = Momentum(broker=broker)

                        trader.add_strategy(strategy)
                        strategy_executors = trader.run_all()

                    else:
                        from lumibot.backtesting import YahooDataBacktesting

                        # Backtest this strategy
                        backtesting_start = datetime(2023, 1, 1)
                        backtesting_end = datetime(2023, 8, 1)

                        results = Momentum.backtest(
                            YahooDataBacktesting,
                            backtesting_start,
                            backtesting_end,
                            benchmark_asset="SPY",
                        )

        if AI_option_trading == 'Lumibots : Stock OCO Strategy':
            ticker = st.text_input("Please enter the ticker needed for investigation")
            if ticker:
                message = (f"Ticker captured : {ticker}")
                st.success(message)
            portfolio = st.number_input("Enter the portfolio size in USD")
            if portfolio:
                st.write(f"The portfolio size in USD Captured is : {portfolio}")
            min_date = datetime(1980, 1, 1)
            # Date input widget with custom minimum date
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:", min_value=min_date)
            with col2:
                end_date = st.date_input("End Date:")
            years = end_date.year - start_date.year
            st.success(f"years captured : {years}")
            if st.button("Trade"):

                """
                Strategy Description

                An example strategy for how to use OCO orders.
                """


                class StockOco(Strategy):
                    parameters = {
                        "buy_symbol": "SPY",
                        "take_profit_price": 405,
                        "stop_loss_price": 395,
                        "quantity": 10,
                    }

                    # =====Overloading lifecycle methods=============

                    def initialize(self):
                        # Set the initial variables or constants

                        # Built in Variables
                        self.sleeptime = "1D"

                        # Our Own Variables
                        self.counter = 0

                    def on_trading_iteration(self):
                        """Buys the self.buy_symbol once, then never again"""

                        buy_symbol = self.parameters["buy_symbol"]
                        take_profit_price = self.parameters["take_profit_price"]
                        stop_loss_price = self.parameters["stop_loss_price"]
                        quantity = self.parameters["quantity"]

                        # What to do each iteration
                        current_value = self.get_last_price(buy_symbol)
                        self.log_message(f"The value of {buy_symbol} is {current_value}")

                        if self.first_iteration:
                            # Market order
                                main_order = self.create_order(
                                    buy_symbol, quantity, "buy",
                                )
                                self.submit_order(main_order)

                                # OCO order
                                order = self.create_order(
                                    buy_symbol,
                                    quantity,
                                    "sell",
                                    take_profit_price=take_profit_price,
                                    stop_loss_price=stop_loss_price,
                                    type="oco",
                                )
                                self.submit_order(order)


                if __name__ == "__main__":
                    is_live = False

                    if is_live:

                        trader = Trader()

                        broker = Alpaca(ALPACA_CONFIG)

                        strategy = StockOco(broker=broker)

                        trader.add_strategy(strategy)
                        strategy_executors = trader.run_all()

                    else:
                        from lumibot.backtesting import YahooDataBacktesting

                        # Backtest this strategy
                        backtesting_start = datetime(2023, 3, 3)
                        backtesting_end = datetime(2023, 3, 10)

                        results = StockOco.backtest(
                            YahooDataBacktesting,
                            backtesting_start,
                            backtesting_end,
                            benchmark_asset="SPY",
                        )


if __name__ == "__main__":
    main()
