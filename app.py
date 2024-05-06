#Library imports
import streamlit as st
from bs4 import BeautifulSoup
import os
from math import sqrt
from pylab import rcParams
import pylab as pl
import calendar
import yfinance as yf
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import tickers as ti
import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
import requests
from pandas_datareader import data as pdr
import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
import ta_functions as ta
import tickers as ti
from tickers import tickers_sp500
import time
import smtplib
from email.message import EmailMessage
import datetime as dt
import time
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime  
import math
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
from config import EMAIL_ADDRESS, EMAIL_PASSWORD , API_FMPCLOUD
warnings.filterwarnings("ignore")
from autoscraper import AutoScraper
from lxml import html
import requests
from datetime import datetime, timedelta
#Page config
st.set_page_config(layout="wide")
st.title('Frankline & Associates LLP. Comprehensive Lite Algorithmic Trading Terminal')
st.success('Identify, Visualize, Predict and Trade')
st.sidebar.info('Welcome to my Algorithmic Trading App Choose your options below')
st.sidebar.info("This application features over 100 programmes for different roles")

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

def main():
    option = st.sidebar.selectbox('Make a choice', ['Find stocks','Stock Data', 'Stock Analysis','Technical Indicators', 'Stock Predictions', 'Portfolio Strategies', "AI Trading"])
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

                # Plot the data
                train = data[:train_data_len]
                valid = data[train_data_len:].assign(Predictions=predictions)
                plt.figure(figsize=(16,8))
                plt.title(f"{stock.upper()} Close Price")
                plt.xlabel('Date', fontsize=16)
                plt.ylabel('Close Price (USD)', fontsize=16)
                plt.plot(train['Close'])
                plt.plot(valid[['Close', 'Predictions']])
                plt.legend(['Train', 'Valid', 'Prediction'], loc='lower right')
                plt.show()

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
                                                                          "Get Dividend calendar", "Green Line Test"])
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

        if pred_option_data == "Get Divident calendar":
            pass
        if pred_option_data == "Get Divident calendar":
            pass
        if pred_option_data == "Get Divident calendar":
            pass
        if pred_option_data == "Get Divident calendar":
            pass

        
    elif option =='Stock Analysis':
        pass
    elif option =='Technical Indicators':
        pass
    elif option =='Stock Predictions':
        pass
    elif option =='Portfolio Strategies':
        pass
    elif option == "AI Trading":
        st.write("This bot allows you to initate a trade")
        pass

if __name__ == "__main__":
    main()
