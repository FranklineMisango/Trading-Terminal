#Library imports
import streamlit as st
from bs4 import BeautifulSoup
import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import tickers as ti
import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
import requests
from config import financial_model_prep
from pandas_datareader import data as pdr
import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
import ta_functions as ta
import tickers as ti
import time
import smtplib
from email.message import EmailMessage
import datetime as dt
from pandas_datareader import data as pdr
import time
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

st.title('Comprehensive Lite Algorithmic Trading Terminal : Identify, Visualize, Predict and Trade')
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
    option = st.sidebar.selectbox('Make a choice', ['Find stocks','Stock Data', 'Stock Analysis','Technical Indicators', 'Stock Predictions', 'Portfolio Strategies'])
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

            # Get the API key
            demo = financial_model_prep()

            # Define search criteria for the stock screener
            marketcap = str(1000000000)
            url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan={marketcap}&betaMoreThan=1&volumeMoreThan=10000&sector=Technology&exchange=NASDAQ&dividendMoreThan=0&limit=1000&apikey={demo}'

            # Fetch list of companies meeting criteria
            screener = requests.get(url).json()

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
                    print(f"Error processing {company}: {e}")

            # Convert to DataFrame and display
            df = pd.DataFrame.from_dict(value_ratios, orient='index')
            print(df.head())

            # Define and apply ranking criteria
            criteria = { 'ROE': 1.2, 'ROA': 1.1, 'Debt_Ratio': -1.1, # etc.
                        'Revenue_Growth': 1.25, 'NetIncome_Growth': 1.10 }

            # Normalize and rank companies
            mean_values = df.mean()
            normalized_df = df / mean_values
            normalized_df['ranking'] = sum(normalized_df[col] * weight for col, weight in criteria.items())

            # Print ranked companies
            print(normalized_df.sort_values(by=['ranking'], ascending=False))

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
            stock = st.chat_input("Enter the ticker you want to monitor")
            if stock:
                message = (f"Ticker captured : {stock}")
                st.success(message)
            
            stock = "NVDA"

            from config import EMAIL_ADDRESS
            from config import EMAIL_PASSWORD

            #Uncomment to make the segments better
            '''
            EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')
            EMAIL_ADDRESS = os.environ.get('EMAIL_USER')

            '''
            # Stock and target price settings
            target_price = st.number_input("Enter the target price")

            # Email setup
            msg = EmailMessage()
            msg['Subject'] = f'Alert on {stock}!'
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = 'franklinemisango@gmail.con'  # Set the recipient email address (this is mine for testing )

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
                    message = f"{stock} has reached the alert price of {target_price}\nCurrent Price: {current_close}"
                    print(message)
                    msg.set_content(message)

                    # Send email
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                        smtp.send_message(msg)
                        print("Email sent successfully.")
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
                        recommendation = driver.find_element(By.CLASS_NAME, "speedometerSignal-pyzN--tL").text
                        counters = [element.text for element in driver.find_elements(By.CLASS_NAME, "counterNumber-3l14ys0C")]
                        # Append data to lists
                        signals.append(recommendation)
                        sells.append(float(counters[0]))
                        neutrals.append(float(counters[1]))
                        buys.append(float(counters[2]))
                        price = pdr.get_data_yahoo(ticker)['Adj Close'][-1]
                        prices.append(price)
                        valid_tickers.append(ticker)

                        st.write(f"{ticker} recommendation: {recommendation}")

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
    
                
    elif option == 'Recent Data':
        pass
    else:
        pass

if __name__ == "__main__":
    main()
