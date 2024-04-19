import streamlit as st
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import time
import sys
import os
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
import tickers as ti


st.title('Comprehensive Lite Algorithmic Trading Terminal : Identify, Visualize, Predict and Trade')
st.sidebar.info('Welcome to my Algorithmic Trading App Choose your options below')
st.success("This application features over 100 programmes for different roles")

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

def main():
    option = st.sidebar.selectbox('Make a choice', ['Find stocks','Stock Data', 'Stock Analysis','Technical Indicators', 'Stock Predictions', 'Portfolio Strategies'])

    if option == 'Find stocks':
        options = st.selectbox("Choose a stock finding method:", ["IDB_RS_Rating", "Correlated Stocks", "Finviz_growth_screener", "Fundamental_screener", "RSI_Stock_tickers", "Green_line Valuations", "Minervini_screener", "Pricing Alert Email", "Basic Sentiment", "Trading View Signals", "Twitter Screener", "Yahoo Recommendations"])
        if options == "IDB_RS_Rating":
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button('Start Analysis'):
                sp500_tickers = ti.tickers_sp500()
                sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]
                # Set S&P 500 index ticker
                sp500_index = '^GSPC'
                # Initialize list for storing relative stock returns
                relative_returns = []
                # Fetch and process S&P 500 historical data
                sp500_df = yf.download(sp500_tickers, start=start_date, end=end_date)
                st.write(sp500_df['Adj Close'])
                # Calculate the percentage change for each column
                percentage_change_df = sp500_df['Adj Close'].pct_change()
                # Concatenate the percentage change DataFrame with the original DataFrame
                sp500_df = pd.concat([sp500_df, percentage_change_df.add_suffix('_PercentChange')], axis=1)
                st.write(sp500_df)
                # Compute relative returns for each S&P 500 stock
                for ticker in sp500_tickers:
                    try:
                        # Download stock data
                        stock_df = yf.download(ticker, start=start_date, end=end_date)
                        percentage_stock_change_df = stock_df['Adj Close'].pct_change()
                        # Concatenate the percentage change DataFrame with the original DataFrame
                        stock_df = pd.concat([percentage_stock_change_df, percentage_change_df.add_suffix('_PercentChange')], axis=1)
                        st.write(stock_df)
                        '''
                        # Calculate cumulative return with added emphasis on recent quarter
                        stock_cumulative_return = (stock_df['Percent Change'].cumprod().iloc[-1] * 2 + 
                                                   stock_df['Percent Change'].cumprod().iloc[-63]) / 3

                        # Calculate relative return compared to S&P 500
                        #relative_return = round(stock_cumulative_return / sp500_cumulative_return, 2)
                        #relative_returns.append(relative_return)
                        message = (f'Ticker: {ticker}; Relative Return against S&P 500: {relative_returns}')
                        st.write(message)
                        time.sleep(1)  # Pause to prevent overloading server
                        '''
                    except Exception as e:
                        st.error(f'Error processing {ticker}: {e}')
                        '''
                    # Create dataframe with relative returns and RS ratings
                    rs_df = pd.DataFrame({'Ticker': sp500_tickers, 'Relative Return': relative_returns})
                    rs_df['RS_Rating'] = rs_df['Relative Return'].rank(pct=True) * 100
                    st.write(rs_df)
                    '''
                else:
                    st.sidebar.error('Error: End date must fall after start date')

    elif option == 'Recent Data':
        pass
    else:
        pass
if __name__ == "__main__":
    main()