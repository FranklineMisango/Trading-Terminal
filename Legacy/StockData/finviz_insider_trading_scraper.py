import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
from langchain_core.tools import tool

@tool
# Set display options for pandas
def tool_finviz_insider_scraper():
    '''This tool allows you to scrape insider trades data from Finviz'''
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
