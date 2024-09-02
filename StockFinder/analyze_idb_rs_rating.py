
from langchain_core.tools import tool
import datetime as dt
import streamlit as  st
import yfinance as yf
import tickers as ti
import pandas as pd


@tool
def analyze_idb_rs_rating(tool_start_date : dt.date, tool_end_date : dt.date):
    '''This tool allows you to analyze the IDB RS Rating of the S&P 500 stocks'''
    sp500_tickers = ti.tickers_sp500()
    sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]
    sp500_df = yf.download(sp500_tickers, start=tool_start_date, end=tool_end_date)
    percentage_change_df = sp500_df['Adj Close'].pct_change()
    sp500_df = pd.concat([sp500_df, percentage_change_df.add_suffix('_PercentChange')], axis=1)
    st.write(sp500_df)