#Full stack streamlit import
import streamlit as st

#Page config
st.set_page_config(page_title='Trading Terminal', page_icon="📈", layout="wide") 
from  project_about import about


#Migrate to submodules for readability
import yfinance as yf
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
import smtplib
import alpaca_trade_api as alpaca

#Env variables
from dotenv import load_dotenv
load_dotenv()


#secrets the application
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')
API_FMPCLOUD = os.environ.get("API_FMPCLOUD")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
BASE_URL = os.environ.get("BASE_URL")
API_KEY_ALPACA = os.environ.get("API_KEY_ALPACA")
SECRET_KEY_ALPACA =os.environ.get("SECRET_KEY_ALPACA")

KRAKEN_CONFIG =  "PASS"
ALPACA_CONFIG = alpaca.REST(API_KEY_ALPACA, SECRET_KEY_ALPACA, base_url= BASE_URL, api_version = 'v2')


#Al Trading recs
from lumibot.brokers import Alpaca
from lumibot.entities import Asset, Order
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from lumibot.backtesting import YahooDataBacktesting #TODO - Move to strategies
from lumibot.backtesting import CcxtBacktesting
from lumibot.example_strategies.crypto_important_functions import ImportantFunctions
from lumibot.entities import TradingFee



#Main tools for stock finding
from langchain_core.tools import tool
from StockFinder.analyze_idb_rs_rating import analyze_idb_rs_rating, tool_analyze_idb_rs_rating
from StockFinder.correlated_stocks import correlated_stocks, tool_correlated_stocks
from StockFinder.finviz_growth_screener import tool_growth_screener
from StockFinder.fundamental_screener import tool_fundamental_screener
from StockFinder.rsi_tickers import tool_rsi_tickers,normal_rsi_tickers
from StockFinder.green_line_valuation import tool_green_line_valuation
from StockFinder.Minervini_screener import tool_miniverini_screener, minervini_screener
from StockFinder.pricing_email_alerter import tool_price_alerter, normal_price_alerter
from StockFinder.trading_view_signals import tool_trading_view_signals, norm_trading_view_signals
from StockFinder.Twitter_Screener import tool_twitter_screener

#Main tools for stock predictions
from StockPredictions.time_series_arima import tool_arima_time_series, arima_time_series
from StockPredictions.stock_recommendations import tool_stock_recommendations_using_moving_averages, normal_stock_recommendations_using_moving_averages
from StockPredictions.stock_regression import tool_stock_regression, normal_stock_regression
from StockPredictions.stock_probabilty import tool_stock_probability, norm_stock_probability
from StockPredictions.stock_pca_analysis import tool_pca_analysis,norm_pca_analysis
from StockPredictions.Technical_indicators_clustering import tool_technical_indicators_clustering, norm_technical_indicators_clustering
from StockPredictions.lstm_predictions import tool_lstm_predictions, normal_lstm_predictions
from StockPredictions.etf_graphical_lasso import tool_etf_graphical_lasso,normal_etf_graphical_lasso
from StockPredictions.kmeans_clustering import tool_kmeans_clustering, normal_kmeans_clustering

#Main tools for stock data
from StockData.dividend_history import tool_dividend_history, norm_dividend_history
from StockData.Fibonacci_retracement import tool_fibonacci_retracement, norm_fibonacci_retracement
from StockData.finviz_scraper import tool_finviz_scraper
from StockData.finviz_insider_trading_scraper import tool_finviz_insider_scraper
from StockData.finviz_stock_data_scraper import tool_finviz_scraper_stock_data
from StockData.get_dividend_calendar import tool_get_dividend_calendar, norm_get_dividends
from StockData.greenline_test import tool_greenline_test, norm_greenline_test
from StockData.high_dividend_yield import tool_high_dividend_yield
from StockData.main_indicators import tool_main_indicators, norm_main_indicators
from StockData.pivots_calculator import tool_pivots_calculator, norm_pivots_calculator
from StockData.email_top_movers import tool_scrape_top_winners
from StockData.stock_vwap import tool_stock_vwap, norm_stock_vwap
from StockData.stock_sms import tool_stock_sms, norm_stock_sms
from StockData.stock_earnings import tool_stock_earnings, norm_stock_earnings
from StockData.trading_view_intraday import tool_trading_view_intraday
from StockData.trading_view_recommendations import tool_trading_view_recommendations, norm_trading_view_recommendations

#Main tools for stock Analysis
from StockAnalysis.backtest_all_indicators import tool_backtest_all_indicators, norm_backtest_all_indicators
from StockAnalysis.capm_analysis import tool_capm_analysis, norm_capm_analysis
from StockAnalysis.earnings_sentiment_analysis import tool_sentiment_analysis, norm_sentiment_analysis
from StockAnalysis.estimating_returns import tool_get_stock_returns, norm_get_stock_returns
from StockAnalysis.kelly_criterion import tool_kelly_criterion, norm_kelly_criterion
from StockAnalysis.intrinsic_analysis import tool_intrinsic_analysis, norm_intrinsic_analysis
from StockAnalysis.ma_backtesting import tool_ma_backtesting, norm_ma_backtesting
from StockAnalysis.ols_regression import tool_ols_regression, norm_ols_regression
from StockAnalysis.perfomance_risk_analysis import tool_perfomance_risk_analysis, norm_perfomance_risk_analysis
from StockAnalysis.risk_return_analysis import tool_risk_return_analysis, norm_risk_return_analysis
from StockAnalysis.seasonal_stock_analysis import tool_seasonal_stock_analysis
from StockAnalysis.sma_histogram import tool_sma_histogram, norm_sma_histogram
from StockAnalysis.sp500_cot_analysis import tool_sp_cot_analysis, norm_sp_cot_analysis
from StockAnalysis.sp500_valuation import tool_sp_500_valuation
from StockAnalysis.stock_pivot_resistance import tool_plot_stock_pivot_resistance, norm_plot_stock_pivot_resistance
from StockAnalysis.stock_profit_loss_analysis import tool_alculate_stock_profit_loss, norm_alculate_stock_profit_loss
from StockAnalysis.stock_return_stastical_analysis import tool_analyze_stock_returns, norm_analyze_stock_returns
from StockAnalysis.stock_returns import tool_view_stock_returns, norm_view_stock_returns
from StockAnalysis.var_analysis import tool_calculate_var, norm_calculate_var

#Main tools for Technical Indicators -> help
from TechnicalIndicators.exponential_weighted_moving_averages import tool_exponential_weighted_moving_averages, norm_exponential_weighted_moving_averages
from TechnicalIndicators.ema_volume import tool_ema_volume, norm_ema_volume
from TechnicalIndicators.ema import tool_ema, norm_ema
from TechnicalIndicators.gann_lines_angles import tool_gann_lines_angles, norm_gann_lines_tracker
from TechnicalIndicators.gmma import tool_gmma, norm_gmma
from TechnicalIndicators.macd import tool_macd, norm_macd
from TechnicalIndicators.mfi import tool_mfi, norm_mfi
from TechnicalIndicators.ma_high_low import tool_ma_high_low, norm_ma_high_low
from TechnicalIndicators.pvi import tool_pvi, norm_pvi
from TechnicalIndicators.pvt import tool_pvt, norm_pvt
from TechnicalIndicators.roc import tool_roc,norm_roc
from TechnicalIndicators.roi import tool_roi, norm_roi
from TechnicalIndicators.rsi import tool_rsi, norm_rsi
from TechnicalIndicators.rsi_bollinger_bands import tool_rsi_bollinger_bands, norm_rsi_bollinger_bands
from TechnicalIndicators.vwap import tool_vwap, norm_vwap
from TechnicalIndicators.wma import tool_wma, norm_wma
from TechnicalIndicators.wsma import tool_wsma, norm_wsma
from TechnicalIndicators.z_score_indicator import tool_z_score, norm_z_score
from TechnicalIndicators.acceleration_bands import tool_accleration_bands, norm_accleration_bands
from TechnicalIndicators.adl import tool_adl, norm_adl
from TechnicalIndicators.aroon import tool_aroon, norm_aroon
from TechnicalIndicators.aroon_oscillator import tool_aroon_oscillator, norm_aroon_oscillator
from TechnicalIndicators.adx import tool_adx, norm_adx
from TechnicalIndicators.atr import tool_atr, norm_atr
from TechnicalIndicators.bp import tool_bp, norm_bp
from TechnicalIndicators.bi import tool_bi, norm_bi
from TechnicalIndicators.bb import tool_bb, norm_bb
from TechnicalIndicators.bbw import tool_bbw, norm_bbw
from TechnicalIndicators.bri import tool_bri, norm_bri
from TechnicalIndicators.car import tool_car, norm_car
from TechnicalIndicators.cpr import tool_cpr, norm_cpr
from TechnicalIndicators.cmf import tool_cmf, norm_cmf
from TechnicalIndicators.co import tool_co, norm_co
from TechnicalIndicators.cci import tool_cci, norm_cci
from TechnicalIndicators.cc import tool_cc, norm_cc
from TechnicalIndicators.cov import tool_cov, norm_cov
from TechnicalIndicators.dpo import tool_dpo, norm_dpo
from TechnicalIndicators.dc import tool_dc, norm_dc
from TechnicalIndicators.dema import tool_dema, norm_dema
from TechnicalIndicators.dmi import tool_dmi, norm_dmi
from TechnicalIndicators.evm import tool_evm, norm_evm
from TechnicalIndicators.fi import tool_fi, norm_fi
from TechnicalIndicators.gri import tool_gri, norm_gri
from TechnicalIndicators.gdc import tool_gdc, norm_gdc
from TechnicalIndicators.hml import tool_hml, norm_hml
from TechnicalIndicators.hma import tool_hma, norm_hma
from TechnicalIndicators.kc import tool_kc, norm_kc
from TechnicalIndicators.lr import tool_lr, norm_lr
from TechnicalIndicators.lrs import tool_lrs, norm_lrs
from TechnicalIndicators.lwma import tool_lwma, norm_lwma
from TechnicalIndicators.mo import tool_mo, norm_mo
from TechnicalIndicators.m import tool_m, norm_m
from TechnicalIndicators.mae import tool_mae, norm_mae
from TechnicalIndicators.mahl import tool_mahl, norm_mahl
from TechnicalIndicators.mar import tool_mar, norm_mar
from TechnicalIndicators.mma import tool_mma, norm_mma
from TechnicalIndicators.mlr import tool_mlr, norm_mlr
from TechnicalIndicators.nhnl import tool_nhnl, norm_nhnl
from TechnicalIndicators.pp import tool_pp, norm_pp
from TechnicalIndicators.pc import tool_pc, norm_pc
from TechnicalIndicators.pr import tool_pr, norm_pr
from TechnicalIndicators.rv import tool_rv, norm_rv
from TechnicalIndicators.rvi import tool_rvi, norm_rvi
from TechnicalIndicators.sma import tool_sma, norm_sma
from TechnicalIndicators.srl import tool_srl, norm_srl
from TechnicalIndicators.sdv import tool_sdv, norm_sdv
from TechnicalIndicators.srsi import tool_srsi, norm_srsi
from TechnicalIndicators.sf import tool_sf, norm_sf
from TechnicalIndicators.sfu import tool_sfu, norm_sfu
from TechnicalIndicators.ss import tool_ss, norm_ss
from TechnicalIndicators.st import tool_st, norm_st
from TechnicalIndicators.tsi import tool_tsi, norm_tsi
from TechnicalIndicators.uo import tool_uo, norm_uo
from TechnicalIndicators.vi import tool_vi, norm_vi
from TechnicalIndicators.vpci import tool_vpci, norm_vpci
from TechnicalIndicators.vwma import tool_vwma, norm_vwma

# Main tools for Algorithmic trading

tools = [tool_analyze_idb_rs_rating,tool_correlated_stocks, tool_growth_screener, 
         tool_fundamental_screener, tool_rsi_tickers,tool_green_line_valuation, 
         tool_miniverini_screener, tool_price_alerter, tool_trading_view_signals,
         tool_twitter_screener, tool_arima_time_series,
         tool_stock_recommendations_using_moving_averages,tool_stock_regression,
         tool_stock_probability, tool_pca_analysis, tool_technical_indicators_clustering,
         tool_lstm_predictions, tool_etf_graphical_lasso,tool_kmeans_clustering,
         tool_dividend_history, tool_fibonacci_retracement, tool_finviz_scraper, tool_finviz_insider_scraper,
         tool_finviz_scraper_stock_data, tool_get_dividend_calendar, tool_greenline_test,
         tool_high_dividend_yield,tool_pivots_calculator,tool_main_indicators,
         tool_scrape_top_winners, tool_stock_vwap, tool_stock_sms,
         tool_stock_earnings, tool_trading_view_intraday,tool_trading_view_recommendations,
         tool_backtest_all_indicators, tool_capm_analysis, tool_sentiment_analysis,
         tool_get_stock_returns,tool_kelly_criterion,tool_intrinsic_analysis,
         tool_ma_backtesting, tool_ols_regression, tool_perfomance_risk_analysis,
         tool_risk_return_analysis, tool_seasonal_stock_analysis,tool_sma_histogram,
         tool_sp_cot_analysis,tool_sp_500_valuation,tool_plot_stock_pivot_resistance,
         tool_alculate_stock_profit_loss,tool_alculate_stock_profit_loss,tool_view_stock_returns,
         tool_calculate_var,tool_analyze_stock_returns, tool_exponential_weighted_moving_averages,
         tool_ema,tool_ema_volume, tool_gann_lines_angles, tool_gmma,
         tool_macd,tool_mfi,tool_ma_high_low,tool_pvi, tool_pvt, tool_roc,tool_roi, tool_rsi,
         tool_rsi_bollinger_bands,tool_vwap,tool_wma,tool_wsma,tool_z_score, tool_accleration_bands,
         tool_adl, tool_aroon, tool_adx, tool_atr, tool_bp, tool_bi, tool_bb, tool_bbw, tool_bri, tool_car,
         tool_cpr, tool_cmf, tool_co, tool_cci, tool_cc, tool_cov, tool_dpo, tool_dc, tool_dema, tool_dmi, tool_evm,
         tool_fi, tool_gri, tool_gdc, tool_hml, tool_hma, tool_kc, tool_lr, tool_lrs, tool_lwma, tool_mo, tool_m, tool_mae,
         tool_mahl, tool_mar, tool_mma, tool_mlr, tool_nhnl, tool_pp, tool_pc, tool_pr, tool_rv, tool_rvi, tool_sma, tool_srl,
         tool_sdv, tool_srsi, tool_sf, tool_sfu, tool_ss, tool_st, tool_tsi, tool_uo, tool_vi, tool_vpci, tool_vwma,
]
   



# Multimodial agent bot configuration
from openai import OpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
import getpass
from langchain_openai import ChatOpenAI
# Set your API key here
api_key = os.environ.get("LANGSMITH_KEY")
if 'conversation' not in st.session_state:
    st.session_state.conversation = ""



##Terminal config
import time 

def main():
    #Main app
    st.title("📈 Frankline and Co. LP Trading Terminal Beta")
    st.sidebar.info('Welcome to my Algorithmic Trading App. Choose your options below. This application is backed over by 100 mathematically powered algorithms handpicked from the internet and modified for different Trading roles')
    # Create a two-column layout
    left_column, right_column = st.columns([2, 2])

    with left_column:
        option = st.sidebar.radio("Make a choice",('About','Find stocks', 'Portfolio Strategies','Stock Data', 'Stock Analysis','Technical Indicators', 'Stock Predictions'))
        if option =='About':
            about()
        if option == 'Find stocks':
            options = st.selectbox("Choose a stock finding method:", ["IDB_RS_Rating", "Correlated Stocks", "Finviz_growth_screener", "Fundamental_screener", "RSI_Stock_tickers", "Green_line Valuations", "Minervini_screener", "Pricing Alert Email", "Trading View Signals", "Twitter Screener", "Yahoo Recommendations"])
           
            if options == "IDB_RS_Rating":
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:")
                with col2:
                    end_date = st.date_input("End Date:")
                if st.button('Start Analysis'):
                    analyze_idb_rs_rating(start_date, end_date)

            if options == "Correlated Stocks":
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:")
                with col2:
                    end_date = st.date_input("End Date:")                
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
                    correlated_stocks(start_date, end_date, selected_sector)
            
            if options == "Finviz_growth_screener":
                if st.button("Scan"):
                    # Execute the screener and store the result
                    tool_input = {}  
                    df = tool_growth_screener(tool_input)
                    st.write('\nGrowth Stocks Screener:')
                    st.write(df)
                    # Extract and print list of tickers
                    tickers = df.index
                    st.write('\nList of Tickers:')
                    st.write(tickers)
                
            if options == "Fundamental_screener":
                st.success("This portion allows you to sp500 for base overview")
                if st.button("Scan"):
                    fund_tool_input = {} 
                    tool_fundamental_screener(fund_tool_input) 

            if options == "RSI_Stock_tickers":
                st.success("This program allows you to view which tickers are overbrought and which ones are over sold")
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:")
                with col2:
                    end_date = st.date_input("End Date:")
                # Get dates for the past year

                if st.button("Check"):
                    normal_rsi_tickers(start_date, end_date)
                   
                    
            if options=="Green_line Valuations":
                st.success("This programme analyzes all tickers to help identify the Green Value ones")
                if st.button("Scan"):
                    # Execute the screener and store the result
                    green_line_tool_input = {}  # Replace with actual input required by the tool
                    tool_fundamental_screener(green_line_tool_input)

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
                    minervini_screener(start_date, end_date)

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

                if st.button("Check"):
                    normal_price_alerter(start_date, end_date, stock, email_address, target_price)

            if options=="Trading View Signals":
                st.success("This program allows you to view the Trading view signals of a particular ticker")
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:")
                with col2:
                    end_date = st.date_input("End Date:")
                
                if st.button("Simulate signals"):
                    norm_trading_view_signals(start_date, end_date)

            if options == "Twitter Screener":
                st.write("This segment allows you to screen for stocks based on Twitter sentiment")
                if st.button("Analyze"):
                    twitter_tool = {}  # Replace with actual input required by the tool
                    tool_twitter_screener(twitter_tool)
                    
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
                   arima_time_series(start_date, end_date, ticker)

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
                    normal_stock_recommendations_using_moving_averages(ticker, start_date, end_date)

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
                    normal_stock_regression(start_date, end_date, ticker)

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
                    norm_stock_probability(start_date, end_date, ticker)
                    
            if pred_option == "sp500 PCA Analysis":
                st.write("This segment analyzes the s&p 500 stocks and identifies those with high/low PCA weights")
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:")
                with col2:
                    end_date = st.date_input("End Date:")
                ticker = st.text_input("Enter the ticker you want to monitor")
                if ticker:
                    message = (f"Ticker captured : {ticker}")
                    st.success(message)

                if st.button("Check"):
                    norm_pca_analysis(start_date, end_date, ticker)
                               
            if pred_option == "Technical Indicators Clustering":
                #Fix the segmentation of all tickers on the graph
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:")
                with col2:
                    end_date = st.date_input("End Date:")
                
                ticker = st.text_input("Enter the ticker you want to monitor")
                if ticker:
                    message = (f"Ticker captured : {ticker}")
                    st.success(message)

                if st.button("Check"):
                    norm_technical_indicators_clustering(start_date, end_date, ticker)

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
                    normal_lstm_predictions(start_date, end_date, ticker)

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
                    normal_etf_graphical_lasso(start_date, end_date,number_of_years)
                    
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
                    normal_kmeans_clustering(start_date, end_date)

        elif option =='Stock Data':
            pred_option_data = st.selectbox("Choose a Data finding method:", ["High Dividend yield","Dividend History", "Fibonacci Retracement", 
                                                                            "Finviz Home scraper", "Finviz Insider Trading scraper", "Finviz stock scraper", 
                                                                            "Get Dividend calendar", "Green Line Test", "Main Indicators", "Pivots Calculator", 
                                                                            "Email Top Movers", "Stock VWAP", "Data SMS stock", "Stock Earnings", "Trading view Intraday", 
                                                                            "Trading view Recommendations", "Yahoo Finance Intraday Data"])
        
            if pred_option_data == "Dividend History": 
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
                    norm_dividend_history(start_date, end_date, ticker)
            
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
                    norm_fibonacci_retracement(ticker, start_date, end_date)


            if pred_option_data == "Finviz Home scraper":
                st.success("This segment allows you to find gainers/losers today and relevant news from Finviz")
                if st.button("Confirm"):
                    finviz_data_tool = {}  # Replace with actual input required by the tool
                    tool_finviz_scraper(finviz_data_tool)
                    

            if pred_option_data == "Finviz Insider Trading scraper":
                st.success("This segment allows you to check the latest insider trades of the market and relevant parties")
                # Set display options for pandas
                if st.button("Check"):
                    insider_trading_tool = {}  # Replace with actual input required by the tool
                    tool_finviz_insider_scraper(insider_trading_tool)

            if pred_option_data == "Finviz stock scraper":

                st.success("This segment allows to analyze overview a ticker from finviz")
                stock = st.text_input("Please type the stock ticker")
                if stock:
                    st.success(f"Stock ticker captured : {stock}")
                if st.button("check"):
                    tool_finviz_scraper_stock_data(stock)
            
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
                    norm_get_dividends(year, month)
                
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
                    norm_greenline_test(start_date , end_date , ticker)
                 


            if pred_option_data == "High Dividend yield":
                st.success("This program allows you to simulate the Dividends given by a company")
                if st.button("Check"):
                    high_dividend_yield_tooler = {}  # Replace with actual input required by the tool
                    tool_high_dividend_yield(high_dividend_yield_tooler)



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
                    # Convert dates to strings
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    norm_main_indicators(ticker, start_date_str, end_date_str)

            if pred_option_data == "Pivots Calculator":
                st.write("This part calculates the pivot shifts within one day")
                # Prompt user for stock ticker input
                stock = st.text_input('Enter a ticker: ')
                interval = st.text_input('Enter an Interval : default for 1d')
                
                if st.button("Check") :
                    # Fetch historical data for the specified stock using yfinance
                    norm_pivots_calculator(stock, interval)
                                
            if pred_option_data == "Email Top Movers":

                st.success("This segment allows you to get updates on Top stock movers")
                email_address = st.text_input("Enter your Email address")
                if email_address:
                    message_three = (f"Email address captured is {email_address}")
                    st.success(message_three)

                if st.button("Check"):
                    # Function to scrape top winner stocks from Yahoo Finance
                    email_top_movers_tool = {}  # Replace with actual input required by the tool
                    tool_scrape_top_winners(email_top_movers_tool)

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
                    norm_stock_vwap(ticker, start_date, end_date)

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
                   norm_stock_sms(ticker, start_date, end_date, email_address)


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
                    norm_stock_earnings(start_date, end_date)


            if pred_option_data == "Trading view Intraday":
                if st.button("Check"):
                    trading_intraday_test = {}  # Replace with actual input required by the tool
                    tool_trading_view_intraday(trading_intraday_test)
                    
            if pred_option_data == "Trading view Recommendations":
                
                st.success("This segment allows us to get recommendations fro Trading View")
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
                    norm_trading_view_recommendations(ticker, start_date, end_date)




                    
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
                    norm_backtest_all_indicators(ticker, start_date, end_date)

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
                    norm_capm_analysis(ticker, start_date, end_date)


            if pred_option_analysis == "Earnings Sentiment Analysis":

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
                    norm_sentiment_analysis(ticker, start_date, end_date)

            if pred_option_analysis == "Estimating Returns":

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
                   norm_get_stock_returns(ticker, start_date, end_date)
                    
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
                    norm_kelly_criterion(ticker, start_date, end_date)

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
                    norm_intrinsic_analysis(ticker, start_date, end_date)

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
                    norm_ma_backtesting(ticker, start_date, end_date)

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
                    norm_ols_regression(ticker, start_date, end_date)

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
                    norm_perfomance_risk_analysis(ticker, start_date, end_date)
                
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
                    norm_risk_return_analysis(sectors, selected_sector, start_date, end_date)

            if pred_option_analysis == "Seasonal Stock Analysis":
                st.write("This segment allows us to analyze the seasonal returns of s&p 500")
                if st.button("Analyze"):
                    # Scrape a list of the S&P 500 components using your ticker function
                    seasonal_stock_analysis_tool = {}  # Replace with actual input required by the tool
                    tool_seasonal_stock_analysis(seasonal_stock_analysis_tool)
                   
        
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
                   tool_sma_histogram(start_date, end_date, ticker)

            if pred_option_analysis == "SP500 COT Sentiment Analysis":
                st.success("This segment allows us to look at COT Sentiment analysis of tickers for the past 1 or more years")                
                col1, col2 = st.columns([2, 2])
                with col1:
                    start_date = st.date_input("Start date:")
                with col2:
                    end_date = st.date_input("End Date:")
                if st.button("Check"):
                    tool_sp_cot_analysis(start_date,end_date)

            if pred_option_analysis == "SP500 Valuation":
                if st.button("Check"):
                    # Load the S&P 500 data
                    #sp_df = pd.read_excel("http://www.stern.nyu.edu/~adamodar/pc/datasets/spearn.xls", sheet_name="Sheet1")
                    sp_500_valuation_tool = {}  # Replace with actual input required by the tool
                    tool_sp_500_valuation(sp_500_valuation_tool)

            if pred_option_analysis == "Stock Pivot Resistance":
            
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
                    norm_plot_stock_pivot_resistance(ticker, start_date, end_date)


            if pred_option_analysis == "Stock Profit/Loss Analysis":
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
                    norm_alculate_stock_profit_loss(ticker, start_date, end_date, portfolio_one)
                
            if pred_option_analysis == "Stock Return Statistical Analysis":
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
                    norm_analyze_stock_returns(ticker, start_date, end_date)

            if pred_option_analysis == "VAR Analysis":
               

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
                    norm_calculate_var(ticker, start_date, end_date)


            if pred_option_analysis == "Stock Returns":
                

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
                    norm_view_stock_returns(ticker, years)
        


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
                           norm_ema(ticker, start_date, end_date)

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
                        norm_ema_volume(start_date, end_date, ticker)

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
                       norm_exponential_weighted_moving_averages(ticker, start_date, end_date) 

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
                    norm_gann_lines_tracker(ticker, start_date, end_date)

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
                    norm_gmma(ticker, start_date, end_date)

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
                    norm_macd(start_date, end_date, ticker)

                
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
                    norm_mfi(ticker, start_date, end_date)


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
                   norm_ma_high_low(ticker, start_date, end_date)



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
                   norm_pvi(ticker, start_date, end_date)

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
                    pass


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
                    norm_roc(ticker, start_date, end_date)

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
                    norm_roi(ticker, start_date, end_date)
                    
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
                    norm_rsi(ticker, start_date, end_date)


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
                    norm_rsi_bollinger_bands(ticker, start_date, end_date)


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
                    norm_vwap(ticker, start_date, end_date)

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
                    norm_wma(ticker, start_date, end_date)

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
                    norm_wsma(ticker, start_date, end_date)


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
                    norm_z_score(ticker, start_date, end_date)

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
                    norm_apo(ticker, start_date, end_date)

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
                    norm_acceleration_bands(ticker, start_date, end_date)


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
                    norm_adl(ticker, start_date, end_date)
                
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
                    norm_aroon(ticker, start_date, end_date)


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
                    tool_aroon_oscillator(ticker, start_date, end_date)

                   
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
                    norm_adx(ticker, start_date, end_date)

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
                    norm_atr(ticker, start_date, end_date)   
                

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
                    norm_bp(ticker, start_date, end_date)   
                    

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
                    norm_bi(ticker, start_date, end_date)

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
                    norm_bb(ticker, start_date, end_date)    

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
                    norm_bbw(ticker, start_date, end_date)


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
                    norm_bri(ticker, start_date, end_date)


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
                    norm_car(ticker, start_date, end_date)

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
                    norm_cpr(ticker, start_date, end_date)   

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
                   norm_cmf(ticker, start_date, end_date) 

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
                  norm_co(ticker, start_date, end_date)  

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
                    norm_cci(ticker, start_date, end_date)

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
                  norm_cc(symbol1, symbol2, start_date, end_date) 

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
                    norm_cov(symbol1, symbol2, start_date, end_date)   

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
                    norm_dpo(ticker, start_date, end_date)

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
                    norm_dc(ticker, start_date, end_date)

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
                    norm_dema(ticker, start_date, end_date)

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
                   norm_dmi(ticker, start_date, end_date)

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
                   norm_evm(ticker, start_date, end_date) 

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
                    norm_fi(ticker, start_date, end_date) 

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
                    norm_gri(ticker, start_date, end_date) 

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
                    norm_gdc(ticker, start_date, end_date)    

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
                    norm_hml(ticker, start_date, end_date)    

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
                    norm_hma(ticker, start_date, end_date)   

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
                    norm_kc(ticker, start_date, end_date) 

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
                    norm_lr(ticker, start_date, end_date)

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
                    norm_lrs(ticker, start_date, end_date)

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
                    norm_lwma(ticker, start_date, end_date)

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
                    norm_mo(ticker, start_date, end_date)

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
                    norm_m(ticker, start_date, end_date)

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
                   norm_mae(ticker, start_date, end_date)

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
                    norm_mahl(ticker, start_date, end_date)

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
                    norm_mar(ticker, start_date, end_date)  

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
                    norm_mma(ticker, start_date, end_date)   

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
                    norm_mlr(ticker, start_date, end_date)   

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
                    norm_nhnl(ticker, start_date, end_date)   


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
                    norm_pp(ticker, start_date, end_date)  

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
                    norm_pc(ticker, start_date, end_date)  


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
                     norm_pr(ticker, start_date, end_date)


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
                    norm_rv(ticker, start_date, end_date)

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
                    norm_rvi(ticker, start_date, end_date)  


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
                    norm_sma(ticker, start_date, end_date)  

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
                    norm_srl(ticker, start_date, end_date)

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
                   norm_sdv(ticker, start_date, end_date) 

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
                    norm_srsi(ticker, start_date, end_date) 


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
                    norm_sf(ticker, start_date, end_date) 

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
                    norm_sfu(ticker, start_date, end_date) 


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
                   norm_ss(ticker, start_date, end_date) 
            
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
                    norm_st(ticker, start_date, end_date)

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
                    norm_tsi(ticker, start_date, end_date)

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
                    norm_uo(ticker, start_date, end_date)

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
                    norm_vi(ticker, start_date, end_date)

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
                    norm_vpci(ticker, start_date, end_date)


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
                     norm_vwma(ticker, start_date, end_date)

        with right_column:
            # Now you can use the key in your code
            prompt = hub.pull("hwchase17/openai-tools-agent", api_key=os.environ.get("LANGSMITH_KEY"))
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=os.environ.get("OPEN_AI_API"))
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            st.markdown("### Frankline & Co. LP Bot 🤖")
            st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

            # Create a text input box for the user to ask a question
            user_query = st.text_input('Ask me to run any any finance tool..like `run idb irs rating from august 1 to sep 1`')

            # Check if user submitted input
            if st.button('Submit'):
                if user_query.strip() == '':
                    st.error("Please enter a query or function to run")
                else:
                    try:
                        st.success("Query running...wait for output")
                        # Use the agent_executor to handle the user query
                        result = agent_executor.invoke({"input": user_query})
                        if result:
                            st.success("Result returned")
                            # Update the conversation history
                            st.session_state.conversation += f"User: {user_query}\nBot: {result}\n"
                            # Display the updated conversation history
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()