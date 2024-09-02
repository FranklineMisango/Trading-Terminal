import yfinance as yf
import numpy as np
from langchain_core.tools import tool
import datetime as dt
import streamlit as  st
import yfinance as yf
import tickers as ti
import pandas as pd
import plotly.figure_factory as ff

@tool
def correlated_stocks(tool_start_date : dt.date, tool_end_date : dt.date):
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
    st.write("Before fetching tickers")
    tickers = sectors[selected_sector]
    st.write("After fetching tickers")
    data = yf.download(tickers, start=tool_start_date, end=tool_end_date)["Adj Close"]
    returns = np.log(data / data.shift(1))
    correlation = returns.corr()
    st.write(correlation)
    st.write("After computing correlation matrix")
    fig = ff.create_annotated_heatmap(
        z=correlation.values,
        x=correlation.columns.tolist(),
        y=correlation.index.tolist(),
        colorscale='Viridis',
        annotation_text=correlation.values.round(2),
        showscale=True
    )
    st.plotly_chart(fig)