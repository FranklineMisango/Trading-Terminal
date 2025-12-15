import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import tickers as ti
from langchain_core.tools import tool
import datetime as dt

@tool
def tool_sp_500_valuation():
    '''This function calculates the valuation of the S&P 500 based on different earnings scenarios.'''
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
