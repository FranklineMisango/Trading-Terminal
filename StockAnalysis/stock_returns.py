import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt
from langchain_core.tools import tool


@tool
def tool_view_stock_returns(symbol:str, num_years: int):
    '''This function fetches stock data and plots returns analysis.'''
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


def norm_view_stock_returns(symbol, num_years):
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