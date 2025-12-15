import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool
def tool_alculate_stock_profit_loss(symbol, start_date, end_date, initial_investment):
    '''This function calculates the profit or loss for a stock investment.'''
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

def norm_alculate_stock_profit_loss(symbol, start_date, end_date, initial_investment):
    '''This function calculates the profit or loss for a stock investment.'''
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