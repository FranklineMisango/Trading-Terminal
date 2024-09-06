import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
import datetime as dt
from langchain_core.tools import tool


@tool
def tool_risk_return_analysis(sectors:dict, selected_sector:str, start_date:dt.time, end_date:dt.time):
    '''This function calculates the risk and return metrics for a portfolio of stocks.'''
    # Downloading and processing stock data
    df = pd.DataFrame()
    for symbol in sectors[selected_sector]:
        df[symbol] = yf.download(symbol, start_date, end_date)['Adj Close']
    # Dropping rows with missing values
    df = df.dropna()
    # Calculating percentage change in stock prices
    rets = df.pct_change(periods=3)

    # Creating correlation matrix heatmap
    corr = rets.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.index,
        y=corr.columns,
        colorscale='Blues'
    ))
    fig.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis_title="Stock Symbols",
        yaxis_title="Stock Symbols"
    )
    st.plotly_chart(fig)

    # Plotting bar charts for risk and average returns
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rets.columns, y=rets.std(), name='Risk (Std. Dev.)', marker_color='red'))
    fig.add_trace(go.Bar(x=rets.columns, y=rets.mean(), name='Average Returns', marker_color='blue'))
    fig.update_layout(
        title="Risk and Average Returns",
        xaxis_title="Stock Symbols",
        yaxis_title="Value",
        barmode='group'
    )
    st.plotly_chart(fig)

    # Stacked bar chart for risk vs return
    fig = go.Figure()
    for i, symbol in enumerate(sectors[selected_sector]):
        fig.add_trace(go.Bar(x=[symbol], y=[rets.mean()[i]], name='Average of Returns', marker_color='blue'))
        fig.add_trace(go.Bar(x=[symbol], y=[rets.std()[i]], name='Risk of Returns', marker_color='red'))
    fig.update_layout(
        title='Risk vs Average Returns',
        xaxis_title='Stock Symbols',
        yaxis_title='Value',
        barmode='stack'
    )
    st.plotly_chart(fig)

    # Scatter plot for expected returns vs risk
    fig = go.Figure()
    for i in range(len(rets.columns)):
        fig.add_trace(go.Scatter(x=rets.mean(), y=rets.std(), mode='markers', text=rets.columns))
        fig.update_layout(
            title='Risk vs Expected Returns',
            xaxis_title='Expected Returns',
            yaxis_title='Risk'
        )
        st.plotly_chart(fig)

    # Display table with risk vs expected returns
    risk_returns_table = pd.DataFrame({'Risk': rets.std(), 'Expected Returns': rets.mean()})
    st.write("Table: Risk vs Expected Returns")
    st.write(risk_returns_table)


def norm_risk_return_analysis(sectors, selected_sector, start_date, end_date):
    df = pd.DataFrame()
    for symbol in sectors[selected_sector]:
        df[symbol] = yf.download(symbol, start_date, end_date)['Adj Close']
    # Dropping rows with missing values
    df = df.dropna()
    # Calculating percentage change in stock prices
    rets = df.pct_change(periods=3)

    # Creating correlation matrix heatmap
    corr = rets.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.index,
        y=corr.columns,
        colorscale='Blues'
    ))
    fig.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis_title="Stock Symbols",
        yaxis_title="Stock Symbols"
    )
    st.plotly_chart(fig)

    # Plotting bar charts for risk and average returns
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rets.columns, y=rets.std(), name='Risk (Std. Dev.)', marker_color='red'))
    fig.add_trace(go.Bar(x=rets.columns, y=rets.mean(), name='Average Returns', marker_color='blue'))
    fig.update_layout(
        title="Risk and Average Returns",
        xaxis_title="Stock Symbols",
        yaxis_title="Value",
        barmode='group'
    )
    st.plotly_chart(fig)

    # Stacked bar chart for risk vs return
    fig = go.Figure()
    for i, symbol in enumerate(sectors[selected_sector]):
        fig.add_trace(go.Bar(x=[symbol], y=[rets.mean()[i]], name='Average of Returns', marker_color='blue'))
        fig.add_trace(go.Bar(x=[symbol], y=[rets.std()[i]], name='Risk of Returns', marker_color='red'))
    fig.update_layout(
        title='Risk vs Average Returns',
        xaxis_title='Stock Symbols',
        yaxis_title='Value',
        barmode='stack'
    )
    st.plotly_chart(fig)

    # Scatter plot for expected returns vs risk
    fig = go.Figure()
    for i in range(len(rets.columns)):
        fig.add_trace(go.Scatter(x=rets.mean(), y=rets.std(), mode='markers', text=rets.columns))
        fig.update_layout(
            title='Risk vs Expected Returns',
            xaxis_title='Expected Returns',
            yaxis_title='Risk'
        )
        st.plotly_chart(fig)

    # Display table with risk vs expected returns
    risk_returns_table = pd.DataFrame({'Risk': rets.std(), 'Expected Returns': rets.mean()})
    st.write("Table: Risk vs Expected Returns")
    st.write(risk_returns_table)