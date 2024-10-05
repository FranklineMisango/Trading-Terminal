import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool


@tool

def tool_cc(symbol1:str, symbol2:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Correlation Coefficient (CC) of a stock along with the stock's closing price.'''
    start = start_date
    end = end_date

    # Read data
    df1 = yf.download(symbol1, start, end)
    df2 = yf.download(symbol2, start, end)

    # Calculate correlation coefficient
    cc = df1["Adj Close"].corr(df2["Adj Close"])

    # Plot correlation coefficient
    cc_chart = go.Scatter(x=df1.index, y=cc, mode='lines', name='Correlation Coefficient')

    layout = go.Layout(title=f'Correlation Coefficient between {symbol1} and {symbol2}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Correlation Coefficient'),
                    showlegend=True)

    fig = go.Figure(data=[cc_chart], layout=layout)
    st.plotly_chart(fig)


def norm_cc(symbol1, symbol2, start_date, end_date):
    start = start_date
    end = end_date

    # Read data
    df1 = yf.download(symbol1, start, end)
    df2 = yf.download(symbol2, start, end)

    # Calculate correlation coefficient
    cc = df1["Adj Close"].corr(df2["Adj Close"])

    # Plot correlation coefficient
    cc_chart = go.Scatter(x=df1.index, y=cc, mode='lines', name='Correlation Coefficient')

    layout = go.Layout(title=f'Correlation Coefficient between {symbol1} and {symbol2}',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Correlation Coefficient'),
                    showlegend=True)

    fig = go.Figure(data=[cc_chart], layout=layout)
    st.plotly_chart(fig)
