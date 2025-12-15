import pandas as pd
import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool




@tool
# Set the start and end dates for historical data retrieval
def tool_greenline_test(start_date : dt.date, end_date : dt.date, ticker: str):
    '''This tool allows you to test the green line strategy for a stock'''
    start = dt.datetime.combine(start_date, dt.datetime.min.time())
    end = dt.datetime.combine(end_date, dt.datetime.min.time())

    df = yf.download(ticker, start, end)
    price = df['Adj Close'][-1]

    # Filter out days with very low trading volume
    df.drop(df[df["Volume"] < 1000].index, inplace=True)

    # Get the monthly maximum of the 'High' column
    df_month = df.groupby(pd.Grouper(freq="M"))["High"].max()

    # Initialize variables for the green line analysis
    glDate, lastGLV, currentDate, currentGLV, counter = 0, 0, "", 0, 0

    # Loop through monthly highs to determine the most recent green line value
    for index, value in df_month.items():
        if value > currentGLV:
            currentGLV = value
            currentDate = index
            counter = 0
        if value < currentGLV:
            counter += 1
            if counter == 3 and ((index.month != end_date.month) or (index.year != end_date.year)):
                if currentGLV != lastGLV:
                    print(currentGLV)
                glDate = currentDate
                lastGLV = currentGLV
                counter = 0

    # Determine the message to display based on green line value and current price
    if lastGLV == 0:
        message = f"{ticker} has not formed a green line yet"
    else:
        diff = price/lastGLV - 1
        diff = round(diff * 100, 3)
        if lastGLV > 1.15 * price:
            message = f"\n{ticker.upper()}'s current price ({round(price, 2)}) is {diff}% away from its last green line value ({round(lastGLV, 2)})"
        else:
            if lastGLV < 1.05 * price:
                st.write(f"\n{ticker.upper()}'s last green line value ({round(lastGLV, 2)}) is {diff}% greater than its current price ({round(price, 2)})")
                message = ("Last Green Line: "+str(round(lastGLV, 2))+" on "+str(glDate.strftime('%Y-%m-%d')))
                
                # Plot interactive graph with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index[-120:], y=df['Close'].tail(120), mode='lines', name='Close Price'))
                fig.add_hline(y=lastGLV, line_dash="dash", line_color="green", name='Last Green Line')
                fig.update_layout(title=f"{ticker.upper()}'s Close Price Green Line Test", xaxis_title='Dates', yaxis_title='Close Price')
                st.plotly_chart(fig)
            else:
                message = ("Last Green Line: "+str(round(lastGLV, 2))+" on "+str(glDate.strftime('%Y-%m-%d')))
                # Plot interactive graph with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
                fig.add_hline(y=lastGLV, line_dash="dash", line_color="green", name='Last Green Line')
                fig.update_layout(title=f"{ticker.upper()}'s Close Price Green Line Test", xaxis_title='Dates', yaxis_title='Close Price')
                st.plotly_chart(fig)
    
    st.write(message)



def norm_greenline_test(start_date , end_date , ticker):
    '''This tool allows you to test the green line strategy for a stock'''
    start = dt.datetime.combine(start_date, dt.datetime.min.time())
    end = dt.datetime.combine(end_date, dt.datetime.min.time())

    df = yf.download(ticker, start, end)
    price = df['Adj Close'][-1]

    # Filter out days with very low trading volume
    df.drop(df[df["Volume"] < 1000].index, inplace=True)

    # Get the monthly maximum of the 'High' column
    df_month = df.groupby(pd.Grouper(freq="M"))["High"].max()

    # Initialize variables for the green line analysis
    glDate, lastGLV, currentDate, currentGLV, counter = 0, 0, "", 0, 0

    # Loop through monthly highs to determine the most recent green line value
    for index, value in df_month.items():
        if value > currentGLV:
            currentGLV = value
            currentDate = index
            counter = 0
        if value < currentGLV:
            counter += 1
            if counter == 3 and ((index.month != end_date.month) or (index.year != end_date.year)):
                if currentGLV != lastGLV:
                    print(currentGLV)
                glDate = currentDate
                lastGLV = currentGLV
                counter = 0

    # Determine the message to display based on green line value and current price
    if lastGLV == 0:
        message = f"{ticker} has not formed a green line yet"
    else:
        diff = price/lastGLV - 1
        diff = round(diff * 100, 3)
        if lastGLV > 1.15 * price:
            message = f"\n{ticker.upper()}'s current price ({round(price, 2)}) is {diff}% away from its last green line value ({round(lastGLV, 2)})"
        else:
            if lastGLV < 1.05 * price:
                st.write(f"\n{ticker.upper()}'s last green line value ({round(lastGLV, 2)}) is {diff}% greater than its current price ({round(price, 2)})")
                message = ("Last Green Line: "+str(round(lastGLV, 2))+" on "+str(glDate.strftime('%Y-%m-%d')))
                
                # Plot interactive graph with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index[-120:], y=df['Close'].tail(120), mode='lines', name='Close Price'))
                fig.add_hline(y=lastGLV, line_dash="dash", line_color="green", name='Last Green Line')
                fig.update_layout(title=f"{ticker.upper()}'s Close Price Green Line Test", xaxis_title='Dates', yaxis_title='Close Price')
                st.plotly_chart(fig)
            else:
                message = ("Last Green Line: "+str(round(lastGLV, 2))+" on "+str(glDate.strftime('%Y-%m-%d')))
                # Plot interactive graph with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
                fig.add_hline(y=lastGLV, line_dash="dash", line_color="green", name='Last Green Line')
                fig.update_layout(title=f"{ticker.upper()}'s Close Price Green Line Test", xaxis_title='Dates', yaxis_title='Close Price')
                st.plotly_chart(fig)
    
    st.write(message)