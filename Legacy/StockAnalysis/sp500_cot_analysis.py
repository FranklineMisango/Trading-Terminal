import os
import zipfile
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from langchain_core.tools import tool
import datetime as dt
from urllib.request import urlopen
import urllib.request
import shutil


#helper 
def download_and_extract_cot_file(url, file_name):
    # Download and extract COT file
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    with zipfile.ZipFile(file_name) as zf:
        zf.extractall()

@tool
def tool_sp_cot_analysis(start_date:dt.time, end_date:dt.time):
    '''This function downloads and analyzes the Commitment of Traders (COT) report for E-MINI S&P 500 futures.'''
    this_year = end_date.year
    start_year = start_date.year
    frames = []
    for year in range(start_year, this_year):
        #TODO - fix the link CFTC link
        url = f'https://www.cftc.gov/files/dea/history/fut_fin_xls_{year}.zip'
        download_and_extract_cot_file(url, f'{year}.zip')
        os.rename('FinFutYY.xls', f'{year}.xls')

        data = pd.read_excel(f'{year}.xls')
        data = data.set_index('Report_Date_as_MM_DD_YYYY')
        data.index = pd.to_datetime(data.index)
        data = data[data['Market_and_Exchange_Names'] == 'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE']
        frames.append(data)

    # Concatenate yearly data frames
    df = pd.concat(frames)
    df.to_csv('COT_sp500_data.csv')

    # Read data for plotting
    df = pd.read_csv('COT_sp500_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index)

    # Plotting Line Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Dealer_Long_All'], mode='lines', name='Dealer Long'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Lev_Money_Long_All'], mode='lines', name='Leveraged Long'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Dealer_Short_All'], mode='lines', name='Dealer Short'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Lev_Money_Short_All'], mode='lines', name='Leveraged Short'))
    fig1.update_layout(title='Net Positions - Line Chart', xaxis_title='Date', yaxis_title='Percentage')
    st.plotly_chart(fig1)

    # Box Plot
    fig2 = go.Figure()
    fig2.add_trace(go.Box(y=df['Pct_of_OI_Dealer_Long_All'], name='Dealer Long'))
    fig2.add_trace(go.Box(y=df['Pct_of_OI_Dealer_Short_All'], name='Dealer Short'))
    fig2.add_trace(go.Box(y=df['Pct_of_OI_Lev_Money_Long_All'], name='Leveraged Money Long'))
    fig2.add_trace(go.Box(y=df['Pct_of_OI_Lev_Money_Short_All'], name='Leveraged Money Short'))
    fig2.update_layout(title='Distribution of Open Interest by Category', yaxis_title='Percentage')
    st.plotly_chart(fig2)

    filtered_df = df.loc[start_date:end_date]
    # Plotting Line Chart
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Dealer_Long_All'], mode='lines', name='Dealer Long'))
    fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Lev_Money_Long_All'], mode='lines', name='Leveraged Long'))
    fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Dealer_Short_All'], mode='lines', name='Dealer Short'))
    fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Lev_Money_Short_All'], mode='lines', name='Leveraged Short'))
    fig3.update_layout(title='Net Positions - Line Chart', xaxis_title='Date', yaxis_title='Percentage')
    st.plotly_chart(fig3)


def norm_sp_cot_analysis(start_date,end_date):
    this_year = end_date.year
    start_year = start_date.year
    frames = []
    for year in range(start_year, this_year):
        #TODO - fix the link CFTC link
        url = f'https://www.cftc.gov/files/dea/history/fut_fin_xls_{year}.zip'
        download_and_extract_cot_file(url, f'{year}.zip')
        os.rename('FinFutYY.xls', f'{year}.xls')

        data = pd.read_excel(f'{year}.xls')
        data = data.set_index('Report_Date_as_MM_DD_YYYY')
        data.index = pd.to_datetime(data.index)
        data = data[data['Market_and_Exchange_Names'] == 'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE']
        frames.append(data)

    # Concatenate yearly data frames
    df = pd.concat(frames)
    df.to_csv('COT_sp500_data.csv')

    # Read data for plotting
    df = pd.read_csv('COT_sp500_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index)

    # Plotting Line Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Dealer_Long_All'], mode='lines', name='Dealer Long'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Lev_Money_Long_All'], mode='lines', name='Leveraged Long'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Dealer_Short_All'], mode='lines', name='Dealer Short'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['Pct_of_OI_Lev_Money_Short_All'], mode='lines', name='Leveraged Short'))
    fig1.update_layout(title='Net Positions - Line Chart', xaxis_title='Date', yaxis_title='Percentage')
    st.plotly_chart(fig1)

    # Box Plot
    fig2 = go.Figure()
    fig2.add_trace(go.Box(y=df['Pct_of_OI_Dealer_Long_All'], name='Dealer Long'))
    fig2.add_trace(go.Box(y=df['Pct_of_OI_Dealer_Short_All'], name='Dealer Short'))
    fig2.add_trace(go.Box(y=df['Pct_of_OI_Lev_Money_Long_All'], name='Leveraged Money Long'))
    fig2.add_trace(go.Box(y=df['Pct_of_OI_Lev_Money_Short_All'], name='Leveraged Money Short'))
    fig2.update_layout(title='Distribution of Open Interest by Category', yaxis_title='Percentage')
    st.plotly_chart(fig2)

    filtered_df = df.loc[start_date:end_date]
    # Plotting Line Chart
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Dealer_Long_All'], mode='lines', name='Dealer Long'))
    fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Lev_Money_Long_All'], mode='lines', name='Leveraged Long'))
    fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Dealer_Short_All'], mode='lines', name='Dealer Short'))
    fig3.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Pct_of_OI_Lev_Money_Short_All'], mode='lines', name='Leveraged Short'))
    fig3.update_layout(title='Net Positions - Line Chart', xaxis_title='Date', yaxis_title='Percentage')
    st.plotly_chart(fig3)