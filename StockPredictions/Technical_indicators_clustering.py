import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import datetime as dt
from bs4 import BeautifulSoup
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from plotly import graph_objects as go
from langchain_core.tools import tool


#Helper functions
def fetch_sp100_tickers():
        response = requests.get("https://en.wikipedia.org/wiki/S%26P_100")
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table", {"class": "wikitable sortable"})
        tickers = [row.findAll("td")[0].text.strip() for row in table.findAll("tr")[1:]]
        return tickers

        # Download historical data for each ticker

def download_stock_data(tickers, start_date, end_date):
    all_data = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            data['Symbol'] = ticker
            all_data = all_data.append(data)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    return all_data

# Add technical indicators to the data
def add_technical_indicators(data):
    # Example implementation for Simple Moving Average (SMA)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_15'] = data['Close'].rolling(window=15).mean()
    # Additional technical indicators can be added here
    return data

@tool
# Function to fetch S&P 100 tickers from Wikipedia
def tool_technical_indicators_clustering(start_date : dt.date, end_date : dt.date, ticker: str):
    '''This tool allows you to cluster S&P 100 stocks based on technical indicators'''
    

    # Get list of S&P 100 tickers
    sp100_tickers = fetch_sp100_tickers()

    # Download stock data
    stock_data = download_stock_data(sp100_tickers, start_date, end_date)

    # Add technical indicators to the data
    stock_data_with_indicators = add_technical_indicators(stock_data)

    # Clustering based on technical indicators
    # This part of the script would involve applying clustering algorithms
    # such as KMeans or Gaussian Mixture Models to the technical indicators
    # Clustering based on technical indicators
    def perform_clustering(data, n_clusters=10):
        # Selecting only the columns with technical indicators
        indicators = data[['SMA_5', 'SMA_15']].copy()  # Add other indicators as needed
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')  # You can choose another strategy if needed
        indicators = imputer.fit_transform(indicators)
        
        # KMeans Clustering
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(indicators)
        data['Cluster'] = kmeans.labels_

        # Gaussian Mixture Model Clustering
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(indicators)
        data['GMM_Cluster'] = gmm.predict(indicators)

        return data

    # Perform clustering on the stock data
    clustered_data = perform_clustering(stock_data_with_indicators)
    st.write(clustered_data)

    # Create the figure
    fig = go.Figure()
    for cluster_num in clustered_data['Cluster'].unique():
        cluster = clustered_data[clustered_data['Cluster'] == cluster_num]
        for _, row in cluster.iterrows():
            fig.add_trace(go.Scatter(x=[row['SMA_5']], y=[row['SMA_15']], mode='markers', name=f'{row["Symbol"]} - Cluster {cluster_num}'))

    # Update the layout
    fig.update_layout(title='Stock Data Clusters', xaxis_title='SMA 5', yaxis_title='SMA 15')

    # Display the Plotly chart inline
    st.plotly_chart(fig)

    # Analyze and interpret the clusters
    # This step involves understanding the characteristics of each cluster
    # and how they differ from each other based on the stock movement patterns they represent
    # Analyzing Clusters
    def analyze_clusters(data):
        # Group data by clusters
        grouped = data.groupby('Cluster')

        # Analyze clusters
        cluster_analysis = pd.DataFrame()
        for name, group in grouped:
            analysis = {
                'Cluster': name,
                'Average_SMA_5': group['SMA_5'].mean(),
                'Average_SMA_15': group['SMA_15'].mean(),
                # Add more analysis as needed
            }
            cluster_analysis = cluster_analysis.append(analysis, ignore_index=True)
        
        return cluster_analysis

    # Perform cluster analysis
    cluster_analysis = analyze_clusters(clustered_data)

    # Display analysis results
    st.write(cluster_analysis)

# Function to fetch S&P 100 tickers from Wikipedia
def norm_technical_indicators_clustering(start_date, end_date, ticker):
    '''This tool allows you to cluster S&P 100 stocks based on technical indicators'''
    

    # Get list of S&P 100 tickers
    sp100_tickers = fetch_sp100_tickers()

    # Download stock data
    stock_data = download_stock_data(sp100_tickers, start_date, end_date)

    # Add technical indicators to the data
    stock_data_with_indicators = add_technical_indicators(stock_data)

    # Clustering based on technical indicators
    # This part of the script would involve applying clustering algorithms
    # such as KMeans or Gaussian Mixture Models to the technical indicators
    # Clustering based on technical indicators
    def perform_clustering(data, n_clusters=10):
        # Selecting only the columns with technical indicators
        indicators = data[['SMA_5', 'SMA_15']].copy()  # Add other indicators as needed
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')  # You can choose another strategy if needed
        indicators = imputer.fit_transform(indicators)
        
        # KMeans Clustering
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(indicators)
        data['Cluster'] = kmeans.labels_

        # Gaussian Mixture Model Clustering
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(indicators)
        data['GMM_Cluster'] = gmm.predict(indicators)

        return data

    # Perform clustering on the stock data
    clustered_data = perform_clustering(stock_data_with_indicators)
    st.write(clustered_data)

    # Create the figure
    fig = go.Figure()
    for cluster_num in clustered_data['Cluster'].unique():
        cluster = clustered_data[clustered_data['Cluster'] == cluster_num]
        for _, row in cluster.iterrows():
            fig.add_trace(go.Scatter(x=[row['SMA_5']], y=[row['SMA_15']], mode='markers', name=f'{row["Symbol"]} - Cluster {cluster_num}'))

    # Update the layout
    fig.update_layout(title='Stock Data Clusters', xaxis_title='SMA 5', yaxis_title='SMA 15')

    # Display the Plotly chart inline
    st.plotly_chart(fig)

    # Analyze and interpret the clusters
    # This step involves understanding the characteristics of each cluster
    # and how they differ from each other based on the stock movement patterns they represent
    # Analyzing Clusters
    def analyze_clusters(data):
        # Group data by clusters
        grouped = data.groupby('Cluster')

        # Analyze clusters
        cluster_analysis = pd.DataFrame()
        for name, group in grouped:
            analysis = {
                'Cluster': name,
                'Average_SMA_5': group['SMA_5'].mean(),
                'Average_SMA_15': group['SMA_15'].mean(),
                # Add more analysis as needed
            }
            cluster_analysis = cluster_analysis.append(analysis, ignore_index=True)
        
        return cluster_analysis

    # Perform cluster analysis
    cluster_analysis = analyze_clusters(clustered_data)

    # Display analysis results
    st.write(cluster_analysis)