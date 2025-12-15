import tickers as ti
import yfinance as yf
from sklearn.cluster import KMeans
import pandas as pd
from math import sqrt
import plotly.graph_objects as go
import datetime as dt
import streamlit as st
from langchain_core.tools import tool


@tool
def tool_kmeans_clustering(start_date: dt.date, end_date: dt.date):
    '''This tool allows you to cluster stocks based on KMeans clustering'''
     # Load stock data from Dow Jones Index
    stocks = ti.tickers_dow()


    # Retrieve adjusted closing prices
    data = yf.download(stocks, start=start_date, end=end_date)['Close']

    # Calculate annual mean returns and variances
    annual_returns = data.pct_change().mean() * 252
    annual_variances = data.pct_change().std() * sqrt(252)

    # Combine returns and variances into a DataFrame
    ret_var = pd.concat([annual_returns, annual_variances], axis=1).dropna()
    ret_var.columns = ["Returns", "Variance"]

    # KMeans clustering
    X = ret_var.values
    sse = [KMeans(n_clusters=k).fit(X).inertia_ for k in range(2, 15)]

    # Convert range to list
    k_values = list(range(2, 15))

    # Plotting the elbow curve using Plotly
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=k_values, y=sse, mode='lines+markers'))
    fig1.update_layout(title="Elbow Curve", xaxis_title="Number of Clusters", yaxis_title="SSE")
    st.plotly_chart(fig1)

    # Apply KMeans with chosen number of clusters
    kmeans = KMeans(n_clusters=5).fit(X)

    # Plotting the clustering result using Plotly
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=kmeans.labels_, colorscale="Rainbow")))
    fig2.update_layout(title="KMeans Clustering of Stocks", xaxis_title="Annual Return", yaxis_title="Annual Variance")
    st.plotly_chart(fig2)

    # Creating a DataFrame with tickers and their cluster labels
    df = pd.DataFrame({'Stock': ret_var.index, 'Cluster': kmeans.labels_})
    df.set_index('Stock', inplace=True)

    st.write(df)


def normal_kmeans_clustering(start_date: dt.date, end_date: dt.date):
    '''This tool allows you to cluster stocks based on KMeans clustering'''
     # Load stock data from Dow Jones Index
    stocks = ti.tickers_dow()


    # Retrieve adjusted closing prices
    data = yf.download(stocks, start=start_date, end=end_date)['Close']

    # Calculate annual mean returns and variances
    annual_returns = data.pct_change().mean() * 252
    annual_variances = data.pct_change().std() * sqrt(252)

    # Combine returns and variances into a DataFrame
    ret_var = pd.concat([annual_returns, annual_variances], axis=1).dropna()
    ret_var.columns = ["Returns", "Variance"]

    # KMeans clustering
    X = ret_var.values
    sse = [KMeans(n_clusters=k).fit(X).inertia_ for k in range(2, 15)]

    # Convert range to list
    k_values = list(range(2, 15))

    # Plotting the elbow curve using Plotly
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=k_values, y=sse, mode='lines+markers'))
    fig1.update_layout(title="Elbow Curve", xaxis_title="Number of Clusters", yaxis_title="SSE")
    st.plotly_chart(fig1)

    # Apply KMeans with chosen number of clusters
    kmeans = KMeans(n_clusters=5).fit(X)

    # Plotting the clustering result using Plotly
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=kmeans.labels_, colorscale="Rainbow")))
    fig2.update_layout(title="KMeans Clustering of Stocks", xaxis_title="Annual Return", yaxis_title="Annual Variance")
    st.plotly_chart(fig2)

    # Creating a DataFrame with tickers and their cluster labels
    df = pd.DataFrame({'Stock': ret_var.index, 'Cluster': kmeans.labels_})
    df.set_index('Stock', inplace=True)

    st.write(df)