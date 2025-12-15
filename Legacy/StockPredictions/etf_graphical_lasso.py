import streamlit as st
import datetime as dt
import numpy as np
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.covariance import GraphicalLassoCV
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from langchain_core.tools import tool
import matplotlib.pyplot as plt


@tool
def tool_etf_graphical_lasso(start_date: dt.date, end_date: dt.date, number_of_years: str):
    '''This tool allows you to perform Graphical Lasso on ETFs'''
    num_years = int(number_of_years)
    start_date = dt.datetime.now() - dt.timedelta(days=num_years * 365.25)
    end_date = dt.datetime.now()

    # ETF symbols and their respective countries
    etfs = {"EWJ": "Japan", "EWZ": "Brazil", "FXI": "China",
            "EWY": "South Korea", "EWT": "Taiwan", "EWH": "Hong Kong",
            "EWC": "Canada", "EWG": "Germany", "EWU": "United Kingdom",
            "EWA": "Australia", "EWW": "Mexico", "EWL": "Switzerland",
            "EWP": "Spain", "EWQ": "France", "EIDO": "Indonesia",
            "ERUS": "Russia", "EWS": "Singapore", "EWM": "Malaysia",
            "EZA": "South Africa", "THD": "Thailand", "ECH": "Chile",
            "EWI": "Italy", "TUR": "Turkey", "EPOL": "Poland",
            "EPHE": "Philippines", "EWD": "Sweden", "EWN": "Netherlands",
            "EPU": "Peru", "ENZL": "New Zealand", "EIS": "Israel",
            "EWO": "Austria", "EIRL": "Ireland", "EWK": "Belgium"}

    # Retrieve adjusted close prices for ETFs
    symbols = list(etfs.keys())
    etf_data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

    # Convert prices to log returns
    log_returns = np.log1p(etf_data.pct_change()).dropna()

    # Replace NaN values with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    log_returns_normalized = log_returns / log_returns.std(axis=0)
    log_returns_normalized_imputed = imputer.fit_transform(log_returns_normalized)
    edge_model = GraphicalLassoCV(cv=10)
    edge_model.fit(log_returns_normalized_imputed)

    # Ensure the shape of the precision matrix matches the number of ETF symbols
    precision_matrix = edge_model.precision_
    # Compare the number of ETF symbols with the precision matrix dimensions
    if precision_matrix.shape[0] != len(etfs):
        # If there's a mismatch, print additional information for debugging
        st.write("Mismatch between the number of ETF symbols and precision matrix dimensions")

    # Create DataFrame with appropriate indices and columns
    precision_df = pd.DataFrame(precision_matrix, index=etfs.keys(), columns=etfs.keys())

    fig = go.Figure(data=go.Heatmap(
        z=precision_df.values,
        x=list(etfs.values()),
        y=list(etfs.values()),
        colorscale='Viridis'))

    fig.update_layout(title="Precision Matrix Heatmap")
    st.plotly_chart(fig)

    # Prepare data for network graph
    links = precision_df.stack().reset_index()
    links.columns = ['ETF1', 'ETF2', 'Value']
    links_filtered = links[(abs(links['Value']) > 0.17) & (links['ETF1'] != links['ETF2'])]

    # Build and display the network graph
    st.set_option('deprecation.showPyplotGlobalUse', False)
    G = nx.from_pandas_edgelist(links_filtered, 'ETF1', 'ETF2')
    pos = nx.spring_layout(G, k=0.2 * 1 / np.sqrt(len(G.nodes())), iterations=20)
    nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', edge_color='grey')
    st.pyplot()


def normal_etf_graphical_lasso(start_date, end_date, number_of_years):
    
    num_years = int(number_of_years)
    start_date = dt.datetime.now() - dt.timedelta(days=num_years * 365.25)
    end_date = dt.datetime.now()
    '''This tool allows you to perform Graphical Lasso on ETFs'''

    # ETF symbols and their respective countries
    etfs = {"EWJ": "Japan", "EWZ": "Brazil", "FXI": "China",
            "EWY": "South Korea", "EWT": "Taiwan", "EWH": "Hong Kong",
            "EWC": "Canada", "EWG": "Germany", "EWU": "United Kingdom",
            "EWA": "Australia", "EWW": "Mexico", "EWL": "Switzerland",
            "EWP": "Spain", "EWQ": "France", "EIDO": "Indonesia",
            "ERUS": "Russia", "EWS": "Singapore", "EWM": "Malaysia",
            "EZA": "South Africa", "THD": "Thailand", "ECH": "Chile",
            "EWI": "Italy", "TUR": "Turkey", "EPOL": "Poland",
            "EPHE": "Philippines", "EWD": "Sweden", "EWN": "Netherlands",
            "EPU": "Peru", "ENZL": "New Zealand", "EIS": "Israel",
            "EWO": "Austria", "EIRL": "Ireland", "EWK": "Belgium"}

    # Retrieve adjusted close prices for ETFs
    symbols = list(etfs.keys())
    etf_data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

    # Convert prices to log returns
    log_returns = np.log1p(etf_data.pct_change()).dropna()

    imputer = SimpleImputer(strategy='mean')
    log_returns_normalized = log_returns / log_returns.std(axis=0)
    log_returns_normalized_imputed = imputer.fit_transform(log_returns_normalized)
    edge_model = GraphicalLassoCV(cv=10)
    edge_model.fit(log_returns_normalized_imputed)

    # Ensure the shape of the precision matrix matches the number of ETF symbols
    precision_matrix = edge_model.precision_
    # Compare the number of ETF symbols with the precision matrix dimensions
    if precision_matrix.shape[0] != len(etfs):
        # If there's a mismatch, print additional information for debugging
        st.write("Mismatch between the number of ETF symbols and precision matrix dimensions")

    # Create DataFrame with appropriate indices and columns
    precision_df = pd.DataFrame(precision_matrix, index=etfs.keys(), columns=etfs.keys())

    fig = go.Figure(data=go.Heatmap(
        z=precision_df.values,
        x=list(etfs.values()),
        y=list(etfs.values()),
        colorscale='Viridis'))

    fig.update_layout(title="Precision Matrix Heatmap")
    st.plotly_chart(fig)

    # Prepare data for network graph
    links = precision_df.stack().reset_index()
    links.columns = ['ETF1', 'ETF2', 'Value']
    links_filtered = links[(abs(links['Value']) > 0.17) & (links['ETF1'] != links['ETF2'])]

    # Build and display the network graph
    G = nx.from_pandas_edgelist(links_filtered, 'ETF1', 'ETF2')
    pos = nx.spring_layout(G, k=0.2 * 1 / np.sqrt(len(G.nodes())), iterations=20)

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()
    nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', edge_color='grey', ax=ax)
    st.pyplot(fig)