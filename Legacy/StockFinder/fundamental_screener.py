 # Get the API key

import streamlit as st
import requests
import pandas as pd
from langchain_core.tools import tool
import os

from dotenv import load_dotenv
load_dotenv()

@tool
def tool_fundamental_screener():
    '''This tool allows you to screen for fundamental stocks using Financial Modeling Prep'''
    API_FMPCLOUD = os.environ.get("API_FMPCLOUD")

    demo = API_FMPCLOUD


    # Define search criteria for the stock screener
    marketcap = str(1000000000)
    url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan={marketcap}&betaMoreThan=1&volumeMoreThan=10000&sector=Technology&exchange=NASDAQ&dividendMoreThan=0&limit=1000&apikey={demo}'

    # Fetch list of companies meeting criteria
    screener = requests.get(url).json()
    #st.write(screener)

    # Extract symbols of companies
    companies = [item['symbol'] for item in screener]

    # Initialize dictionary for storing financial ratios
    value_ratios = {}

    # Limit the number of companies for ratio extraction
    max_companies = 30

    # Process financial ratios for each company
    for count, company in enumerate(companies):
        if count >= max_companies:
            break

        try:
            # Fetch financial and growth ratios
            fin_url = f'https://financialmodelingprep.com/api/v3/ratios/{company}?apikey={demo}'
            growth_url = f'https://financialmodelingprep.com/api/v3/financial-growth/{company}?apikey={demo}'

            fin_ratios = requests.get(fin_url).json()
            growth_ratios = requests.get(growth_url).json()

            # Store required ratios
            ratios = { 'ROE': fin_ratios[0]['returnOnEquity'], 
                    'ROA': fin_ratios[0]['returnOnAssets'], 
                    # Additional ratios can be added here
                    }

            growth = { 'Revenue_Growth': growth_ratios[0]['revenueGrowth'],
                    'NetIncome_Growth': growth_ratios[0]['netIncomeGrowth'],
                    # Additional growth metrics can be added here
                    }

            value_ratios[company] = {**ratios, **growth}
        except Exception as e:
            st.write(f"Error processing {company}: {e}")

    # Convert to DataFrame and display
    df = pd.DataFrame.from_dict(value_ratios, orient='index')
    # Define and apply ranking criteria
    criteria = { 'ROE': 1.2,
                 'ROA': 1.1, 
                 #'Debt_Ratio': -1.1, # etc.
                'Revenue_Growth': 1.25, 
                'NetIncome_Growth': 1.10 }

    # Normalize and rank companies
    mean_values = df.mean()
    normalized_df = df / mean_values
    normalized_df['ranking'] = sum(normalized_df[col] * weight for col, weight in criteria.items())

    # Print ranked companies
    st.write(normalized_df.sort_values(by=['ranking'], ascending=False))
