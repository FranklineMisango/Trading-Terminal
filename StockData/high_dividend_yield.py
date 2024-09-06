import requests
import pandas as pd
import streamlit as st
from langchain_core.tools import tool
import tickers as ti
import os
from dotenv import load_dotenv
load_dotenv()

API_FMPCLOUD = os.environ.get("API_FMPCLOUD")

@tool
# Create an instance of the financial model prep class for API access
def tool_high_dividend_yield():
    '''This tool allows you to find companies with high dividend yields'''
    demo = API_FMPCLOUD

    # Read ticker symbols from a saved pickle file
    symbols = ti.tickers_sp500()

    # Initialize a dictionary to store dividend yield data for each company
    yield_dict = {}

    # Loop through each ticker symbol in the list
    for company in symbols:
        try:
            # Fetch company data from FinancialModelingPrep API using the ticker symbol
            companydata = requests.get(f'https://fmpcloud.io/api/v3/profile/{company}?apikey={demo}')
            st.write(companydata)
            companydata = companydata.json()  # Convert response to JSON format
            
            # Extract relevant data for dividend calculation
            latest_Annual_Dividend = companydata[0]['lastDiv']
            price = companydata[0]['price']
            market_Capitalization = companydata[0]['mktCap']
            name = companydata[0]['companyName']
            exchange = companydata[0]['exchange']

            # Calculate the dividend yield
            dividend_yield = latest_Annual_Dividend / price

            # Store the extracted data in the yield_dict dictionary
            yield_dict[company] = {
                'dividend_yield': dividend_yield,
                'latest_price': price,
                'latest_dividend': latest_Annual_Dividend,
                'market_cap': market_Capitalization / 1000000,  # Convert market cap to millions
                'company_name': name,
                'exchange': exchange
            }

        except Exception as e:
            # Skip to the next ticker if there's an error with the current one
            print(f"Error processing {company}: {e}")
            continue

    # Convert the yield_dict dictionary to a pandas DataFrame
    yield_dataframe = pd.DataFrame.from_dict(yield_dict, orient='index')

    # Sort the DataFrame by dividend yield in descending order
    yield_dataframe = yield_dataframe.sort_values('dividend_yield', ascending=False)

    # Display the sorted DataFrame
    st.write(yield_dataframe)
