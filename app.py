import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import tickers as ti
import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen

st.title('Comprehensive Lite Algorithmic Trading Terminal : Identify, Visualize, Predict and Trade')
st.sidebar.info('Welcome to my Algorithmic Trading App Choose your options below')
st.success("This application features over 100 programmes for different roles")

@st.cache_resource
def correlated_stocks(start_date, end_date, tickers):
    print("Inside correlated_stocks function")
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    returns = np.log(data / data.shift(1))
    correlation = returns.corr()
    return correlation

# Function to visualize correlations as a heatmap using Plotly
def visualize_correlation_heatmap(correlation):
    print("Inside visualize_correlation_heatmap function")
    fig = ff.create_annotated_heatmap(
        z=correlation.values,
        x=correlation.columns.tolist(),
        y=correlation.index.tolist(),
        colorscale='Viridis',
        annotation_text=correlation.values.round(2),
        showscale=True
    )
    fig.update_layout(
        title='Correlation Matrix',
        xaxis=dict(title='Tickers', tickangle=90),
        yaxis=dict(title='Tickers'),
        width=1000,
        height=1000
    )
    st.plotly_chart(fig)

def main():
    option = st.sidebar.selectbox('Make a choice', ['Find stocks','Stock Data', 'Stock Analysis','Technical Indicators', 'Stock Predictions', 'Portfolio Strategies'])
    if option == 'Find stocks':
        options = st.selectbox("Choose a stock finding method:", ["IDB_RS_Rating", "Correlated Stocks", "Finviz_growth_screener", "Fundamental_screener", "RSI_Stock_tickers", "Green_line Valuations", "Minervini_screener", "Pricing Alert Email", "Basic Sentiment", "Trading View Signals", "Twitter Screener", "Yahoo Recommendations"])
        if options == "IDB_RS_Rating":
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            if st.button('Start Analysis'):
                sp500_tickers = ti.tickers_sp500()
                sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]
                sp500_df = yf.download(sp500_tickers, start=start_date, end=end_date)
                percentage_change_df = sp500_df['Adj Close'].pct_change()
                sp500_df = pd.concat([sp500_df, percentage_change_df.add_suffix('_PercentChange')], axis=1)
                st.write(sp500_df)
        if options == "Correlated Stocks":
            col1, col2 = st.columns([2, 2])
            with col1:
                start_date = st.date_input("Start date:")
            with col2:
                end_date = st.date_input("End Date:")
            st.title('Correlation Viewer for Stocks')
            
            # Add more stocks to the portfolio
            sectors = {
                "Technology": ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'INTC', 'CSCO', 'ADBE', 'AVGO', 'PYPL'],
                "Health Care": ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'MDT', 'DHR', 'AMGN', 'LLY'],
                "Financials": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'USB'],
                "Consumer Discretionary": ['AMZN', 'TSLA', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'BKNG', 'LOW', 'CMG'],
                "Communication Services": ['GOOGL', 'META', 'DIS', 'CMCSA', 'NFLX', 'T', 'CHTR', 'DISCA', 'FOXA', 'VZ'],
                "Industrials": ['BA', 'HON', 'UNP', 'UPS', 'CAT', 'LMT', 'MMM', 'GD', 'GE', 'CSX'],
                "Consumer Staples": ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'MO', 'EL', 'KHC', 'MDLZ'],
                "Utilities": ['NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'PEG', 'XEL', 'ED'],
                "Real Estate": ['AMT', 'PLD', 'CCI', 'EQIX', 'WY', 'PSA', 'DLR', 'BXP', 'O', 'SBAC'],
                "Materials": ['LIN', 'APD', 'SHW', 'DD', 'ECL', 'DOW', 'NEM', 'PPG', 'VMC', 'FCX']
            }

            selected_sector = st.selectbox('Select Sector', list(sectors.keys()))
            if st.button("Start Analysis"):
                print("Before fetching tickers")
                tickers = sectors[selected_sector]
                print("After fetching tickers")
                corr_matrix = correlated_stocks(start_date, end_date, tickers)
                print("After computing correlation matrix")
                visualize_correlation_heatmap(corr_matrix)
        if options == "Finviz_growth_screener":
            # Set display options for pandas DataFrame
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            def growth_screener():
                try:
                    frames = []
                    # Loop through different pages of FinViz growth screener
                    for i in range(1, 105, 20):
                        url = (f"https://finviz.com/screener.ashx?v=151&f=fa_epsqoq_o15,fa_epsyoy_pos,fa_epsyoy1_o25,fa_grossmargin_pos,fa_salesqoq_o25,ind_stocksonly,sh_avgvol_o300,sh_insttrans_pos,sh_price_o10,ta_perf_52w50o,ta_sma200_pa,ta_sma50_pa&ft=4&o=-marketcap&r=0{i}")
                        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        webpage = urlopen(req).read()
                        html = soup(webpage, "html.parser")

                        # Extracting stock data from the HTML
                        stocks = pd.read_html(str(html))[-2]
                        stocks = stocks.set_index('Ticker')
                        frames.append(stocks)

                    # Concatenate all DataFrame objects from different pages
                    df = pd.concat(frames)
                    df = df.drop_duplicates()
                    df = df.drop(columns=['No.'])
                    return df
                except Exception as e:
                    print(f"Error occurred: {e}")
                    return pd.DataFrame()

            # Execute the screener and store the result
            df = growth_screener()

            # Display the results
            st.write('\nGrowth Stocks Screener:')
            st.write(df)

            # Extract and print list of tickers
            tickers = df.index
            st.write('\nList of Tickers:')
            st.write(tickers)
        if options == "s":
        # Import dependencies
            import requests
            import pandas as pd
            from config import financial_model_prep

            # Get the API key
            demo = financial_model_prep()

            # Define search criteria for the stock screener
            marketcap = str(1000000000)
            url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan={marketcap}&betaMoreThan=1&volumeMoreThan=10000&sector=Technology&exchange=NASDAQ&dividendMoreThan=0&limit=1000&apikey={demo}'

            # Fetch list of companies meeting criteria
            screener = requests.get(url).json()

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
                    print(f"Error processing {company}: {e}")

            # Convert to DataFrame and display
            df = pd.DataFrame.from_dict(value_ratios, orient='index')
            print(df.head())

            # Define and apply ranking criteria
            criteria = { 'ROE': 1.2, 'ROA': 1.1, 'Debt_Ratio': -1.1, # etc.
                        'Revenue_Growth': 1.25, 'NetIncome_Growth': 1.10 }

            # Normalize and rank companies
            mean_values = df.mean()
            normalized_df = df / mean_values
            normalized_df['ranking'] = sum(normalized_df[col] * weight for col, weight in criteria.items())

            # Print ranked companies
            print(normalized_df.sort_values(by=['ranking'], ascending=False))

    elif option == 'Recent Data':
        pass
    else:
        pass

if __name__ == "__main__":
    main()
