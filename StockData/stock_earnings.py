import streamlit as st
import pandas as pd
from yahoo_earnings_calendar import YahooEarningsCalendar
from datetime import datetime, timedelta
import datetime as dt
import dateutil.parser
from langchain_core.tools import tool


@tool
def tool_stock_earnings( start_date : dt.time, end_date : dt.time):

    '''This tool retrieves earnings data for a specific date, date range, and ticker'''

    # Setting pandas display options
    pd.set_option('display.max_columns', None)

    # Download earnings report for a specific date
    report_date = datetime.now().date()
    yec = YahooEarningsCalendar()

    # Fetch earnings data for the specified dates
    earnings_df_list = yec.earnings_between(start_date, end_date)

    # Create a DataFrame from the earnings data and drop unnecessary columns
    earnings_day_df = pd.DataFrame(earnings_df_list)
    earnings_day_df.drop(columns=['gmtOffsetMilliSeconds', 'quoteType', 'timeZoneShortName'], inplace=True)

    # Rename columns for clarity and reorganize them
    earnings_day_df.columns = ['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']
    earnings_day_df = earnings_day_df[['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']]

    # Adjust datetime to local timezone
    earnings_day_df['DateTime'] = pd.to_datetime(earnings_day_df['DateTime']).dt.tz_localize(None)

    # Print the DataFrame
    st.write(earnings_day_df)

    # Download earnings for a range of dates
    DAYS_AHEAD = 7
    start_date = datetime.now().date()
    end_date = (datetime.now().date() + timedelta(days=DAYS_AHEAD))

    # Fetch earnings data between specified dates
    earnings_range_list = yec.earnings_between(start_date, end_date)

    # Create a DataFrame from the fetched data and drop unnecessary columns
    earnings_range_df = pd.DataFrame(earnings_range_list)
    earnings_range_df.drop(columns=['gmtOffsetMilliSeconds', 'quoteType', 'timeZoneShortName'], inplace=True)

    # Rename columns for clarity and reorganize them
    earnings_range_df.columns = ['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']
    earnings_range_df = earnings_range_df[['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']]

    # Adjust datetime to local timezone
    earnings_range_df['DateTime'] = pd.to_datetime(earnings_range_df['DateTime']).dt.tz_localize(None)

    # Print the DataFrame
    print(earnings_range_df)

    # Download earnings for a specific ticker within a date range
    TICKER = 'AAPL'
    DAYS_AHEAD = 180
    start_date = datetime.now().date()
    end_date = (datetime.now().date() + timedelta(days=DAYS_AHEAD))

    # Fetch earnings data for the specified ticker
    earnings_ticker_list = yec.get_earnings_of(TICKER)

    # Create a DataFrame from the fetched data and filter by date range
    earnings_ticker_df = pd.DataFrame(earnings_ticker_list)
    earnings_ticker_df['report_date'] = earnings_ticker_df['startdatetime'].apply(lambda x: dateutil.parser.isoparse(x).date())
    earnings_ticker_df = earnings_ticker_df[earnings_ticker_df['report_date'].between(start_date, end_date)].sort_values('report_date')
    earnings_ticker_df.drop(columns=['gmtOffsetMilliSeconds', 'quoteType', 'timeZoneShortName', 'report_date'], inplace=True)

    # Rename columns for clarity and reorganize them
    earnings_ticker_df.columns = ['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']
    earnings_ticker_df = earnings_ticker_df[['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']]

    # Adjust datetime to local timezone
    earnings_ticker_df['DateTime'] = pd.to_datetime(earnings_ticker_df['DateTime']).dt.tz_localize(None)

    print(earnings_ticker_df)


def norm_stock_earnings(start_date, end_date):

    '''This tool retrieves earnings data for a specific date, date range, and ticker'''

    # Setting pandas display options
    pd.set_option('display.max_columns', None)

    # Download earnings report for a specific date
    report_date = datetime.now().date()
    yec = YahooEarningsCalendar()

    # Fetch earnings data for the specified dates
    earnings_df_list = yec.earnings_between(start_date, end_date)

    # Create a DataFrame from the earnings data and drop unnecessary columns
    earnings_day_df = pd.DataFrame(earnings_df_list)
    earnings_day_df.drop(columns=['gmtOffsetMilliSeconds', 'quoteType', 'timeZoneShortName'], inplace=True)

    # Rename columns for clarity and reorganize them
    earnings_day_df.columns = ['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']
    earnings_day_df = earnings_day_df[['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']]

    # Adjust datetime to local timezone
    earnings_day_df['DateTime'] = pd.to_datetime(earnings_day_df['DateTime']).dt.tz_localize(None)

    # Print the DataFrame
    st.write(earnings_day_df)

    # Download earnings for a range of dates
    DAYS_AHEAD = 7
    start_date = datetime.now().date()
    end_date = (datetime.now().date() + timedelta(days=DAYS_AHEAD))

    # Fetch earnings data between specified dates
    earnings_range_list = yec.earnings_between(start_date, end_date)

    # Create a DataFrame from the fetched data and drop unnecessary columns
    earnings_range_df = pd.DataFrame(earnings_range_list)
    earnings_range_df.drop(columns=['gmtOffsetMilliSeconds', 'quoteType', 'timeZoneShortName'], inplace=True)

    # Rename columns for clarity and reorganize them
    earnings_range_df.columns = ['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']
    earnings_range_df = earnings_range_df[['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']]

    # Adjust datetime to local timezone
    earnings_range_df['DateTime'] = pd.to_datetime(earnings_range_df['DateTime']).dt.tz_localize(None)

    # Print the DataFrame
    print(earnings_range_df)

    # Download earnings for a specific ticker within a date range
    TICKER = 'AAPL'
    DAYS_AHEAD = 180
    start_date = datetime.now().date()
    end_date = (datetime.now().date() + timedelta(days=DAYS_AHEAD))

    # Fetch earnings data for the specified ticker
    earnings_ticker_list = yec.get_earnings_of(TICKER)

    # Create a DataFrame from the fetched data and filter by date range
    earnings_ticker_df = pd.DataFrame(earnings_ticker_list)
    earnings_ticker_df['report_date'] = earnings_ticker_df['startdatetime'].apply(lambda x: dateutil.parser.isoparse(x).date())
    earnings_ticker_df = earnings_ticker_df[earnings_ticker_df['report_date'].between(start_date, end_date)].sort_values('report_date')
    earnings_ticker_df.drop(columns=['gmtOffsetMilliSeconds', 'quoteType', 'timeZoneShortName', 'report_date'], inplace=True)

    # Rename columns for clarity and reorganize them
    earnings_ticker_df.columns = ['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']
    earnings_ticker_df = earnings_ticker_df[['Ticker', 'Company Name', 'DateTime', 'Type', 'EPS Estimate', 'EPS Actual', 'EPS Surprise PCT']]

    # Adjust datetime to local timezone
    earnings_ticker_df['DateTime'] = pd.to_datetime(earnings_ticker_df['DateTime']).dt.tz_localize(None)

    print(earnings_ticker_df)