import streamlit as st
import pandas as pd
import requests
import datetime as dt
import calendar
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()

API_FMPCLOUD = os.environ.get("API_FMPCLOUD")
pd.set_option('display.max_columns', None)

# Helper class
class DividendCalendar:
    def __init__(self, year, month):
        self.year = year
        self.month = month
        self.url = 'https://api.nasdaq.com/api/calendar/dividends'
        self.hdrs = {
            'Accept': 'text/plain, */*',
            'DNT': "1",
            'Origin': 'https://www.nasdaq.com/',
            'Sec-Fetch-Mode': 'cors',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0)'
        }
        self.calendars = []

    def date_str(self, day):
        return dt.date(self.year, self.month, day).strftime('%Y-%m-%d')

    def scraper(self, date_str):
        response = requests.get(self.url, headers=self.hdrs, params={'date': date_str})
        return response.json()

    def dict_to_df(self, dictionary):
        rows = dictionary.get('data').get('calendar').get('rows', [])
        calendar_df = pd.DataFrame(rows)
        self.calendars.append(calendar_df)
        return calendar_df

    def calendar(self, day):
        date_str = self.date_str(day)
        dictionary = self.scraper(date_str)
        return self.dict_to_df(dictionary)

@tool
def tool_get_dividend_calendar(year: int, month: int):
    '''This tool allows you to view dividend data for a given year and month'''
    try:
        dc = DividendCalendar(year, month)
        days_in_month = calendar.monthrange(year, month)[1]

        for day in range(1, days_in_month + 1):
            dc.calendar(day)

        concat_df = pd.concat(dc.calendars).dropna(how='any')
        concat_df = concat_df.set_index('companyName').reset_index()
        concat_df = concat_df.drop(columns=['announcement_Date'])
        concat_df.columns = ['Company Name', 'Ticker', 'Dividend Date', 'Payment Date', 
                            'Record Date', 'Dividend Rate', 'Annual Rate']
        concat_df = concat_df.sort_values(['Annual Rate', 'Dividend Date'], ascending=[False, False])
        concat_df = concat_df.drop_duplicates()
        return concat_df
    except Exception as e:
        return str(e)

def norm_get_dividends(year: int, month: int):
    try:
        dc = DividendCalendar(year, month)
        days_in_month = calendar.monthrange(year, month)[1]

        for day in range(1, days_in_month + 1):
            dc.calendar(day)

        concat_df = pd.concat(dc.calendars).dropna(how='any')
        concat_df = concat_df.set_index('companyName').reset_index()
        concat_df = concat_df.drop(columns=['announcement_Date'])
        concat_df.columns = ['Company Name', 'Ticker', 'Dividend Date', 'Payment Date', 
                            'Record Date', 'Dividend Rate', 'Annual Rate']
        concat_df = concat_df.sort_values(['Annual Rate', 'Dividend Date'], ascending=[False, False])
        concat_df = concat_df.drop_duplicates()
        return concat_df
    except Exception as e:
        return str(e)
