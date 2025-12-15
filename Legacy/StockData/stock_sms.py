import requests
import pandas as pd
from bs4 import BeautifulSoup
import smtplib
import os
import requests
import time
import schedule
import datetime as dt
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText
from langchain_core.tools import tool
import streamlit as st
import yfinance as yf
import ta_functions as ta
from urllib.request import Request, urlopen
from nltk.sentiment.vader import SentimentIntensityAnalyzer


from dotenv import load_dotenv
load_dotenv()

EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')

@tool
def tool_stock_sms(ticker : str , start_date : dt.time, end_date : dt.time, email_address : str):
    '''This tool retrieves stock data and sends an email with the data'''
 # Function to send an email message
    def send_message(text, sender_email, receiver_email, password):
        try:
            # Create and configure the email message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = "Stock Data"
            msg.attach(MIMEText(text, 'plain'))

            # Establish SMTP connection and send the email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            print('Email sent successfully')
        except Exception as e:
            print(f'Error sending email: {e}')

    # Function to retrieve and process stock data
    def get_data(ticker):
        sender_email = EMAIL_ADDRESS  # Sender's email
        receiver_email = email_address  # Receiver's email
        password = EMAIL_PASSWORD  # Sender's email password

        try:
            # Fetch historical stock data
            stock  = ticker
            df = yf.download(stock, start_date, end_date)
            st.write(f'Retrieving data for {stock}')

            # Compute current price and other metrics
            price = round(df['Adj Close'][-1], 2)
            df["rsi"] = ta.RSI(df["Adj Close"])
            rsi = round(df["rsi"].tail(14).mean(), 2)

            # Scrape and analyze news sentiment
            finviz_url = 'https://finviz.com/quote.ashx?t='
            req = Request(url=finviz_url + stock, headers={'user-agent': 'Mozilla/5.0'})
            response = urlopen(req).read()
            html = BeautifulSoup(response, "html.parser")
            news_df = pd.read_html(str(html), attrs={'id': 'news-table'})[0]
            # Process news data
            news_df.columns = ['datetime', 'headline']
            news_df['date'] = pd.to_datetime(news_df['datetime'].str.split(' ').str[0], errors='coerce')
            news_df['date'] = news_df['date'].fillna(method='ffill')
            news_df['sentiment'] = news_df['headline'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
            sentiment = round(news_df['sentiment'].mean(), 2)

            # Prepare and send email message
            output = f"\nTicker: {stock}\nCurrent Price: {price}\nNews Sentiment: {sentiment}\nRSI: {rsi}"
            send_message(output, sender_email, receiver_email, password)

        except Exception as e:
            st.write(f'Error processing data for {stock}: {e}')
            
    get_data(ticker)

def norm_stock_sms(ticker, start_date, end_date, email_address):
    '''This tool retrieves stock data and sends an email with the data'''
 # Function to send an email message
    def send_message(text, sender_email, receiver_email, password):
        try:
            # Create and configure the email message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = "Stock Data"
            msg.attach(MIMEText(text, 'plain'))

            # Establish SMTP connection and send the email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            print('Email sent successfully')
        except Exception as e:
            print(f'Error sending email: {e}')

    # Function to retrieve and process stock data
    def get_data(ticker):
        sender_email = EMAIL_ADDRESS  # Sender's email
        receiver_email = email_address  # Receiver's email
        password = EMAIL_PASSWORD  # Sender's email password

        try:
            # Fetch historical stock data
            stock  = ticker
            df = yf.download(stock, start_date, end_date)
            st.write(f'Retrieving data for {stock}')

            # Compute current price and other metrics
            price = round(df['Adj Close'][-1], 2)
            df["rsi"] = ta.RSI(df["Adj Close"])
            rsi = round(df["rsi"].tail(14).mean(), 2)

            # Scrape and analyze news sentiment
            finviz_url = 'https://finviz.com/quote.ashx?t='
            req = Request(url=finviz_url + stock, headers={'user-agent': 'Mozilla/5.0'})
            response = urlopen(req).read()
            html = BeautifulSoup(response, "html.parser")
            news_df = pd.read_html(str(html), attrs={'id': 'news-table'})[0]
            # Process news data
            news_df.columns = ['datetime', 'headline']
            news_df['date'] = pd.to_datetime(news_df['datetime'].str.split(' ').str[0], errors='coerce')
            news_df['date'] = news_df['date'].fillna(method='ffill')
            news_df['sentiment'] = news_df['headline'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
            sentiment = round(news_df['sentiment'].mean(), 2)

            # Prepare and send email message
            output = f"\nTicker: {stock}\nCurrent Price: {price}\nNews Sentiment: {sentiment}\nRSI: {rsi}"
            send_message(output, sender_email, receiver_email, password)

        except Exception as e:
            st.write(f'Error processing data for {stock}: {e}')
            
    get_data(ticker)