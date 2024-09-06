import datetime as dt
from email.message import EmailMessage
import streamlit as st
import yfinance as yf
from time import sleep
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import os
import smtplib
import time
from langchain_core.tools import tool


load_dotenv()
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

@tool
def tool_price_alerter(start_date : dt.date, end_date : dt.date, stock : str, target_price : float, email_address : str):
    '''This tool allows you to set an alert for a stock price'''
    msg = EmailMessage()
    msg['Subject'] = f'Alert on {stock} from Frank & Co. Trading Terminal!'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email_address # Set the recipient email address (this is mine for testing )

    col1, col2 = st.columns([2, 2])
    with col1:
        start_date = st.date_input("Start date:")
    with col2:
        end_date = st.date_input("End Date:")
    # Initialize alerted flag
    alerted = False

    if st.button("Check"):
        # Fetch stock data
        df = yf.download(stock, start_date, end_date)
        current_close = df["Adj Close"][-1]
        # Check if the target price is reached
        if current_close > target_price and not alerted:
            alerted = True
            message_three = f"Your Monitored {stock} has reached the alert price of {target_price}\nCurrent Price: {current_close}\nContact Our Trading Partners to Buy\n Automated Message by Trading Terminal"
            st.write(message_three)
            msg.set_content(message_three)

            # Send email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)
                st.write("Email sent successfully, User Alerted.")
        else:
            print("No new alerts.")

        # Wait for 60 seconds before the next check
        time.sleep(60)

def normal_price_alerter(start_date, end_date, stock, email_address, target_price):
    # Email setup
    msg = EmailMessage()
    msg['Subject'] = f'Alert on {stock} from Frank & Co. LP Trading Terminal!'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email_address # Set the recipient email address (this is mine for testing )

    col1, col2 = st.columns([2, 2])
    with col1:
        start_date = st.date_input("Start date:")
    with col2:
        end_date = st.date_input("End Date:")
    # Initialize alerted flag
    alerted = False

    if st.button("Check"):
        # Fetch stock data
        df = yf.download(stock, start_date, end_date)
        current_close = df["Adj Close"][-1]
        # Check if the target price is reached
        if current_close > target_price and not alerted:
            alerted = True
            message_three = f"Your Monitored {stock} has reached the alert price of {target_price}\nCurrent Price: {current_close}\nContact Our Trading Partners to Buy\n Automated Message by Trading Terminal"
            st.write(message_three)
            msg.set_content(message_three)

            # Send email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)
                st.write("Email sent successfully, User Alerted.")
        else:
            print("No new alerts.")

        # Wait for 60 seconds before the next check
        time.sleep(60)