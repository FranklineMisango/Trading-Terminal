import requests
import streamlit as st
from textblob import TextBlob
import datetime
import numpy as np
import datetime as dt
import os
from langchain_core.tools import tool

API_FMPCLOUD = os.environ.get("API_FMPCLOUD")


@tool
def tool_sentiment_analysis(ticker : str, start_date : dt.date, end_date : dt.date):
    '''This tool allows you to analyze the sentiment of a company's earnings call transcript.'''
     # Retrieves earnings call transcript from API
    # TODO identify the error with the API with key error 0
    def get_earnings_call_transcript(api_key, company, quarter, year):
        url = f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{company}?quarter={quarter}&year={year}&apikey={api_key}'
        response = requests.get(url)
        return response.json()[0]['content']

    # Performs sentiment analysis on the transcript.
    def analyze_sentiment(transcript):
        sentiment_call = TextBlob(transcript)
        return sentiment_call

    # Counts the number of positive, negative, and neutral sentences.
    def count_sentiments(sentiment_call):
        positive, negative, neutral = 0, 0, 0
        all_sentences = []

        for sentence in sentiment_call.sentences:
            polarity = sentence.sentiment.polarity
            if polarity < 0:
                negative += 1
            elif polarity > 0:
                positive += 1
            else:
                neutral += 1
            all_sentences.append(polarity)

        return positive, negative, neutral, np.array(all_sentences)

    api_key = API_FMPCLOUD
    # Get transcript and perform sentiment analysis
    transcript = get_earnings_call_transcript(api_key, ticker, 3, 2020)
    sentiment_call = analyze_sentiment(transcript)

    # Count sentiments and calculate mean polarity
    positive, negative, neutral, all_sentences = count_sentiments(sentiment_call)
    mean_polarity = all_sentences.mean()

    # Print results
    st.write(f"Earnings Call Transcript for {ticker}:\n{transcript}\n")
    st.write(f"Overall Sentiment: {sentiment_call.sentiment}")
    st.write(f"Positive Sentences: {positive}, Negative Sentences: {negative}, Neutral Sentences: {neutral}")
    st.write(f"Average Sentence Polarity: {mean_polarity}")

    # Print very positive sentences
    print("\nHighly Positive Sentences:")
    for sentence in sentiment_call.sentences:
        if sentence.sentiment.polarity > 0.8:
            st.write(sentence)


def norm_sentiment_analysis(company, start_date, end_date):
    '''This tool allows you to analyze the sentiment of a company's earnings call transcript.'''
    def get_earnings_call_transcript(api_key, company, quarter, year):
        url = f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{company}?quarter={quarter}&year={year}&apikey={api_key}'
        response = requests.get(url)
        return response.json()[0]['content']

    # Performs sentiment analysis on the transcript.
    def analyze_sentiment(transcript):
        sentiment_call = TextBlob(transcript)
        return sentiment_call

    # Counts the number of positive, negative, and neutral sentences.
    def count_sentiments(sentiment_call):
        positive, negative, neutral = 0, 0, 0
        all_sentences = []

        for sentence in sentiment_call.sentences:
            polarity = sentence.sentiment.polarity
            if polarity < 0:
                negative += 1
            elif polarity > 0:
                positive += 1
            else:
                neutral += 1
            all_sentences.append(polarity)

        return positive, negative, neutral, np.array(all_sentences)


    api_key = API_FMPCLOUD

    # Get transcript and perform sentiment analysis
    transcript = get_earnings_call_transcript(api_key, company, 3, 2020)
    sentiment_call = analyze_sentiment(transcript)

    # Count sentiments and calculate mean polarity
    positive, negative, neutral, all_sentences = count_sentiments(sentiment_call)
    mean_polarity = all_sentences.mean()

    # Print results
    st.write(f"Earnings Call Transcript for {company}:\n{transcript}\n")
    st.write(f"Overall Sentiment: {sentiment_call.sentiment}")
    st.write(f"Positive Sentences: {positive}, Negative Sentences: {negative}, Neutral Sentences: {neutral}")
    st.write(f"Average Sentence Polarity: {mean_polarity}")

    # Print very positive sentences
    print("\nHighly Positive Sentences:")
    for sentence in sentiment_call.sentences:
        if sentence.sentiment.polarity > 0.8:
            st.write(sentence)