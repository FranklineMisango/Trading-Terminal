import datetime as dt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from langchain_core.tools import tool
import ta_functions as ta
import pandas as pd
import quandl as q


@tool
def tool_bri(ticker:str, start_date: dt.time, end_date: dt.time):
    ''' This tool plots The Breadth Indicator (BRI) of a stock along with the stock's closing price.'''
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    #df["Adj Close"][1:]

    # ## On Balance Volume
    OBV = ta.OBV(df["Adj Close"], df["Volume"])

    Advances = q.get("URC/NYSE_ADV", start_date="2017-07-27")["Numbers of Stocks"]
    Declines = q.get("URC/NYSE_DEC", start_date="2017-07-27")["Numbers of Stocks"]

    adv_vol = q.get("URC/NYSE_ADV_VOL", start_date="2017-07-27")["Numbers of Stocks"]
    dec_vol = q.get("URC/NYSE_DEC_VOL", start_date="2017-07-27")["Numbers of Stocks"]

    data = pd.DataFrame()
    data["Advances"] = Advances
    data["Declines"] = Declines
    data["adv_vol"] = adv_vol
    data["dec_vol"] = dec_vol

    data["Net_Advances"] = data["Advances"] - data["Declines"]
    data["Ratio_Adjusted"] = (
        data["Net_Advances"] / (data["Advances"] + data["Declines"])
    ) * 1000
    data["19_EMA"] = ta.EMA(data["Ratio_Adjusted"], timeperiod=19)
    data["39_EMA"] = ta.EMA(data["Ratio_Adjusted"], timeperiod=39)
    data["RANA"] = (
        (data["Advances"] - data["Declines"]) / (data["Advances"] + data["Declines"]) * 1000
    )

    # Finding the TRIN Value
    data["ad_ratio"] = data["Advances"].divide(data["Declines"])  # AD Ratio
    data["ad_vol"] = data["adv_vol"].divide(data["dec_vol"])  # AD Volume Ratio
    data["TRIN"] = data["ad_ratio"].divide(data["adv_vol"])  # TRIN Value

    # Function to calculate Force Index
    def ForceIndex(data, n):
        ForceIndex = pd.Series(df["Adj Close"].diff(n) * df["Volume"], name="ForceIndex")
        data = data.join(ForceIndex)
        return data

    # Calculate Force Index
    n = 10
    ForceIndex = ForceIndex(df, n)
    ForceIndex = ForceIndex["ForceIndex"]

    # Market Price Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Close Price'))
    fig.update_layout(title="Market Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Close Price")
    st.plotly_chart(fig)

    # Force Index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ForceIndex, mode='lines', name='Force Index'))
    fig.update_layout(title="Force Index",
                    xaxis_title="Date",
                    yaxis_title="Force Index")
    st.plotly_chart(fig)


    # Function to calculate Chaikin Oscillator
    def Chaikin(data):
        money_flow_volume = (
            (2 * data["Adj Close"] - data["High"] - data["Low"])
            / (data["High"] - data["Low"])
            * data["Volume"]
        )
        ad = money_flow_volume.cumsum()
        Chaikin = pd.Series(
            ad.ewm(com=(3 - 1) / 2).mean() - ad.ewm(com=(10 - 1) / 2).mean(), name="Chaikin"
        )
        #data["Chaikin"] = data.join(Chaikin)
        data["Chaikin"] = Chaikin
        return data


    # Calculate Chaikin Oscillator
    Chaikin(df)


    # Chaikin Oscillator Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Chaikin"], mode='lines', name='Chaikin Oscillator'))
    fig.update_layout(title="Chaikin Oscillator",
                    xaxis_title="Date",
                    yaxis_title="Chaikin Oscillator")
    
    st.plotly_chart(fig)


    # Calculate Cumulative Volume Index
    data["CVI"] = data["Net_Advances"].shift(1) + (data["Advances"] - data["Declines"])

    # Cumulative Volume Index Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["CVI"], mode='lines', name='Cumulative Volume Index'))
    fig.update_layout(title="Cumulative Volume Index",
                    xaxis_title="Date",
                    yaxis_title="CVI")
    
    st.plotly_chart(fig)

def norm_bri(ticker, start_date, end_date):
    symbol = ticker
    start = start_date
    end = end_date

    # Read data
    df = yf.download(symbol, start, end)

    #df["Adj Close"][1:]

    # ## On Balance Volume
    OBV = ta.OBV(df["Adj Close"], df["Volume"])

    Advances = q.get("URC/NYSE_ADV", start_date="2017-07-27")["Numbers of Stocks"]
    Declines = q.get("URC/NYSE_DEC", start_date="2017-07-27")["Numbers of Stocks"]

    adv_vol = q.get("URC/NYSE_ADV_VOL", start_date="2017-07-27")["Numbers of Stocks"]
    dec_vol = q.get("URC/NYSE_DEC_VOL", start_date="2017-07-27")["Numbers of Stocks"]

    data = pd.DataFrame()
    data["Advances"] = Advances
    data["Declines"] = Declines
    data["adv_vol"] = adv_vol
    data["dec_vol"] = dec_vol

    data["Net_Advances"] = data["Advances"] - data["Declines"]
    data["Ratio_Adjusted"] = (
        data["Net_Advances"] / (data["Advances"] + data["Declines"])
    ) * 1000
    data["19_EMA"] = ta.EMA(data["Ratio_Adjusted"], timeperiod=19)
    data["39_EMA"] = ta.EMA(data["Ratio_Adjusted"], timeperiod=39)
    data["RANA"] = (
        (data["Advances"] - data["Declines"]) / (data["Advances"] + data["Declines"]) * 1000
    )

    # Finding the TRIN Value
    data["ad_ratio"] = data["Advances"].divide(data["Declines"])  # AD Ratio
    data["ad_vol"] = data["adv_vol"].divide(data["dec_vol"])  # AD Volume Ratio
    data["TRIN"] = data["ad_ratio"].divide(data["adv_vol"])  # TRIN Value

    # Function to calculate Force Index
    def ForceIndex(data, n):
        ForceIndex = pd.Series(df["Adj Close"].diff(n) * df["Volume"], name="ForceIndex")
        data = data.join(ForceIndex)
        return data

    # Calculate Force Index
    n = 10
    ForceIndex = ForceIndex(df, n)
    ForceIndex = ForceIndex["ForceIndex"]

    # Market Price Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name='Close Price'))
    fig.update_layout(title="Market Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Close Price")
    st.plotly_chart(fig)

    # Force Index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=ForceIndex, mode='lines', name='Force Index'))
    fig.update_layout(title="Force Index",
                    xaxis_title="Date",
                    yaxis_title="Force Index")
    st.plotly_chart(fig)


    # Function to calculate Chaikin Oscillator
    def Chaikin(data):
        money_flow_volume = (
            (2 * data["Adj Close"] - data["High"] - data["Low"])
            / (data["High"] - data["Low"])
            * data["Volume"]
        )
        ad = money_flow_volume.cumsum()
        Chaikin = pd.Series(
            ad.ewm(com=(3 - 1) / 2).mean() - ad.ewm(com=(10 - 1) / 2).mean(), name="Chaikin"
        )
        #data["Chaikin"] = data.join(Chaikin)
        data["Chaikin"] = Chaikin
        return data


    # Calculate Chaikin Oscillator
    Chaikin(df)


    # Chaikin Oscillator Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Chaikin"], mode='lines', name='Chaikin Oscillator'))
    fig.update_layout(title="Chaikin Oscillator",
                    xaxis_title="Date",
                    yaxis_title="Chaikin Oscillator")
    
    st.plotly_chart(fig)


    # Calculate Cumulative Volume Index
    data["CVI"] = data["Net_Advances"].shift(1) + (data["Advances"] - data["Declines"])

    # Cumulative Volume Index Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["CVI"], mode='lines', name='Cumulative Volume Index'))
    fig.update_layout(title="Cumulative Volume Index",
                    xaxis_title="Date",
                    yaxis_title="CVI")
    
    st.plotly_chart(fig)

