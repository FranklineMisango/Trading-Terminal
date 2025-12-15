from langchain_core.tools import tool
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import datetime as dt
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

@tool
def tool_arima_time_series(start_date : dt.date, end_date : dt.date, ticker: str):
    '''This tool allows you to perform time series analysis using ARIMA'''
 # Fetch data
    data = yf.download(ticker, start_date, end_date)

    # TradingView Chart Integration (Hybrid: JS-based Trading Chart with Custom Data)
    st.subheader(f"TradingView-Style Chart for {ticker.upper()} (Powered by Your Python Data)")
    # Prepare data for TradingView Lightweight Charts (custom OHLCV from yfinance)
    data_ohlcv = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    data_ohlcv['time'] = (data_ohlcv.index.astype(int) // 10**9).astype(int)  # Convert to Unix timestamp
    chart_data = data_ohlcv[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
    chart_data_json = str(chart_data).replace("'", '"')  # Simple JSON-like string for embedding

    tradingview_html = f"""
    <div id="tvchart"></div>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <script>
        const chart = LightweightCharts.createChart(document.getElementById('tvchart'), {{
            width: 980,
            height: 610,
            layout: {{
                backgroundColor: '#ffffff',
                textColor: '#333',
            }},
            grid: {{
                vertLines: {{
                    color: '#e1ecf2',
                }},
                horzLines: {{
                    color: '#e1ecf2',
                }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            rightPriceScale: {{
                borderColor: '#cccccc',
            }},
            timeScale: {{
                borderColor: '#cccccc',
            }},
        }});

        const candlestickSeries = chart.addCandlestickSeries({{
            upColor: '#00C853',
            downColor: '#FF1744',
            borderVisible: false,
            wickUpColor: '#00C853',
            wickDownColor: '#FF1744',
        }});

        // Load custom data from Python
        const data = {chart_data_json};
        candlestickSeries.setData(data);

        // Add volume series
        const volumeSeries = chart.addHistogramSeries({{
            color: '#26a69a',
            priceFormat: {{
                type: 'volume',
            }},
            priceScaleId: '',
            scaleMargins: {{
                top: 0.7,
                bottom: 0,
            }},
        }});
        volumeSeries.setData(data.map(d => ({{
            time: d.time,
            value: d.volume,
            color: d.close > d.open ? '#00C853' : '#FF1744'
        }})));
    </script>
    """
    components.html(tradingview_html, height=650)

    # Original Plotly Chart for Custom Analysis
    st.subheader(f"Custom Plotly Chart for {ticker.upper()}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Prices'))
    fig.update_layout(title=f"{ticker} Closing Price", xaxis_title='Dates', yaxis_title='Close Prices')
    # Display the closing price plot using Streamlit
    st.plotly_chart(fig)

    # Test for stationarity
    def test_stationarity(timeseries):
        # Rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        
        fig = go.Figure()
        # Add traces for the original series, rolling mean, and rolling standard deviation
        fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries.values, mode='lines', name='Original', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=timeseries.index, y=rolmean.values, mode='lines', name='Rolling Mean', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=timeseries.index, y=rolstd.values, mode='lines', name='Rolling Std', line=dict(color='black')))

        # Update layout
        fig.update_layout(
            title='Rolling Mean & Standard Deviation',
            xaxis_title='Dates',
            yaxis_title='Values'
        )

        # Display the rolling statistics plot using Streamlit
        st.plotly_chart(fig)
                            
        # Dickey-Fuller Test
        st.write("Results of Dickey Fuller Test")
        adft = adfuller(timeseries, autolag='AIC')
        output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
        for key, value in adft[4].items():
            output[f'critical value ({key})'] = value
        st.write(output)
        
    test_stationarity(data['Close'])

    # Decompose the series
    def days_between_dates(dt1, dt2):
        date_format = "%d/%m/%Y"
        a = time.mktime(time.strptime(dt1, date_format))
        b = time.mktime(time.strptime(dt2, date_format))
        delta = b - a
        return int(delta / 86400)

    
    #difference = days_between_dates(start_date, end_date)
    #st.write(difference)


    def business_days_between_dates(start_date, end_date):
        business_days = np.busday_count(start_date, end_date)
        return business_days

    # Example usage:
    business_days = business_days_between_dates(start_date, end_date)
    st.write("Business days between the two dates:", business_days)


    result = seasonal_decompose(data['Close'], model='multiplicative', period=int(business_days/2))
    fig = result.plot()  
    fig.set_size_inches(16, 9)

    st.write(data['Close'])
    # Log transform
    df_log = data['Close']
    #df_log = np.log(data['Close'])
    #st.write(df_log)
    moving_avg = df_log.rolling(12).mean()
    std_dev = df_log.rolling(12).std()

    # Plot moving average
    fig = go.Figure()

    # Add traces for the moving average and standard deviation
    fig.add_trace(go.Scatter(x=std_dev.index, y=std_dev.values, mode='lines', name='Standard Deviation', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=moving_avg.index, y=moving_avg.values, mode='lines', name='Mean', line=dict(color='red')))

    # Update layout
    fig.update_layout(
        title='Moving Average',
        xaxis_title='Dates',
        yaxis_title='Values',
        legend=dict(x=0, y=1)
    )

    # Display the moving average plot using Streamlit
    st.plotly_chart(fig)

    # Split data into train and test sets
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

    # Build and fit ARIMA model
    model = ARIMA(train_data, order=(3, 1, 2))  
    fitted = model.fit()  
    st.write(fitted.summary())

    # Forecast
    
    #fc, se, conf = fitted.forecast(len(test_data), alpha=0.05)
    fc = fitted.forecast(len(test_data), alpha=0.05)

    # Plot forecast
    fc_series = pd.Series(fc, index=test_data.index)
    st.write(fc_series)
    lower_series = pd.Series(index=test_data.index)
    upper_series = pd.Series(index=test_data.index)

    # Create stock price prediction plot with Plotly
    fig = go.Figure()

    # Plot training data
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data.values, mode='lines', name='Training'))

    # Plot actual price
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data.values, mode='lines', name='Actual Price', line=dict(color='blue')))

    # Plot predicted price
    fig.add_trace(go.Scatter(x=test_data.index, y=fc, mode='lines', name='Predicted Price', line=dict(color='orange')))

    # Add shaded region for confidence interval
    fig.add_trace(go.Scatter(x=lower_series.index, y=lower_series.values, mode='lines', fill=None, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=lower_series.index, y=upper_series.values, mode='lines', fill='tonexty', line=dict(color='rgba(0,0,0,0)'), name='Confidence Interval'))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price Prediction',
        xaxis_title='Time',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Display the stock price prediction plot using Streamlit
    st.plotly_chart(fig)

    # Performance metrics
    mse = mean_squared_error(test_data, fc)
    mae = mean_absolute_error(test_data, fc)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
    st.write(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

    # Auto ARIMA
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0, test='adf', max_p=3, max_q=3, m=1, d=None, seasonal=False, start_P=0, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    plt.show() 


def arima_time_series(start_date, end_date, ticker):
    '''This tool allows you to perform time series analysis using ARIMA'''
    # Fetch data
    data = yf.download(ticker, start_date, end_date)

    # Create closing price plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Prices'))
    fig.update_layout(title=f"{ticker} Closing Price", xaxis_title='Dates', yaxis_title='Close Prices')
    # Display the closing price plot using Streamlit
    st.plotly_chart(fig)

    # Test for stationarity
    def test_stationarity(timeseries):
        # Rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        
        fig = go.Figure()
        # Add traces for the original series, rolling mean, and rolling standard deviation
        fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries.values, mode='lines', name='Original', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=timeseries.index, y=rolmean.values, mode='lines', name='Rolling Mean', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=timeseries.index, y=rolstd.values, mode='lines', name='Rolling Std', line=dict(color='black')))

        # Update layout
        fig.update_layout(
            title='Rolling Mean & Standard Deviation',
            xaxis_title='Dates',
            yaxis_title='Values'
        )

        # Display the rolling statistics plot using Streamlit
        st.plotly_chart(fig)
                            
        # Dickey-Fuller Test
        st.write("Results of Dickey Fuller Test")
        adft = adfuller(timeseries, autolag='AIC')
        output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
        for key, value in adft[4].items():
            output[f'critical value ({key})'] = value
        st.write(output)
        
    test_stationarity(data['Close'])

    # Decompose the series
    def days_between_dates(dt1, dt2):
        date_format = "%d/%m/%Y"
        a = time.mktime(time.strptime(dt1, date_format))
        b = time.mktime(time.strptime(dt2, date_format))
        delta = b - a
        return int(delta / 86400)

    
    #difference = days_between_dates(start_date, end_date)
    #st.write(difference)


    def business_days_between_dates(start_date, end_date):
        business_days = np.busday_count(start_date, end_date)
        return business_days

    # Example usage:
    business_days = business_days_between_dates(start_date, end_date)
    st.write("Business days between the two dates:", business_days)


    result = seasonal_decompose(data['Close'], model='multiplicative', period=int(business_days/2))
    fig = result.plot()  
    fig.set_size_inches(16, 9)

    st.write(data['Close'])
    # Log transform
    df_log = data['Close']
    #df_log = np.log(data['Close'])
    #st.write(df_log)
    moving_avg = df_log.rolling(12).mean()
    std_dev = df_log.rolling(12).std()

    # Plot moving average
    fig = go.Figure()

    # Add traces for the moving average and standard deviation
    fig.add_trace(go.Scatter(x=std_dev.index, y=std_dev.values, mode='lines', name='Standard Deviation', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=moving_avg.index, y=moving_avg.values, mode='lines', name='Mean', line=dict(color='red')))

    # Update layout
    fig.update_layout(
        title='Moving Average',
        xaxis_title='Dates',
        yaxis_title='Values',
        legend=dict(x=0, y=1)
    )

    # Display the moving average plot using Streamlit
    st.plotly_chart(fig)

    # Split data into train and test sets
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

    # Build and fit ARIMA model
    model = ARIMA(train_data, order=(3, 1, 2))  
    fitted = model.fit()  
    st.write(fitted.summary())

    # Forecast
    
    #fc, se, conf = fitted.forecast(len(test_data), alpha=0.05)
    fc = fitted.forecast(len(test_data), alpha=0.05)

    # Plot forecast
    fc_series = pd.Series(fc, index=test_data.index)
    st.write(fc_series)
    lower_series = pd.Series(index=test_data.index)
    upper_series = pd.Series(index=test_data.index)

    # Create stock price prediction plot with Plotly
    fig = go.Figure()

    # Plot training data
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data.values, mode='lines', name='Training'))

    # Plot actual price
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data.values, mode='lines', name='Actual Price', line=dict(color='blue')))

    # Plot predicted price
    fig.add_trace(go.Scatter(x=test_data.index, y=fc, mode='lines', name='Predicted Price', line=dict(color='orange')))

    # Add shaded region for confidence interval
    fig.add_trace(go.Scatter(x=lower_series.index, y=lower_series.values, mode='lines', fill=None, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=lower_series.index, y=upper_series.values, mode='lines', fill='tonexty', line=dict(color='rgba(0,0,0,0)'), name='Confidence Interval'))

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price Prediction',
        xaxis_title='Time',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Display the stock price prediction plot using Streamlit
    st.plotly_chart(fig)

    # Performance metrics
    mse = mean_squared_error(test_data, fc)
    mae = mean_absolute_error(test_data, fc)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
    st.write(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

    # Auto ARIMA
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0, test='adf', max_p=3, max_q=3, m=1, d=None, seasonal=False, start_P=0, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    plt.show() 