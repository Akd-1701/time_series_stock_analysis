
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

# Streamlit app title
st.title('Stock Market Time Series Forecasting App')

# Sidebar for data upload
st.sidebar.header('Data Upload')
uploaded_file = st.sidebar.file_uploader('Upload stock_data.csv', type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    st.write('Data Preview:')
    st.dataframe(df.head())
    stocks = df['Name'].unique()
    selected_stock = st.sidebar.selectbox('Select Stock', stocks)
    stock_data = df[df['Name'] == selected_stock]['Close']
else:
    st.write('Please upload stock_data.csv with Date, Close, and Name columns.')
    st.stop()

# Sidebar for model selection
st.sidebar.header('Model Selection')
model_type = st.sidebar.selectbox('Choose Model', ['ARIMA', 'SARIMA', 'Prophet'])

# ACF/PACF plots for parameter selection
st.subheader('ACF and PACF Plots (Parameter Guidance)')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(stock_data, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)')
plot_pacf(stock_data, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')
st.pyplot(fig)
st.write('Use ACF to estimate q (MA order) and PACF to estimate p (AR order).')

# Parameter inputs based on model selection
if model_type == 'ARIMA':
    st.sidebar.subheader('ARIMA Parameters')
    p = st.sidebar.number_input('p (AR order from PACF)', min_value=0, value=1)
    d = st.sidebar.number_input('d (Differencing)', min_value=0, value=1)
    q = st.sidebar.number_input('q (MA order from ACF)', min_value=0, value=1)
elif model_type == 'SARIMA':
    st.sidebar.subheader('SARIMA Parameters')
    p = st.sidebar.number_input('p (AR order from PACF)', min_value=0, value=1)
    d = st.sidebar.number_input('d (Differencing)', min_value=0, value=1)
    q = st.sidebar.number_input('q (MA order from ACF)', min_value=0, value=1)
    P = st.sidebar.number_input('P (Seasonal AR)', min_value=0, value=1)
    D = st.sidebar.number_input('D (Seasonal Differencing)', min_value=0, value=1)
    Q = st.sidebar.number_input('Q (Seasonal MA)', min_value=0, value=1)
    s = st.sidebar.number_input('s (Seasonal Period)', min_value=1, value=5)  # Default to trading days
elif model_type == 'Prophet':
    st.sidebar.subheader('Prophet Parameters')
    growth = st.sidebar.selectbox('Growth Model', ['linear', 'logistic'])
    seasonality_mode = st.sidebar.selectbox('Seasonality Mode', ['additive', 'multiplicative'])

# Forecast horizon
forecast_horizon = st.sidebar.number_input('Forecast Horizon (Days)', min_value=1, value=30)

# Button to run the forecast
if st.sidebar.button('Run Forecast'):
    try:
        if model_type == 'ARIMA':
            model = ARIMA(stock_data, order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_horizon)
            forecast_index = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
        elif model_type == 'SARIMA':
            model = SARIMAX(stock_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=forecast_horizon)
            forecast_index = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
        elif model_type == 'Prophet':
            df_prophet = stock_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
            model = Prophet(growth=growth, seasonality_mode=seasonality_mode)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=forecast_horizon, freq='B')
            forecast_df = model.predict(future)
            forecast = forecast_df['yhat'].tail(forecast_horizon)
            forecast_index = future['ds'].tail(forecast_horizon)

        # Plot historical data and forecast
        st.subheader('Forecast Visualization')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(stock_data, label='Historical Data')
        ax.plot(forecast_index, forecast, label='Forecast', color='red')
        ax.set_title(f'Forecast for {selected_stock} using {model_type}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        st.pyplot(fig)

        # Display forecast values
        st.subheader('Forecast Values')
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f'Error: {str(e)}')

# Plot historical data alone
st.subheader('Historical Closing Prices')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(stock_data, label='Close Price')
ax.set_title(f'Closing Prices for {selected_stock}')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
st.pyplot(fig)