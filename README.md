# NEPSE Index Dashboard

A comprehensive stock market dashboard for the Nepal Stock Exchange (NEPSE) index, built with Streamlit and Plotly.

## Features

- **Interactive Price Charts**: Candlestick and line chart options
- **Technical Indicators**:
  - Simple Moving Averages (SMA 20, SMA 50)
  - Exponential Moving Averages (EMA 20, EMA 50)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Average True Range (ATR)
- **Price Forecasting**:
  - ETS (Holt-Winters) model
  - ARIMA model
- **Dark/Light Theme Support**
- **Data Export**: Download filtered data as CSV
- **Statistics Summary**: Key metrics and daily returns distribution

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Place your NEPSE data file at:
```
/Users/shitalbhandary/Downloads/nepse_adb_data.csv
```

The data file should contain the following columns:
- BUSINESS_DATE
- OPEN_PRICE
- HIGH_PRICE
- LOW_PRICE
- CLOSE_PRICE
- TOTAL_TRADED_QUANTITY
- TOTAL_TRADES
- TOTAL_TRADED_VALUE

## Running the Dashboard

```bash
streamlit run nepse_dashboard.py
```

The dashboard will open in your default web browser.

## Usage

- **Date Range**: Select the date range from the sidebar to filter data
- **Chart Type**: Choose between Candlestick and Line charts
- **Indicators**: Toggle various technical indicators on/off
- **Forecasting**: Enable forecast and select model type (ETS, ARIMA, or Both)
- **Theme**: Switch between dark and light modes

## Requirements

- streamlit
- pandas
- plotly
- numpy
- ta
- statsmodels
