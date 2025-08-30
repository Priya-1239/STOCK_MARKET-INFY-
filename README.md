# Stock Prediction Web Application

A Flask-based web application for stock price analysis and prediction using Yahoo Finance data.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Initialize the database: `python init_db.py`
3. Run the application: `python app.py`
4. Access at `http://localhost:5000`

## Features
- User authentication (signup/login)
- Stock data analysis with technical indicators
- Interactive charts for RSI, MACD, Williams %R, and more
- Future price predictions
- CSV export with metadata

## Usage
1. Sign up or log in (default: admin/admin).
2. Enter a stock symbol and date range on the dashboard.
3. View comprehensive analysis, charts, and predictions.
4. Download results as CSV or predict prices for a specific future date.