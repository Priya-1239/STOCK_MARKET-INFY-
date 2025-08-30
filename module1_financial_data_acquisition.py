# module1_financial_data_acquisition.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

class FinancialDataAcquisition:
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def get_stock_data(self, symbol, start_date, end_date, use_cache=True):
        """
        Connect to Yahoo Finance API and fetch stock data
        Implements caching for better performance
        """
        cache_filename = f"{self.cache_dir}/{symbol}_{start_date}_{end_date}.csv"
        
        # Check cache first
        if use_cache and os.path.exists(cache_filename):
            try:
                df = pd.read_csv(cache_filename, index_col=0, parse_dates=True)
                print(f"Data loaded from cache: {cache_filename}")
                return df
            except Exception as e:
                print(f"Cache read error: {e}, fetching fresh data...")
        
        try:
            # Fetch from Yahoo Finance
            print(f"Fetching fresh data for {symbol} from {start_date} to {end_date}")
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Fix MultiIndex columns if present
            if df.columns.nlevels > 1:
                df.columns = df.columns.droplevel(1)
            
            # Cache the data
            df.to_csv(cache_filename)
            print(f"Data cached to: {cache_filename}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def get_multiple_stocks(self, symbols, start_date, end_date):
        """Get data for multiple stocks"""
        all_data = {}
        for symbol in symbols:
            try:
                all_data[symbol] = self.get_stock_data(symbol, start_date, end_date)
            except Exception as e:
                print(f"Failed to get data for {symbol}: {e}")
        return all_data
    
    def export_to_csv(self, df, symbol, filename=None):
        """Export data to CSV with metadata"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_stock_data_{timestamp}.csv"
        
        # Add metadata
        metadata = {
            'symbol': symbol,
            'export_date': datetime.now().isoformat(),
            'data_points': len(df),
            'date_range': f"{df.index[0].date()} to {df.index[-1].date()}"
        }
        
        # Save metadata as comment in CSV
        with open(filename, 'w') as f:
            f.write(f"# Stock Data Export - {json.dumps(metadata)}\n")
            df.to_csv(f)
        
        return filename
    
    def get_real_time_price(self, symbol):
        """Get current real-time price"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'change': info.get('currentPrice', 0) - info.get('previousClose', 0),
                'change_percent': ((info.get('currentPrice', 0) - info.get('previousClose', 0)) / info.get('previousClose', 1)) * 100
            }
        except Exception as e:
            print(f"Error getting real-time data: {e}")
            return None
    
    def validate_symbol(self, symbol):
        """Validate if stock symbol exists"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return 'regularMarketPrice' in info or 'currentPrice' in info
        except:
            return False