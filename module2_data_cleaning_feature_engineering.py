# module2_data_cleaning_feature_engineering.py
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataCleaningAndFeatureEngineering:
    def __init__(self):
        self.processed_data = None
        self.cleaning_report = {}
    
    def clean_data(self, df):
        """
        Comprehensive data cleaning including missing values, outliers, and non-trading days
        """
        cleaning_report = {
            'original_shape': df.shape,
            'missing_values_before': df.isnull().sum().to_dict(),
            'outliers_detected': {},
            'non_trading_days_filled': 0
        }
        
        # Create a copy to avoid modifying original data
        df_clean = df.copy()
        
        # 1. Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # 2. Detect and handle outliers
        df_clean, outliers_info = self._handle_outliers(df_clean)
        cleaning_report['outliers_detected'] = outliers_info
        
        # 3. Handle non-trading days (weekends, holidays)
        df_clean = self._handle_non_trading_days(df_clean)
        
        # 4. Ensure data consistency
        df_clean = self._ensure_data_consistency(df_clean)
        
        cleaning_report['final_shape'] = df_clean.shape
        cleaning_report['missing_values_after'] = df_clean.isnull().sum().to_dict()
        
        self.cleaning_report = cleaning_report
        return df_clean
    
    def _handle_missing_values(self, df):
        """Handle missing values using appropriate methods"""
        # Forward fill for stock prices (carry last observation forward)
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # Volume: fill with median of recent 5 days
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(df['Volume'].rolling(5).median())
            df['Volume'] = df['Volume'].fillna(df['Volume'].median())
        
        return df
    
    def _handle_outliers(self, df, z_threshold=3):
        """Detect and handle outliers using Z-score and IQR methods"""
        outliers_info = {}
        
        for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if column in df.columns:
                # Z-score method
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                z_outliers = np.where(z_scores > z_threshold)[0]
                
                # IQR method
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                iqr_outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | 
                                 (df[column] > (Q3 + 1.5 * IQR))].index
                
                outliers_info[column] = {
                    'z_score_outliers': len(z_outliers),
                    'iqr_outliers': len(iqr_outliers)
                }
                
                # Replace extreme outliers with median
                if len(iqr_outliers) > 0 and len(iqr_outliers) < len(df) * 0.05:  # Less than 5% of data
                    median_val = df[column].median()
                    df.loc[iqr_outliers, column] = median_val
        
        return df, outliers_info
    
    def _handle_non_trading_days(self, df):
        """Handle non-trading days by forward filling"""
        # Ensure we have a complete date range
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df_reindexed = df.reindex(full_range)
        
        # Forward fill for non-trading days
        df_reindexed = df_reindexed.fillna(method='ffill')
        
        # Only keep business days for final output
        return df_reindexed[df_reindexed.index.dayofweek < 5]  # Monday=0, Friday=4
    
    def _ensure_data_consistency(self, df):
        """Ensure logical consistency in stock data"""
        # High should be >= max(Open, Close)
        # Low should be <= min(Open, Close)
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Fix High values
            df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
            
            # Fix Low values  
            df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
        
        return df
    
    def create_technical_features(self, df):
        """
        Create comprehensive technical analysis features
        """
        df_features = df.copy()
        
        # Basic price features
        df_features['Daily_Return'] = df_features['Close'].pct_change()
        df_features['Log_Return'] = np.log(df_features['Close'] / df_features['Close'].shift(1))
        df_features['Price_Range'] = df_features['High'] - df_features['Low']
        df_features['Price_Change'] = df_features['Close'] - df_features['Open']
        
        # Moving Averages (multiple periods)
        for period in [5, 10, 20, 50, 100, 200]:
            df_features[f'MA_{period}'] = df_features['Close'].rolling(period).mean()
            df_features[f'MA_{period}_Signal'] = np.where(
                df_features['Close'] > df_features[f'MA_{period}'], 1, -1
            )
        
        # Exponential Moving Averages
        for period in [12, 26]:
            df_features[f'EMA_{period}'] = df_features['Close'].ewm(span=period).mean()
        
        # RSI (Relative Strength Index)
        df_features['RSI_14'] = self._calculate_rsi(df_features['Close'], 14)
        df_features['RSI_7'] = self._calculate_rsi(df_features['Close'], 7)
        
        # MACD
        df_features['MACD'], df_features['MACD_Signal'], df_features['MACD_Histogram'] = self._calculate_macd(df_features['Close'])
        
        # Bollinger Bands
        df_features['BB_Upper'], df_features['BB_Middle'], df_features['BB_Lower'] = self._calculate_bollinger_bands(df_features['Close'])
        df_features['BB_Width'] = df_features['BB_Upper'] - df_features['BB_Lower']
        df_features['BB_Position'] = (df_features['Close'] - df_features['BB_Lower']) / (df_features['BB_Upper'] - df_features['BB_Lower'])
        
        # Williams %R
        for period in [14, 21]:
            df_features[f'Williams_R_{period}'] = self._calculate_williams_r(
                df_features['High'], df_features['Low'], df_features['Close'], period
            )
        
        # Momentum indicators
        for period in [5, 10, 20]:
            df_features[f'Momentum_{period}'] = df_features['Close'].diff(period)
            df_features[f'ROC_{period}'] = df_features['Close'].pct_change(period) * 100
        
        # Volatility measures
        df_features['Volatility_10'] = df_features['Daily_Return'].rolling(10).std()
        df_features['Volatility_30'] = df_features['Daily_Return'].rolling(30).std()
        
        # Volume indicators
        if 'Volume' in df_features.columns:
            df_features['Volume_MA_10'] = df_features['Volume'].rolling(10).mean()
            df_features['Volume_Ratio'] = df_features['Volume'] / df_features['Volume_MA_10']
            df_features['Price_Volume'] = df_features['Close'] * df_features['Volume']
        
        # Support and Resistance levels
        df_features['Support'] = df_features['Low'].rolling(20).min()
        df_features['Resistance'] = df_features['High'].rolling(20).max()
        
        # Trend indicators
        df_features['Trend_5'] = np.where(df_features['Close'] > df_features['Close'].shift(5), 1, -1)
        df_features['Trend_20'] = np.where(df_features['MA_20'] > df_features['MA_20'].shift(1), 1, -1)
        
        self.processed_data = df_features
        return df_features
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def get_cleaning_report(self):
        """Get detailed cleaning report"""
        return self.cleaning_report
    
    def export_processed_data(self, filename=None):
        """Export processed data with all features"""
        if self.processed_data is not None:
            if filename is None:
                filename = f"processed_stock_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.processed_data.to_csv(filename)
            return filename
        else:
            raise ValueError("No processed data available. Run create_technical_features first.")