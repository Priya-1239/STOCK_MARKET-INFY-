# Set Matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import os
import json
import logging
import tempfile
import uuid

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import our modules
from module1_financial_data_acquisition import FinancialDataAcquisition
from module2_data_cleaning_feature_engineering import DataCleaningAndFeatureEngineering

app = Flask(__name__)
app.secret_key = b'\xffN\xdf\x0c\xc0\xba\xf4D\x84\xe9i]=\xceTi\xe7\x17A\xd0\x159\x0fb'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

# Initialize modules
financial_data = FinancialDataAcquisition()
data_processor = DataCleaningAndFeatureEngineering()

# Simple user login (use admin/admin)
users = {"admin": "admin"}

@app.route("/")
def index():
    return redirect(url_for("signup"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users:
            flash("Username already exists! Use login.", "danger")
            return redirect(url_for("login"))
        users[username] = password
        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users and users[username] == password:
            session["logged_in"] = True
            session["username"] = username
            session.permanent = True
            return redirect(url_for("main"))
        else:
            flash("Invalid credentials. Try again.", "danger")
    return render_template("login.html")

@app.route("/main")
def main():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("main.html", username=session.get("username"))

def perform_comprehensive_analysis(df, symbol):
    logger.debug(f"Performing analysis for {symbol}")
    try:
        if len(df) < 30:
            raise ValueError("Insufficient data points for analysis (minimum 30 required)")
        
        latest_data = df.iloc[-1]
        rsi_value = latest_data['RSI_14']
        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
        rsi_action = "Sell" if rsi_value > 70 else "Buy" if rsi_value < 30 else "Hold"
        
        macd_status = "Bullish" if latest_data['MACD'] > latest_data['MACD_Signal'] else "Bearish"
        macd_action = "Buy" if macd_status == "Bullish" else "Sell"
        
        williams_r_value = latest_data['Williams_R_14']
        williams_r_status = "Overbought" if williams_r_value > -20 else "Oversold" if williams_r_value < -80 else "Neutral"
        williams_r_action = "Sell" if williams_r_value > -20 else "Buy" if williams_r_value < -80 else "Hold"
        
        bb_position = latest_data['BB_Position']
        bb_status = "Above Upper" if bb_position > 1 else "Below Lower" if bb_position < 0 else "Within Bands"
        bb_action = "Sell" if bb_position > 1 else "Buy" if bb_position < 0 else "Hold"
        
        ma_analysis = {}
        for period in [5, 10, 20, 50]:
            ma_key = f'MA_{period}'
            if latest_data['Close'] > latest_data[ma_key]:
                ma_analysis[ma_key] = {'status': 'Above', 'action': 'Buy'}
            else:
                ma_analysis[ma_key] = {'status': 'Below', 'action': 'Sell'}
        
        signals = [rsi_action, macd_action, williams_r_action, bb_action]
        buy_count = signals.count("Buy")
        sell_count = signals.count("Sell")
        overall_recommendation = "Buy" if buy_count >= 3 else "Sell" if sell_count >= 3 else "Hold"
        confidence = 80 if buy_count >= 3 or sell_count >= 3 else 60
        
        analysis = {
            'rsi': {'value': float(rsi_value), 'signal': {'status': rsi_status, 'action': rsi_action}, 'description': 'Relative Strength Index (14-day)'},
            'macd': {'value': float(latest_data['MACD']), 'signal_line': float(latest_data['MACD_Signal']), 'analysis': {'status': macd_status, 'action': macd_action}},
            'williams_r': {'value': float(williams_r_value), 'signal': {'status': williams_r_status, 'action': williams_r_action}},
            'bollinger_bands': {'position': float(bb_position), 'upper': float(latest_data['BB_Upper']), 'lower': float(latest_data['BB_Lower']), 'signal': {'status': bb_status, 'action': bb_action}},
            'moving_averages': ma_analysis,
            'overall': {'recommendation': overall_recommendation, 'confidence': f"{confidence}%"},
        }
        logger.debug(f"Analysis result: {analysis}")
        return analysis
    except Exception as e:
        logger.error(f"Error in perform_comprehensive_analysis: {str(e)}", exc_info=True)
        raise

def create_comprehensive_charts(df, symbol):
    logger.debug(f"Creating charts for {symbol}")
    chart_urls = {}
    
    def plot_to_base64():
        try:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"Error converting plot to base64: {str(e)}")
            raise
    
    try:
        if len(df) < 30:
            raise ValueError("Insufficient data points for chart generation (minimum 30 required)")
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Close Price', color='blue')
        plt.plot(df.index, df['MA_20'], label='MA20', color='orange')
        plt.plot(df.index, df['BB_Upper'], label='BB Upper', color='red', linestyle='--')
        plt.plot(df.index, df['BB_Lower'], label='BB Lower', color='green', linestyle='--')
        plt.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
        plt.title(f'{symbol} Price Analysis')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()
        chart_urls['price_analysis'] = plot_to_base64()
        
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df['RSI_14'], label='RSI (14)', color='purple')
        plt.axhline(70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(30, color='green', linestyle='--', alpha=0.5)
        plt.title(f'{symbol} RSI Analysis')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid()
        chart_urls['rsi_analysis'] = plot_to_base64()
        
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df['MACD'], label='MACD', color='blue')
        plt.plot(df.index, df['MACD_Signal'], label='Signal Line', color='orange')
        plt.bar(df.index, df['MACD_Histogram'], label='Histogram', color='gray', alpha=0.3)
        plt.title(f'{symbol} MACD Analysis')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid()
        chart_urls['macd_analysis'] = plot_to_base64()
        
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df['Williams_R_14'], label='Williams %R (14)', color='green')
        plt.axhline(-20, color='red', linestyle='--', alpha=0.5)
        plt.axhline(-80, color='green', linestyle='--', alpha=0.5)
        plt.title(f'{symbol} Williams %R Analysis')
        plt.xlabel('Date')
        plt.ylabel('Williams %R')
        plt.legend()
        plt.grid()
        chart_urls['williams_r_analysis'] = plot_to_base64()
        
        plt.figure(figsize=(12, 4))
        plt.bar(df.index, df['Volume'], label='Volume', color='blue', alpha=0.5)
        plt.plot(df.index, df['Volume_MA_10'], label='Volume MA10', color='orange')
        plt.title(f'{symbol} Volume Analysis')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid()
        chart_urls['volume_analysis'] = plot_to_base64()
        
        logger.debug(f"Chart URLs generated: {chart_urls.keys()}")
        return chart_urls
    
    except Exception as e:
        logger.error(f"Error generating charts: {str(e)}", exc_info=True)
        raise

def generate_predictions(df, symbol):
    logger.debug(f"Generating predictions for {symbol}")
    try:
        if len(df) < 30:
            raise ValueError("Insufficient data points for predictions (minimum 30 required)")
        
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        current_price = df['Close'].iloc[-1]
        trend = (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / 30
        volatility = df['Volatility_30'].iloc[-1]
        ma_20 = df['MA_20'].iloc[-1]
        
        predictions = []
        for i, future_date in enumerate(future_dates):
            days_ahead = i + 1
            linear_pred = current_price + (trend * days_ahead)
            ma_reversion = current_price * 0.5 + ma_20 * 0.5
            vol_adjust = np.random.normal(0, volatility * np.sqrt(days_ahead))
            linear_pred = max(linear_pred + vol_adjust, 0)
            ma_reversion = max(ma_reversion + vol_adjust, 0)
            avg_pred = (linear_pred + ma_reversion) / 2
            predictions.append({
                'Date': future_date,
                'Linear_Regression': linear_pred,
                'MA_Reversion': ma_reversion,
                'Average_Prediction': avg_pred
            })
        
        predictions_df = pd.DataFrame(predictions)
        logger.debug(f"Predictions shape: {predictions_df.shape}")
        return predictions_df
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}", exc_info=True)
        raise

@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    stock_symbol = request.form.get("stock_symbol", "").upper().strip()
    start = request.form.get("start_date")
    end = request.form.get("end_date")

    logger.debug(f"Predict route called with symbol={stock_symbol}, start={start}, end={end}")

    if not stock_symbol or not start or not end:
        flash("Please provide stock symbol and date range.", "warning")
        return redirect(url_for("main"))

    try:
        # Validate date format and range
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        if start_date >= end_date:
            flash("Start date must be before end date.", "warning")
            return redirect(url_for("main"))
        if (end_date - start_date).days < 30:
            flash("Date range must be at least 30 days for analysis.", "warning")
            return redirect(url_for("main"))

        # Validate stock symbol
        logger.debug(f"Validating symbol: {stock_symbol}")
        try:
            is_valid = financial_data.validate_symbol(stock_symbol)
            logger.debug(f"Validation result for {stock_symbol}: {is_valid}")
            if not is_valid:
                ticker = yf.Ticker(stock_symbol)
                info = ticker.info
                logger.debug(f"yfinance info for {stock_symbol}: {info}")
                if not info or info.get('symbol') != stock_symbol:
                    flash(f"Invalid stock symbol: {stock_symbol}. Please try another.", "warning")
                    return redirect(url_for("main"))
                logger.debug(f"Fallback validation succeeded for {stock_symbol}")
                is_valid = True
        except Exception as e:
            logger.error(f"Error validating symbol {stock_symbol}: {str(e)}", exc_info=True)
            flash(f"Error validating stock symbol {stock_symbol}: {str(e)}", "warning")
            return redirect(url_for("main"))

        # Module 1: Financial Data Acquisition
        logger.debug(f"Fetching stock data for {stock_symbol} from {start} to {end}")
        df_raw = financial_data.get_stock_data(stock_symbol, start, end)
        logger.debug(f"Raw data shape: {df_raw.shape}, columns: {df_raw.columns.tolist()}")
        if df_raw.empty or len(df_raw) < 30:
            flash(f"Insufficient data retrieved for {stock_symbol}. Try a different symbol or date range.", "warning")
            return redirect(url_for("main"))
        
        # Module 2: Data Cleaning & Feature Engineering
        logger.debug(f"Cleaning data for {stock_symbol}")
        df_clean = data_processor.clean_data(df_raw)
        logger.debug(f"Clean data shape: {df_clean.shape}, columns: {df_clean.columns.tolist()}")
        if df_clean.empty or len(df_clean) < 30:
            flash(f"Insufficient data after cleaning for {stock_symbol}. Try a different date range.", "warning")
            return redirect(url_for("main"))
        
        logger.debug(f"Creating technical features for {stock_symbol}")
        df_features = data_processor.create_technical_features(df_clean)
        logger.debug(f"Features data shape: {df_features.shape}, columns: {df_features.columns.tolist()}")
        if df_features.empty or len(df_features) < 30:
            flash(f"Insufficient data after feature engineering for {stock_symbol}. Try a different date range.", "warning")
            return redirect(url_for("main"))
        
        # Validate DataFrame before saving
        required_columns = ['Close', 'MA_20', 'Volatility_30', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Williams_R_14', 'BB_Upper', 'BB_Lower', 'BB_Position']
        missing_columns = [col for col in required_columns if col not in df_features.columns]
        if missing_columns:
            logger.error(f"Missing required columns in df_features: {missing_columns}")
            flash(f"Data processing error: Missing columns {missing_columns}", "danger")
            return redirect(url_for("main"))
        
        # Get current price and real-time info
        current_price = df_features["Close"].iloc[-1]
        real_time_info = financial_data.get_real_time_price(stock_symbol)
        logger.debug(f"Current price: {current_price}, Real-time info: {real_time_info}")
        
        # Generate comprehensive analysis
        analysis = perform_comprehensive_analysis(df_features, stock_symbol)
        
        # Create visualizations
        chart_urls = create_comprehensive_charts(df_features, stock_symbol)
        
        # Generate predictions for next 7 days
        predictions = generate_predictions(df_features, stock_symbol)
        
        # Get cleaning report
        cleaning_report = data_processor.get_cleaning_report()
        logger.debug(f"Cleaning report: {cleaning_report}")
        
        # Store data for CSV export and future predictions using file
        try:
            os.makedirs("temp", exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create temp directory: {str(e)}", exc_info=True)
            flash(f"Error creating temporary directory: {str(e)}", "danger")
            return redirect(url_for("main"))
        
        unique_id = uuid.uuid4().hex
        data_filepath = os.path.join("temp", f"{stock_symbol}_{unique_id}_data.json")
        try:
            # Ensure datetime index is properly formatted
            df_features.index = df_features.index.map(lambda x: x.isoformat())
            json_data = df_features.to_json(date_format='iso', orient='index')
            with open(data_filepath, 'w') as f:
                f.write(json_data)
            logger.debug(f"Data saved to file: {data_filepath}, size={os.path.getsize(data_filepath)} bytes")
        except Exception as e:
            logger.error(f"Failed to write data file {data_filepath}: {str(e)}", exc_info=True)
            flash(f"Error saving data file: {str(e)}", "danger")
            return redirect(url_for("main"))
        
        analysis_filepath = os.path.join("temp", f"{stock_symbol}_{unique_id}_analysis.json")
        try:
            with open(analysis_filepath, 'w') as f:
                json.dump(analysis, f)
            logger.debug(f"Analysis saved to file: {analysis_filepath}, size={os.path.getsize(analysis_filepath)} bytes")
        except Exception as e:
            logger.error(f"Failed to write analysis file {analysis_filepath}: {str(e)}", exc_info=True)
            flash(f"Error saving analysis file: {str(e)}", "danger")
            return redirect(url_for("main"))
        
        session[f"{stock_symbol}_data_path"] = data_filepath
        session[f"{stock_symbol}_analysis_path"] = analysis_filepath
        session[f"{stock_symbol}_start"] = start
        session[f"{stock_symbol}_end"] = end
        
        logger.debug(f"Rendering results.html with: stock_symbol={stock_symbol}, "
                    f"chart_urls={list(chart_urls.keys())}, predictions_shape={predictions.shape}, "
                    f"analysis_keys={list(analysis.keys())}, cleaning_report_keys={list(cleaning_report.keys())}")

        return render_template("results.html",
                             stock_symbol=stock_symbol,
                             current_price=current_price,
                             real_time_info=real_time_info,
                             analysis=analysis,
                             chart_urls=chart_urls,
                             predictions=predictions,
                             cleaning_report=cleaning_report)

    except Exception as e:
        logger.error(f"Error processing predict request: {str(e)}", exc_info=True)
        flash(f"Error processing request: {str(e)}", "danger")
        return redirect(url_for("main"))

@app.route("/api/predict_future", methods=["POST"])
def predict_future():
    if not session.get("logged_in"):
        logger.error("Unauthorized access to predict_future: User not logged in")
        return jsonify({"error": "Not logged in"}), 401
    
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in predict_future")
            return jsonify({"error": "No JSON data provided"}), 400
        
        stock_symbol = data.get("stock_symbol", "").upper().strip()
        target_date = data.get("target_date")
        if not stock_symbol or not target_date:
            logger.error(f"Missing required fields: stock_symbol={stock_symbol}, target_date={target_date}")
            return jsonify({"error": "Missing stock_symbol or target_date"}), 400
        
        logger.debug(f"Predict future called with symbol={stock_symbol}, target_date={target_date}")
        
        # Validate target_date format
        try:
            target_date_dt = datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Invalid target_date format: {target_date}, error: {str(e)}")
            return jsonify({"error": f"Invalid target_date format: {target_date}"}), 400

        # Get stored data from file
        data_path = session.get(f"{stock_symbol}_data_path")
        logger.debug(f"Session data_path for {stock_symbol}: {data_path}")
        df_features = None
        if data_path and os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    data_json = f.read()
                logger.debug(f"Reading data file: {data_path}, size={os.path.getsize(data_path)} bytes")
                df_features = pd.read_json(data_json, orient='index')
                df_features.index = pd.to_datetime(df_features.index)
                logger.debug(f"Loaded data from file: {data_path}, shape={df_features.shape}, columns={df_features.columns.tolist()}")
                if df_features.empty or len(df_features) < 30:
                    logger.error(f"Loaded data is empty or insufficient: shape={df_features.shape}")
                    df_features = None  # Trigger fallback
            except Exception as e:
                logger.error(f"Failed to read data file {data_path}: {str(e)}", exc_info=True)
                df_features = None  # Trigger fallback
        else:
            logger.warning(f"No data file for {stock_symbol} at {data_path}. Attempting to re-fetch.")
        
        # Fallback: Re-fetch data if file not found or invalid
        if df_features is None or df_features.empty:
            logger.debug(f"Entering fallback for {stock_symbol}")
            start = session.get(f"{stock_symbol}_start")
            end = session.get(f"{stock_symbol}_end")
            if not start or not end:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            else:
                start_date = start
                end_date = end
            logger.debug(f"Fetching data for {stock_symbol} from {start_date} to {end_date}")
            df_raw = financial_data.get_stock_data(stock_symbol, start_date, end_date)
            logger.debug(f"Raw data shape: {df_raw.shape}, columns: {df_raw.columns.tolist()}")
            if df_raw.empty or len(df_raw) < 30:
                logger.error(f"Unable to fetch sufficient data for {stock_symbol}: shape={df_raw.shape}")
                return jsonify({"error": f"Unable to fetch sufficient data for {stock_symbol}. Please run analysis first."}), 400
            df_clean = data_processor.clean_data(df_raw)
            logger.debug(f"Clean data shape: {df_clean.shape}, columns: {df_clean.columns.tolist()}")
            if df_clean.empty or len(df_clean) < 30:
                logger.error(f"Insufficient data after cleaning for {stock_symbol}: shape={df_clean.shape}")
                return jsonify({"error": f"Insufficient data after cleaning for {stock_symbol}."}), 400
            df_features = data_processor.create_technical_features(df_clean)
            logger.debug(f"Features data shape: {df_features.shape}, columns: {df_features.columns.tolist()}")
            if df_features.empty or len(df_features) < 30:
                logger.error(f"Insufficient data after feature engineering for {stock_symbol}: shape={df_features.shape}")
                return jsonify({"error": f"Insufficient data after feature engineering for {stock_symbol}."}), 400
            try:
                os.makedirs("temp", exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create temp directory: {str(e)}", exc_info=True)
                return jsonify({"error": f"Failed to create temporary directory: {str(e)}"}), 500
            unique_id = uuid.uuid4().hex
            data_filepath = os.path.join("temp", f"{stock_symbol}_{unique_id}_data.json")
            try:
                df_features.index = df_features.index.map(lambda x: x.isoformat())
                json_data = df_features.to_json(date_format='iso', orient='index')
                with open(data_filepath, 'w') as f:
                    f.write(json_data)
                session[f"{stock_symbol}_data_path"] = data_filepath
                logger.debug(f"Fallback data fetched and saved to {data_filepath}: shape={df_features.shape}")
            except Exception as e:
                logger.error(f"Failed to write fallback data file {data_filepath}: {str(e)}", exc_info=True)
                return jsonify({"error": f"Failed to save fallback data: {str(e)}"}), 500
        
        if len(df_features) < 30:
            logger.error(f"Insufficient data points for prediction: shape={df_features.shape}")
            return jsonify({"error": "Insufficient data points for prediction (minimum 30 required)"}), 400
        
        last_date = df_features.index[-1]
        days_ahead = (target_date_dt - last_date).days
        
        if days_ahead <= 0:
            logger.error(f"Target date {target_date} is not in the future (last date: {last_date})")
            return jsonify({"error": "Target date must be in the future"}), 400
        
        if days_ahead > 365:
            logger.error(f"Target date {target_date} exceeds 1 year ahead (days_ahead: {days_ahead})")
            return jsonify({"error": "Cannot predict more than 1 year ahead"}), 400
        
        try:
            current_price = df_features['Close'].iloc[-1]
            ma_20 = df_features['MA_20'].iloc[-1]
            volatility = df_features['Volatility_30'].iloc[-1]
            trend = (df_features['Close'].iloc[-1] - df_features['Close'].iloc[-30]) / 30
        except KeyError as e:
            logger.error(f"Missing required column in df_features: {str(e)}", exc_info=True)
            return jsonify({"error": f"Missing required data column: {str(e)}"}), 500
        
        base_prediction = current_price + (trend * days_ahead)
        volatility_adjustment = np.random.normal(0, volatility * np.sqrt(days_ahead))
        predicted_price = max(base_prediction + volatility_adjustment, 0)
        
        current_analysis = perform_comprehensive_analysis(df_features, stock_symbol)
        
        confidence = max(50, 95 - (days_ahead * 0.5))
        
        result = {
            "stock_symbol": stock_symbol,
            "target_date": target_date_dt.strftime('%Y-%m-%d'),
            "days_ahead": days_ahead,
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "price_change": float(predicted_price - current_price),
            "price_change_percent": float((predicted_price - current_price) / current_price * 100),
            "confidence": f"{confidence:.1f}%",
            "recommendation": current_analysis['overall']['recommendation'],
            "volatility": float(volatility),
            "trend": "Upward" if trend > 0 else "Downward"
        }
        
        logger.debug(f"Future prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_future: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/download_csv/<stock_symbol>")
def download_csv(stock_symbol):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    
    try:
        data_path = session.get(f"{stock_symbol}_data_path")
        logger.debug(f"Downloading CSV for {stock_symbol}, data_path: {data_path}")
        if data_path and os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    data_json = f.read()
                logger.debug(f"Reading data file: {data_path}, size={os.path.getsize(data_path)} bytes")
                try:
                    df = pd.read_json(data_json, orient='index')
                    df.index = pd.to_datetime(df.index)
                    logger.debug(f"Loaded data from file: {data_path}, shape={df.shape}, columns={df.columns.tolist()}")
                except ValueError as e:
                    logger.error(f"Failed to parse JSON data: {str(e)}", exc_info=True)
                    # Attempt to inspect JSON content
                    try:
                        json_data = json.loads(data_json)
                        logger.debug(f"JSON content sample: {json.dumps(json_data, indent=2)[:1000]}")
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON format in {data_path}")
                    flash(f"Error reading data file: Invalid data format", "danger")
                    return redirect(url_for("main"))
            except Exception as e:
                logger.error(f"Failed to read data file {data_path}: {str(e)}", exc_info=True)
                flash(f"Error reading data file: {str(e)}", "danger")
                return redirect(url_for("main"))
        else:
            logger.warning(f"No data file for {stock_symbol}. Attempting to re-fetch.")
            start = session.get(f"{stock_symbol}_start")
            end = session.get(f"{stock_symbol}_end")
            if not start or not end:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            else:
                start_date = start
                end_date = end
            logger.debug(f"Fetching data for {stock_symbol} from {start_date} to {end_date}")
            df_raw = financial_data.get_stock_data(stock_symbol, start_date, end_date)
            logger.debug(f"Raw data shape: {df_raw.shape}, columns: {df_raw.columns.tolist()}")
            if df_raw.empty or len(df_raw) < 30:
                flash(f"Unable to fetch sufficient data for {stock_symbol}. Please run analysis first.", "warning")
                return redirect(url_for("main"))
            df_clean = data_processor.clean_data(df_raw)
            logger.debug(f"Clean data shape: {df_clean.shape}, columns: {df_clean.columns.tolist()}")
            if df_clean.empty or len(df_clean) < 30:
                flash(f"Insufficient data after cleaning for {stock_symbol}.", "warning")
                return redirect(url_for("main"))
            df = data_processor.create_technical_features(df_clean)
            logger.debug(f"Features data shape: {df.shape}, columns: {df.columns.tolist()}")
            if df.empty or len(df) < 30:
                flash(f"Insufficient data after feature engineering for {stock_symbol}.", "warning")
                return redirect(url_for("main"))
            try:
                os.makedirs("temp", exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create temp directory: {str(e)}", exc_info=True)
                flash(f"Error creating temporary directory: {str(e)}", "danger")
                return redirect(url_for("main"))
            unique_id = uuid.uuid4().hex
            data_filepath = os.path.join("temp", f"{stock_symbol}_{unique_id}_data.json")
            try:
                df.index = df.index.map(lambda x: x.isoformat())
                json_data = df.to_json(date_format='iso', orient='index')
                with open(data_filepath, 'w') as f:
                    f.write(json_data)
                session[f"{stock_symbol}_data_path"] = data_filepath
                logger.debug(f"Fallback data fetched and saved to {data_filepath}: shape={df.shape}")
            except Exception as e:
                logger.error(f"Failed to write fallback data file {data_filepath}: {str(e)}", exc_info=True)
                flash(f"Error saving fallback data: {str(e)}", "danger")
                return redirect(url_for("main"))
        
        # Validate DataFrame before generating CSV
        required_columns = ['Close', 'MA_20', 'Volatility_30', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Williams_R_14', 'BB_Upper', 'BB_Lower', 'BB_Position']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in df: {missing_columns}")
            flash(f"Data processing error: Missing columns {missing_columns}", "danger")
            return redirect(url_for("main"))
        
        filename = f"{stock_symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', prefix='stock_', dir='temp') as temp_file:
                filepath = temp_file.name
                logger.debug(f"Creating temporary CSV file: {filepath}")
                temp_file.write(f"# Stock Analysis Data Export\n")
                temp_file.write(f"# Symbol: {stock_symbol}\n")
                temp_file.write(f"# Export Date: {datetime.now().isoformat()}\n")
                temp_file.write(f"# Data Points: {len(df)}\n")
                temp_file.write(f"# Date Range: {df.index[0].date()} to {df.index[-1].date()}\n")
                temp_file.write(f"# Features: {len(df.columns)} technical indicators included\n")
                temp_file.write("#\n")
                df.to_csv(temp_file)
        except Exception as e:
            logger.error(f"Failed to create temporary CSV file {filepath}: {str(e)}", exc_info=True)
            flash(f"Error creating CSV file: {str(e)}", "danger")
            return redirect(url_for("main"))
        
        logger.debug(f"CSV generated successfully: {filepath}")
        response = send_file(filepath, as_attachment=True, download_name=filename)
        
        try:
            os.unlink(filepath)
            logger.debug(f"Cleaned up temporary file: {filepath}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary file {filepath}: {str(e)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error downloading CSV: {str(e)}", exc_info=True)
        flash(f"Error downloading CSV: {str(e)}", "danger")
        return redirect(url_for("main"))

@app.route("/api/validate_symbol/<symbol>")
def validate_symbol(symbol):
    symbol = symbol.upper().strip()
    logger.debug(f"Validating symbol via API: {symbol}")
    try:
        is_valid = financial_data.validate_symbol(symbol)
        logger.debug(f"Validation result for {symbol}: {is_valid}")
        if not is_valid:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            logger.debug(f"yfinance info for {symbol}: {info}")
            is_valid = info.get('symbol') == symbol
    except Exception as e:
        logger.error(f"Error validating symbol {symbol}: {str(e)}", exc_info=True)
        is_valid = False
    return jsonify({"valid": is_valid, "symbol": symbol})

@app.route("/debug_validate/<symbol>")
def debug_validate(symbol):
    symbol = symbol.upper().strip()
    logger.debug(f"Debug validate called for: {symbol}")
    result = {"symbol": symbol, "valid": False, "details": {}}
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        result["details"] = {
            "symbol_from_yfinance": info.get('symbol'),
            "short_name": info.get('shortName'),
            "has_info": bool(info)
        }
        result["valid"] = info.get('symbol') == symbol
        logger.debug(f"Debug validation result: {result}")
    except Exception as e:
        result["details"]["error"] = str(e)
        logger.error(f"Debug validation error for {symbol}: {str(e)}", exc_info=True)
    return jsonify(result)

@app.route("/debug_session")
def debug_session():
    session_data = dict(session)
    temp_files = os.listdir('temp') if os.path.exists('temp') else []
    return jsonify({"session": session_data, "temp_files": temp_files})

@app.route("/debug_json/<stock_symbol>")
def debug_json(stock_symbol):
    data_path = session.get(f"{stock_symbol}_data_path")
    result = {"stock_symbol": stock_symbol, "data_path": data_path, "content": None, "error": None}
    if data_path and os.path.exists(data_path):
        try:
            with open(data_path, 'r') as f:
                data_json = f.read()
            json_data = json.loads(data_json)
            result["content"] = json.dumps(json_data, indent=2)[:1000]  # Limit output size
            result["size"] = os.path.getsize(data_path)
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to read JSON file {data_path}: {str(e)}", exc_info=True)
    else:
        result["error"] = f"No data file found for {stock_symbol}"
        logger.warning(f"No data file for {stock_symbol} at {data_path}")
    return jsonify(result)

@app.route("/logout")
def logout():
    for key in list(session.keys()):
        if '_data_path' in key or '_analysis_path' in key:
            path = session.pop(key, None)
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"Cleaned up file: {path}")
                except Exception as e:
                    logger.error(f"Error cleaning up file {path}: {str(e)}")
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)