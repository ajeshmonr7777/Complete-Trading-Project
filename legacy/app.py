import streamlit as st
import pandas as pd
import yfinance as yf
from backtester import BacktestEngine
from converters import PineScriptParser
import importlib.util
import os

st.set_page_config(page_title="Backtesting Algorithm Script", layout="wide")

st.title("Backtesting Algorithm Script")

# Sidebar for Configuration
st.sidebar.header("Configuration")
# Symbol Selection
market_type = st.sidebar.selectbox("Market Type", ["Indian Stocks", "US Stocks", "Forex", "Crypto", "Custom"])

if market_type == "Indian Stocks":
    # Popular NSE/BSE stocks
    options = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TATAMOTORS.NS", "SBIN.NS", "LICI.NS", "BHARTIARTL.NS", "ITC.NS"]
    ticker = st.sidebar.selectbox("Select Ticker", options, index=0)
elif market_type == "US Stocks":
    options = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "SPY"]
    ticker = st.sidebar.selectbox("Select Ticker", options, index=0)
elif market_type == "Forex":
    options = ["EURUSD=X", "GBPUSD=X", "JPY=X", "INR=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"]
    ticker = st.sidebar.selectbox("Select Ticker", options, index=0)
elif market_type == "Crypto":
    options = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]
    ticker = st.sidebar.selectbox("Select Ticker", options, index=0)
else:
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
initial_capital = st.sidebar.number_input("Initial Capital", value=10000.0)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m", "1m"], index=0)

# Risk Management
st.sidebar.subheader("Risk Management")
enable_risk_mgmt = st.sidebar.checkbox("Enable Stop Loss & R:R")
stop_loss_pct = 0.0
risk_reward_ratio = 0.0

if enable_risk_mgmt:
    stop_loss_pct = st.sidebar.number_input("Stop Loss (%)", min_value=0.1, value=2.0, step=0.1)
    risk_reward_ratio = st.sidebar.number_input("Risk:Reward Ratio", min_value=0.1, value=2.0, step=0.1)

# File Uploader
# Strategy Selection Mode
mode = st.radio("Strategy Source", ["Select Saved Strategy", "Upload Strategy"])

file_path = None
uploaded_file = None

if mode == "Upload Strategy":
    uploaded_file = st.file_uploader("Upload a Python (.py) or Pine Script (.pine/.txt) file", type=["py", "pine", "txt"])
    if uploaded_file:
        # Save uploaded file temporarily
        os.makedirs("strategies", exist_ok=True)
        file_path = os.path.join("strategies", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

elif mode == "Select Saved Strategy":
    os.makedirs("strategies", exist_ok=True)
    strategy_files = [f for f in os.listdir("strategies") if f.endswith((".py", ".pine", ".txt"))]
    if strategy_files:
        selected_strategy = st.selectbox("Select Strategy", strategy_files)
        file_path = os.path.join("strategies", selected_strategy)
    else:
        st.warning("No strategies found in 'strategies' folder.")

if st.button("Run Backtest"):
    if file_path and ticker:
        # For uploaded files, we already saved them.
        # For selected files, file_path is already set.
        
        # Initialize Backtester
        engine = BacktestEngine(ticker, start_date, end_date, initial_capital, interval)
        data = engine.load_data()
        
        if data is None or data.empty:
            st.error("No data found for the given ticker and date range.")
        else:
            st.success(f"Loaded {len(data)} rows of data for {ticker}")
            
            # Determine logic based on file type
            if file_path.endswith(".py"):
                # Load Python module dynamically
                spec = importlib.util.spec_from_file_location("strategy_module", file_path)
                strategy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(strategy_module)
                
                if hasattr(strategy_module, "strategy"):
                    results = engine.run_strategy(strategy_module.strategy, stop_loss_pct=stop_loss_pct if enable_risk_mgmt else None, risk_reward_ratio=risk_reward_ratio if enable_risk_mgmt else None)
                else:
                    st.error("Python file must contain a 'strategy(data)' function.")
                    results = None
            else:
                # Parse Pine Script
                with open(file_path, "r") as f:
                    pine_code = f.read()
                parser = PineScriptParser(pine_code)
                python_logic = parser.parse()
                results = engine.run_strategy(python_logic, stop_loss_pct=stop_loss_pct if enable_risk_mgmt else None, risk_reward_ratio=risk_reward_ratio if enable_risk_mgmt else None)

            # Display Results
            if results is not None:
                st.subheader("Equity Curve")
                st.line_chart(results['equity_curve'])
                
                st.subheader("Performance Metrics")
                metrics = engine.calculate_metrics(results)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Return", f"{metrics['total_return']:.2f}%")
                c2.metric("Final Equity", f"${metrics['final_equity']:.2f}")
                c3.metric("Number of Trades", metrics['num_trades'])
                
                st.subheader("Trade Log")
                st.dataframe(results['trades'])
    else:
        st.warning("Please select or upload a strategy file and specify a ticker.")
