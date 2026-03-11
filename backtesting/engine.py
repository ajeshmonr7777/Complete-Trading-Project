import pandas as pd
import yfinance as yf

class BacktestEngine:
    def __init__(self, ticker, start_date, end_date, initial_capital=10000.0, interval="1d"):
        self.ticker = ticker
        # Ensure dates are in string format 'YYYY-MM-DD' for yfinance
        self.start_date = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
        self.end_date = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
        self.initial_capital = initial_capital
        self.interval = interval
        self.data = None

    def load_data(self):
        try:
            # Smart buffer logic based on interval
            import datetime
            # Determine Max History based on Interval (Yahoo Finance Limits)
            max_days = 0 # 0 means unlimited/max available (usually 5-10 years daily)
            
            if self.interval == '1m':
                max_days = 7
            elif self.interval in ['2m', '5m', '15m', '30m', '90m']:
                max_days = 60
            elif self.interval in ['60m', '1h']:
                max_days = 730
                
            # Calculate earliest allowable start date
            today = datetime.datetime.now()
            
            # Start logic
            start_dt = pd.to_datetime(self.start_date)
            
            # If max_days is set, clamp the start_date
            if max_days > 0:
                earliest_possible = today - datetime.timedelta(days=max_days - 1) # buffer 1 day safely
                if start_dt < earliest_possible:
                    print(f"Warning: Requested start date {start_dt.date()} is too far back for {self.interval} interval.")
                    print(f"Adjusting start date to {earliest_possible.date()} (Max {max_days} days).")
                    start_dt = earliest_possible

            # Add buffer for indicators
            # For intraday, we simply download from start_dt because 'max_days' is a HARD LIMIT for yfinance download start.
            # You simply cannot request data older than 'max_days'. So buffer is impossible if we are already at the limit.
            if max_days > 0:
                 download_start_dt = start_dt # Can't go back further
            else:
                 # Daily/Weekly - safe to add buffer
                 download_start_dt = start_dt - datetime.timedelta(days=365)
            
            download_start = download_start_dt.strftime('%Y-%m-%d')
            
            # --- Try Upstox First for Indian Markets ---
            is_indian_market = self.ticker.endswith('.NS') or self.ticker.endswith('.BO')
            upstox_df = None
            if is_indian_market:
                # Add a few days of buffer to download_start to ensure enough data for indicator calculation
                upstox_download_dt = download_start_dt - datetime.timedelta(days=7) if self.interval in ['1d'] else download_start_dt
                upstox_download_start = upstox_download_dt.strftime('%Y-%m-%d')
                upstox_df = self._fetch_upstox_data(self.ticker, upstox_download_start, self.end_date, self.interval)
            
            if upstox_df is not None:
                self.data = upstox_df
            else:
                self.data = yf.download(self.ticker, start=download_start, end=self.end_date, interval=self.interval, progress=False)
                if self.data.columns.nlevels > 1:
                     # Flatten columns if multi-level (e.g., from recent yfinance versions)
                    self.data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in self.data.columns.values]
                    
                    # Normalize column names to standard 'Open', 'High', 'Low', 'Close', 'Volume'
                    rename_map = {}
                    for col in self.data.columns:
                        if 'Close' in col: rename_map[col] = 'Close'
                        elif 'Open' in col: rename_map[col] = 'Open'
                        elif 'High' in col: rename_map[col] = 'High'
                        elif 'Low' in col: rename_map[col] = 'Low'
                        elif 'Volume' in col: rename_map[col] = 'Volume'
                    self.data.rename(columns=rename_map, inplace=True)
                    
                    # Ensure unique index
                    self.data = self.data[~self.data.index.duplicated(keep='first')]
                
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _fetch_upstox_data(self, ticker, start_date, end_date, interval):
        import requests
        import pandas as pd
        
        # Determine Upstox Interval
        interval_map = {
            '1m': '1minute',
            '30m': '30minute',
            '60m': '60minute',
            '1h': '60minute',
            '1d': 'day',
            '1wk': 'week',
            '1mo': 'month'
        }
        ux_interval = interval_map.get(interval)
        if not ux_interval:
            print(f"⚠️ Upstox doesn't support interval '{interval}'. Falling back to yfinance.")
            return None
            
        # Extract base symbol
        base_symbol = ticker.replace('.NS', '').replace('.BO', '')
        instrument_key = f"NSE_EQ|{base_symbol}"
        
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{ux_interval}/{end_date}/{start_date}"
        headers = {'Accept': 'application/json'}
        
        try:
            print(f"🔄 Attempting to fetch {base_symbol} from Upstox API...")
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
                    candles = data['data']['candles']
                    if not candles:
                        print("⚠️ Upstox returned empty data. Falling back to yfinance.")
                        return None
                    
                    df_data = []
                    for c in candles:
                        ts_str = c[0]
                        if "+" in ts_str:
                             ts_str = ts_str.split("+")[0]
                        df_data.append({
                            'Date': pd.to_datetime(ts_str),
                            'Open': float(c[1]),
                            'High': float(c[2]),
                            'Low': float(c[3]),
                            'Close': float(c[4]),
                            'Volume': float(c[5])
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)
                    # Convert index to timezone-naive to match expected processing downstream if needed
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    print(f"✅ Successfully loaded {len(df)} rows from Upstox for {base_symbol}")
                    return df
            print(f"⚠️ Upstox fetch non-200 or failure (status {res.status_code}). Falling back to yfinance.")
        except Exception as e:
            print(f"⚠️ Upstox fetch error: {e}. Falling back to yfinance.")
        return None

    def run_strategy(self, strategy_func, stop_loss_pct=None, risk_reward_ratio=None, max_active_trades=1, exit_on_signal=True):
        """
        Runs the provided strategy function on the data.
        strategy_func: python function that takes dataframe and returns a dataframe with 'Signal' column.
        Signal: 1 (Buy), -1 (Sell), 0 (Hold)
        max_active_trades: Integer, default 1. Allows concurrent trades if > 1.
        exit_on_signal: Boolean, default True. If False, ignores Sell signals (-1) and only exits on SL/TP/End.
        """
        if self.data is None:
            return None
        
        # Make a copy to avoid modifying original data
        df = self.data.copy()
        
        # Execute strategy (Calculation happens on FULL data including buffer)
        try:
            df = strategy_func(df)
        except Exception as e:
            print(f"Strategy execution failed: {e}")
            return None
            
        if 'Signal' not in df.columns:
            print("Strategy did not return a 'Signal' column.")
            return None

        # SLICE the dataframe to the requested Backtest Start Date
        df = df[df.index >= self.start_date].copy()
        
        if df.empty:
            print("No data remaining after slicing to start_date.")
            return None

        # --- Multi-Trade Engine ---
        equity_curve = []
        trades = []
        
        # State
        current_equity = self.initial_capital
        cash = self.initial_capital
        positions = [] # List of dicts: { 'entry_price', 'shares', 'stop_price', 'target_price', 'entry_time' }
        
        # Helper: Calculate current total equity
        def get_total_equity(current_px):
            val = cash
            for p in positions:
                val += p['shares'] * current_px
            return val

        # Start Loop
        # Initial point
        equity_curve.append(current_equity)
        
        for i in range(1, len(df)):
            price = df['Close'].iloc[i]
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            open_px = df['Open'].iloc[i]
            signal = df['Signal'].iloc[i]
            timestamp = df.index[i]
            
            # 1. Process Exits (SL/TP) or Exit Signals for ACTIVE positions
            # We iterate backwards to safely remove items
            for idx in range(len(positions) - 1, -1, -1):
                p = positions[idx]
                exit_price = None
                exit_reason = None
                
                # Check Stop Loss / Take Profit
                if stop_loss_pct is not None and risk_reward_ratio is not None:
                    if low <= p['stop_price']:
                        # Gap check
                        exit_price = open_px if open_px < p['stop_price'] else p['stop_price']
                        exit_reason = 'Stop Loss'
                    elif high >= p['target_price']:
                        # Gap check
                        exit_price = open_px if open_px > p['target_price'] else p['target_price']
                        exit_reason = 'Take Profit'
                
                # Check Sell Signal
                # Only if exit_on_signal is TRUE
                if exit_price is None and exit_on_signal and signal == -1:
                    exit_price = price
                    exit_reason = 'Signal'

                # Execute Exit
                if exit_price is not None:
                    revenue = p['shares'] * exit_price
                    cash += revenue
                    
                    # Record Trade
                    trades.append({
                        'Symbol': self.ticker,
                        'Type': 'Buy', # Long only for now
                        'Entry Time': p['entry_time'],
                        'Entry Price': p['entry_price'],
                        'Exit Time': timestamp,
                        'Exit Price': exit_price,
                        'Exit Reason': exit_reason,
                        'PnL': revenue - (p['shares'] * p['entry_price'])
                    })
                    
                    positions.pop(idx) # Remove from active

            # Update Equity after exits (Cash updated)
            # Check Entries
            
            # 2. Process Entry (Buy)
            # Allow entry if we have slots left
            if signal == 1:
                # If max_active_trades > 1, we allow adding even if we have positions
                if len(positions) < max_active_trades:
                    # Calculate Position Size
                    # Simple Model: Allocation = Current Total Equity / Max Trades
                    # This implies rebalancing logic or fixed fractional?
                    # Let's use: Allowable Capital = Current Equity / Max Trades.
                    # If Cash is sufficient, take it.
                    
                    total_eq = get_total_equity(price)
                    target_alloc = total_eq / max_active_trades
                    
                    # Ensure we don't spend more than we have (Cash constraint)
                    # And don't spend tiny amounts usually?
                    alloc = min(cash, target_alloc)
                    
                    if alloc > 0:
                        entry_price = price
                        shares = alloc / entry_price
                        cash -= alloc
                        
                        # Risk Params
                        sp = 0
                        tp = 0
                        if stop_loss_pct is not None and risk_reward_ratio is not None:
                             stop_dist = entry_price * (stop_loss_pct / 100.0)
                             sp = entry_price - stop_dist
                             tp = entry_price + (stop_dist * risk_reward_ratio)
                        
                        positions.append({
                            'entry_price': entry_price,
                            'shares': shares,
                            'stop_price': sp,
                            'target_price': tp,
                            'entry_time': timestamp
                        })

            # 3. Mark to Market
            # Update equity curve for this timestamp
            equity_curve.append(get_total_equity(price))

        # End of Data: Force Close Remaining
        last_price = df['Close'].iloc[-1]
        last_time = df.index[-1]
        
        for p in positions:
            revenue = p['shares'] * last_price
            cash += revenue
            trades.append({
                'Symbol': self.ticker,
                'Type': 'Buy',
                'Entry Time': p['entry_time'],
                'Entry Price': p['entry_price'],
                'Exit Time': last_time,
                'Exit Price': last_price,
                'Exit Reason': 'End of Backtest',
                'PnL': revenue - (p['shares'] * p['entry_price'])
            })
            
        positions.clear() # Clean up

        results = {
            'equity_curve': pd.Series(equity_curve, index=df.index),
            'trades': pd.DataFrame(trades)
        }
        return results

    def calculate_metrics(self, results):
        equity = results['equity_curve']
        total_return = ((equity.iloc[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate win rate
        trades_df = results['trades']
        win_rate = 0.0
        if len(trades_df) > 0:
            wins = len(trades_df[trades_df['PnL'] > 0])
            win_rate = (wins / len(trades_df)) * 100
        
        return {
            'total_return': total_return,
            'final_equity': equity.iloc[-1],
            'num_trades': len(results['trades']),
            'win_rate': win_rate
        }

