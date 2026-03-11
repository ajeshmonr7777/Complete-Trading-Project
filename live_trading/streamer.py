import pandas as pd
import numpy as np
import websockets
import asyncio
import json
import time
import ssl
import certifi
import os
import requests
from datetime import datetime, timedelta
from typing import Dict
from binance.client import Client
from dotenv import load_dotenv

# Load env vars
load_dotenv()

class BinanceOHLCVStreamer:
    def __init__(self, symbols, api_key=None, secret_key=None, trading_analyst=None, interval='1m'):
        self.symbols = [symbol.lower() for symbol in symbols]
        self.interval = interval
        self.interval_minutes = int(interval.replace('m', ''))
        # Use provided keys or fall back to environment variables
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.secret_key = secret_key or os.getenv("BINANCE_SECRET_KEY")
        
        self.trading_analyst = trading_analyst  # Reference to trading analyst
        
        if self.api_key and self.secret_key:
            self.client = Client(self.api_key, self.secret_key)
        else:
            self.client = None
        
        # Initialize OHLCV data storage
        self.ohlcv_data = {symbol: [] for symbol in self.symbols}
        self.current_minute_data = {symbol: {} for symbol in self.symbols}
        self.max_candles = 250  # Keep last 250 candles
        
        # Create data directory
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize current minute data structure
        for symbol in self.symbols:
            self.current_minute_data[symbol] = {
                'open': None,
                'high': None,
                'low': None,
                'close': None,
                'volume': 0,
                'timestamp': None,
                'open_time': None,
                'close_time': None
            }
        
        self.running = False
        self.last_minute = None
        self.minutes_elapsed = 0
        self.last_analysis_time = None
        
    def get_csv_filename(self, symbol):
        """Get CSV filename for a symbol"""
        return os.path.join(self.data_dir, f"{symbol.upper()}_ohlcv_1m.csv")
    
    def load_existing_data(self, symbol):
        """Load existing data from CSV file"""
        filename = self.get_csv_filename(symbol)
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                # Convert timestamp string to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if 'open_time' in df.columns:
                    df['open_time'] = pd.to_datetime(df['open_time'])
                if 'close_time' in df.columns:
                    df['close_time'] = pd.to_datetime(df['close_time'])
                
                return df.to_dict('records')
            except Exception as e:
                print(f"Error loading existing data for {symbol}: {e}")
        return []
    
    def save_candle_to_csv(self, symbol, candle):
        """Save a single candle to CSV file"""
        filename = self.get_csv_filename(symbol)
        
        # Create DataFrame from candle
        candle_df = pd.DataFrame([candle])
        
        # Check if file exists
        if os.path.exists(filename):
            # Append to existing file
            existing_df = pd.read_csv(filename)
            updated_df = pd.concat([existing_df, candle_df], ignore_index=True)
            
            # Remove duplicates based on timestamp
            updated_df['timestamp'] = pd.to_datetime(updated_df['timestamp'])
            updated_df = updated_df.drop_duplicates(subset=['timestamp'], keep='last')
            updated_df = updated_df.sort_values('timestamp').reset_index(drop=True)
            
            # Keep only last 250 candles
            updated_df = updated_df.tail(self.max_candles)
            
            # Save back to CSV
            updated_df.to_csv(filename, index=False)
        else:
            # Create new file
            candle_df.to_csv(filename, index=False)
    
    def save_all_data_to_csv(self, symbol):
        """Save all OHLCV data for a symbol to CSV"""
        filename = self.get_csv_filename(symbol)
        if symbol in self.ohlcv_data and self.ohlcv_data[symbol]:
            df = pd.DataFrame(self.ohlcv_data[symbol])
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
                df.to_csv(filename, index=False)
    
    def fetch_historical_ohlcv(self, symbol, limit=250, interval='1m'):
        """Fetch historical OHLCV data from Binance API"""
        if not self.client:
            return []
            
        try:
            # Get klines data from Binance
            klines = self.client.get_klines(
                symbol=symbol.upper(),
                interval=interval,
                limit=limit
            )
            
            historical_data = []
            for kline in klines:
                candle = {
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                    'symbol': symbol.upper(),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'open_time': datetime.fromtimestamp(kline[0] / 1000),
                    'close_time': datetime.fromtimestamp(kline[6] / 1000),
                    'is_final': True,
                    'action': '',
                    'confidence': 0.0,
                    'justification': ''
                }
                historical_data.append(candle)
            
            return historical_data
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    def initialize_with_historical_data(self, limit=250, interval='1m'):
        """Initialize with historical data for all symbols"""
        print("🕐 Loading historical data...")
        for symbol in self.symbols:
            # First load any existing data from CSV
            existing_data = self.load_existing_data(symbol)
            
            # Fetch historical data from Binance
            historical_data = self.fetch_historical_ohlcv(symbol, limit, interval)
            
            # Combine existing and historical data
            all_data = existing_data + historical_data
            
            if all_data:
                # Remove duplicates and sort by timestamp
                df = pd.DataFrame(all_data)
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
                df = df.sort_values('timestamp').tail(self.max_candles)
                
                # Convert back to list of dictionaries
                self.ohlcv_data[symbol] = df.to_dict('records')
                
                # Save the combined data to CSV
                self.save_all_data_to_csv(symbol)
        
        print("✅ Historical data loaded")
    
    def initialize_current_minute(self, symbol, price, volume, timestamp):
        """Initialize new minute candle"""
        # Snap open_time down to the nearest interval blocks
        minute_bucket = (timestamp.minute // self.interval_minutes) * self.interval_minutes
        open_time = timestamp.replace(minute=minute_bucket, second=0, microsecond=0)
        close_time = open_time + timedelta(minutes=self.interval_minutes) - timedelta(microseconds=1)
        
        self.current_minute_data[symbol] = {
            'open': float(price),
            'high': float(price),
            'low': float(price),
            'close': float(price),
            'volume': float(volume),
            'timestamp': open_time,
            'open_time': open_time,
            'close_time': close_time
        }
        self.last_minute = open_time
        
    def update_current_minute(self, symbol, price, volume, timestamp):
        """Update current minute candle with new trade data"""
        price_float = float(price)
        volume_float = float(volume)
        
        # If new minute, finalize previous candle and start new one
        minute_bucket = (timestamp.minute // self.interval_minutes) * self.interval_minutes
        current_minute = timestamp.replace(minute=minute_bucket, second=0, microsecond=0)
        
        if self.current_minute_data[symbol]['open_time'] is None:
            self.initialize_current_minute(symbol, price, volume, timestamp)
            return
        
        if current_minute > self.current_minute_data[symbol]['open_time']:
            # Finalize the completed candle
            self.finalize_candle(symbol)
            # Start new candle
            self.initialize_current_minute(symbol, price, volume, timestamp)
        else:
            # Update current candle
            current_data = self.current_minute_data[symbol]
            current_data['close'] = price_float
            current_data['high'] = max(current_data['high'], price_float)
            current_data['low'] = min(current_data['low'], price_float)
            current_data['volume'] += volume_float
            current_data['close_time'] = timestamp
    
    async def trigger_trading_analysis(self, symbol):
        """Trigger trading analysis when new data is available"""
        if self.trading_analyst:
            try:
                # Get the latest OHLCV data as DataFrame
                df = self.get_ohlcv_data(symbol)
                
                if len(df) >= 200:
                    # Update portfolio history with current price before analysis
                    current_price = df['close'].iloc[-1]
                    self.trading_analyst.update_portfolio_history({symbol: current_price})
                    
                    # Update minutes elapsed
                    current_time = datetime.now()
                    if self.last_analysis_time:
                        self.minutes_elapsed += (current_time - self.last_analysis_time).seconds // 60
                    else:
                        self.minutes_elapsed = 1
                    self.last_analysis_time = current_time
                    
                    # Call trading analysis - PASS SYMBOL PARAMETER
                    decision = await self.trading_analyst.analyze_and_decide(df, self.minutes_elapsed, symbol)
                    
                    # Update the latest candle with trading decision
                    if "error" not in decision and self.ohlcv_data[symbol]:
                        latest_candle_index = len(self.ohlcv_data[symbol]) - 1
                        self.ohlcv_data[symbol][latest_candle_index]['action'] = decision.get('action', '')
                        self.ohlcv_data[symbol][latest_candle_index]['confidence'] = decision.get('confidence', 0.0)
                        self.ohlcv_data[symbol][latest_candle_index]['justification'] = decision.get('justification', '')
                        
                        # Update CSV file with the new columns
                        self.save_all_data_to_csv(symbol)
                        
                        # Print trading decision
                        action = decision.get('action', 'HOLD')
                        confidence = decision.get('confidence', 0.0)
                        justification = decision.get('justification', '')[:100]  # First 100 chars
                        
                        print(f"🎯 TRADING DECISION: {action} (Confidence: {confidence:.2f})")
                        print(f"   Reason: {justification}...")
                        print("─" * 50)
                    
            except Exception as e:
                print(f"Error in trading analysis: {e}")
    
    def finalize_candle(self, symbol):
        """Finalize the completed candle and add to OHLCV data"""
        current_data = self.current_minute_data[symbol]
        
        if current_data['open'] is not None:
            # Set final close_time to the end of the minute
            final_close_time = current_data['open_time'] + timedelta(minutes=self.interval_minutes) - timedelta(microseconds=1)
            
            candle = {
                'timestamp': current_data['open_time'],
                'symbol': symbol.upper(),
                'open': current_data['open'],
                'high': current_data['high'],
                'low': current_data['low'],
                'close': current_data['close'],
                'volume': current_data['volume'],
                'open_time': current_data['open_time'],
                'close_time': final_close_time,
                'is_final': True,
                'action': '',  # Initialize empty trading decision columns
                'confidence': 0.0,
                'justification': ''
            }
            
            # Add to OHLCV data
            self.ohlcv_data[symbol].append(candle)
            
            # Keep only recent candles
            if len(self.ohlcv_data[symbol]) > self.max_candles:
                self.ohlcv_data[symbol].pop(0)
            
            # Save candle to CSV file
            self.save_candle_to_csv(symbol, candle)
            
            # Print OHLCV details
            print(f"\n🕯️ NEW CANDLE - {symbol.upper()}")
            print(f"   Time:    {candle['timestamp'].strftime('%H:%M:%S')}")
            print(f"   Open:    ${candle['open']:.2f}")
            print(f"   High:    ${candle['high']:.2f}")
            print(f"   Low:     ${candle['low']:.2f}")
            print(f"   Close:   ${candle['close']:.2f}")
            print(f"   Volume:  {candle['volume']:.2f}")
            
            # Trigger trading analysis for this symbol
            asyncio.create_task(self.trigger_trading_analysis(symbol))
    
    async def connect_websocket(self):
        """Connect to Binance WebSocket stream for trade data (continuous)"""
        streams = [f"{symbol}@trade" for symbol in self.symbols]
        stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
        
        print(f"🚀 Starting live streaming for: {[s.upper() for s in self.symbols]}")
        print("─" * 50)
        
        reconnect_delay = 5  # seconds
        max_reconnect_delay = 60  # Maximum 60 seconds
        
        while self.running:
            try:
                # SSL context
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                
                async with websockets.connect(
                    stream_url, 
                    ssl=ssl_context,
                    ping_interval=20,    # Send ping every 20 seconds
                    ping_timeout=10,     # Wait 10 seconds for pong
                    close_timeout=10,    # Wait 10 seconds for close
                    max_size=2**20       # 1MB max message size
                ) as websocket:
                    print("✅ WebSocket connected successfully")
                    last_message_time = time.time()
                    reconnect_delay = 5  # Reset reconnect delay on successful connection
                    
                    while self.running:
                        try:
                            # Wait for message with timeout
                            message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                            last_message_time = time.time()
                            
                            data = json.loads(message)
                            
                            # Process trade data
                            if data.get('e') == 'trade':
                                symbol = data['s'].lower()
                                if symbol in self.symbols:
                                    timestamp = datetime.fromtimestamp(data['E'] / 1000)
                                    price = data['p']
                                    volume = data['q']
                                    
                                    # Update OHLCV data
                                    self.update_current_minute(symbol, price, volume, timestamp)
                        
                        except asyncio.TimeoutError:
                            # Check if we haven't received messages for too long
                            current_time = time.time()
                            if current_time - last_message_time > 45:  # 45 seconds without data
                                print("⚠️ No data received for 45 seconds, reconnecting...")
                                break
                                
                            # Check if we need to finalize any candles due to time passing
                            current_time_dt = datetime.now()
                            minute_bucket = (current_time_dt.minute // self.interval_minutes) * self.interval_minutes
                            current_bucket_time = current_time_dt.replace(minute=minute_bucket, second=0, microsecond=0)
                            
                            for symbol in self.symbols:
                                if (self.current_minute_data[symbol]['open_time'] is not None and 
                                    current_bucket_time > self.current_minute_data[symbol]['open_time']):
                                    # Check if we haven't already finalized this candle
                                    if not hasattr(self, f'last_finalized_{symbol}') or \
                                    getattr(self, f'last_finalized_{symbol}') != self.current_minute_data[symbol]['open_time']:
                                        self.finalize_candle(symbol)
                                        setattr(self, f'last_finalized_{symbol}', self.current_minute_data[symbol]['open_time'])
                                    # Use the last known price to initialize new candle
                                    last_price = self.current_minute_data[symbol]['close']
                                    self.initialize_current_minute(symbol, last_price, 0, current_time_dt)
                            continue
                        
                        except websockets.exceptions.ConnectionClosed:
                            print("🔌 WebSocket connection closed, reconnecting...")
                            break
                        
                        except Exception as e:
                            print(f"WebSocket receive error: {e}")
                            # Continue to next iteration instead of breaking
                            continue
                
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                print(f"Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)  # Exponential backoff with cap
        
    def get_ohlcv_data(self, symbol):
        """Get OHLCV data for a symbol as DataFrame"""
        symbol = symbol.lower()
        if symbol in self.ohlcv_data and self.ohlcv_data[symbol]:
            df = pd.DataFrame(self.ohlcv_data[symbol])
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        return pd.DataFrame()
    
    def get_current_candle(self, symbol):
        """Get the current incomplete candle data"""
        symbol = symbol.lower()
        if symbol in self.current_minute_data:
            return self.current_minute_data[symbol]
        return None
    
    def get_total_candles(self, symbol):
        """Get total number of candles (historical + live)"""
        symbol = symbol.lower()
        return len(self.ohlcv_data.get(symbol, []))
    
    async def start_streaming(self, fetch_historical=True, historical_limit=250, interval="1m"):
        """Start continuous streaming"""
        # Fetch historical data first if requested
        if fetch_historical:
            self.initialize_with_historical_data(limit=historical_limit)
        
        # Start live streaming
        self.running = True
        await self.connect_websocket()
    
    def stop_streaming(self):
        """Stop the streaming"""
        self.running = False
        print("🛑 Streaming stopped")
    def get_streaming_health(self) -> Dict:
        """Get streaming health status"""
        health = {
            'running': self.running,
            'symbols': self.symbols,
            'total_candles': {symbol: len(self.ohlcv_data.get(symbol, [])) for symbol in self.symbols},
            'last_analysis_time': self.last_analysis_time,
            'minutes_elapsed': self.minutes_elapsed
        }
        
        # Check if we're receiving data
        for symbol in self.symbols:
            df = self.get_ohlcv_data(symbol)
            if len(df) > 0:
                latest_candle = df.iloc[-1]
                health[f'{symbol}_latest_candle'] = latest_candle['timestamp']
            else:
                health[f'{symbol}_latest_candle'] = 'No data'
        
        return health

class UpstoxOHLCVStreamer:
    def __init__(self, symbols, api_key=None, secret_key=None, trading_analyst=None, access_token=None, interval='1m'):
        self.symbols = [symbol.upper() for symbol in symbols]
        self.interval = interval
        self.interval_minutes = int(interval.replace('m', ''))
        self.access_token = access_token
        self.trading_analyst = trading_analyst
        
        # Initialize OHLCV data storage
        self.ohlcv_data = {symbol: [] for symbol in self.symbols}
        self.current_minute_data = {symbol: {} for symbol in self.symbols}
        self.max_candles = 250
        
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize current minute data structure
        for symbol in self.symbols:
            self.current_minute_data[symbol] = {
                'open': None, 'high': None, 'low': None, 'close': None, 'volume': 0,
                'timestamp': None, 'open_time': None, 'close_time': None
            }
        
        self.running = False
        self.last_minute = None
        self.minutes_elapsed = 0
        self.last_analysis_time = None

    def get_instrument_key(self, symbol):
        # Basic assumption: NSE Equity
        if "|" in symbol:
            return symbol
            
        # Hardcoded ISIN maps for the Live Trading Dashboard dropdown options
        known_stocks = {
            "INFY": "NSE_EQ|INE009A01021",
            "SBIN": "NSE_EQ|INE062A01020",
            "RELIANCE": "NSE_EQ|INE002A01018",
            "TCS": "NSE_EQ|INE467B01029",
            "HDFCBANK": "NSE_EQ|INE040A01034",
            "BHARTIARTL": "NSE_EQ|INE397D01024",
            "ITC": "NSE_EQ|INE154A01025",
            "ICICIBANK": "NSE_EQ|INE090A01021"
        }
        
        symbol_upper = symbol.upper()
        if symbol_upper in known_stocks:
            return known_stocks[symbol_upper]
            
        return f"NSE_EQ|{symbol}"

    def get_csv_filename(self, symbol):
        return os.path.join(self.data_dir, f"{symbol.upper()}_ohlcv_1m.csv")
    
    def load_existing_data(self, symbol):
        filename = self.get_csv_filename(symbol)
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.to_dict('records')
            except Exception:
                return []
        return []

    def save_candle_to_csv(self, symbol, candle):
        filename = self.get_csv_filename(symbol)
        candle_df = pd.DataFrame([candle])
        # Add missing columns if not present
        if 'action' not in candle_df.columns: candle_df['action'] = ''
        if 'confidence' not in candle_df.columns: candle_df['confidence'] = 0.0
        if 'justification' not in candle_df.columns: candle_df['justification'] = ''
        
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            updated_df = pd.concat([existing_df, candle_df], ignore_index=True)
            updated_df['timestamp'] = pd.to_datetime(updated_df['timestamp'])
            updated_df = updated_df.drop_duplicates(subset=['timestamp'], keep='last')
            updated_df = updated_df.sort_values('timestamp').tail(self.max_candles)
            updated_df.to_csv(filename, index=False)
        else:
            candle_df.to_csv(filename, index=False)

    def save_all_data_to_csv(self, symbol):
        filename = self.get_csv_filename(symbol)
        if symbol in self.ohlcv_data and self.ohlcv_data[symbol]:
            df = pd.DataFrame(self.ohlcv_data[symbol])
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
                df.to_csv(filename, index=False)

    def fetch_historical_ohlcv(self, symbol, limit=250):
        if not self.access_token:
            print("❌ No Upstox Access Token")
            return []
            
        try:
            instrument_key = self.get_instrument_key(symbol)
            to_date = datetime.now().strftime('%Y-%m-%d')
            # For larger intervals, fetch slightly more history to fill 250 bars
            days_history = 5
            if self.interval == '15m': days_history = 15
            elif self.interval == '5m': days_history = 10
            
            from_date = (datetime.now() - timedelta(days=days_history)).strftime('%Y-%m-%d')
            
            # Map shorthand to upstox API enum (1minute, 30minute, day, 1month etc. Not all sub-intervals perfectly supported directly, fallback typically 1minute aggregate)
            # Standard Upstox intervals: 1minute, 30minute, day, 1month. Others might fail directly.
            # Upstox doesn't natively expose direct 3m/5m/15m links cleanly in the V2 path for some accounts, so we pull 1minute and aggregate locally if preferred, 
            # OR we try standard deep-link intervals (e.g. '30minute' exists). For safety, we fall back to upstox formatting:
            interval_str = f"{self.interval_minutes}minute"
            
            url_history = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval_str}/{to_date}/{from_date}"
            url_intraday = f"https://api.upstox.com/v2/historical-candle/intraday/{instrument_key}/{interval_str}"
            headers = {'Accept': 'application/json'}
            
            raw_candles = []
            
            # 1. Fetch Past Days
            try:
                res_hist = requests.get(url_history, headers=headers)
                data_hist = res_hist.json()
                if data_hist.get('status') == 'success' and data_hist.get('data') and data_hist['data'].get('candles'):
                    raw_candles.extend(data_hist['data']['candles'])
            except Exception as e:
                print(f"Upstox history warning: {e}")
                
            # 2. Fetch Today's Intraday (The missing gap)
            try:
                res_intra = requests.get(url_intraday, headers=headers)
                data_intra = res_intra.json()
                if data_intra.get('status') == 'success' and data_intra.get('data') and data_intra['data'].get('candles'):
                    raw_candles.extend(data_intra['data']['candles'])
            except Exception as e:
                print(f"Upstox intraday warning: {e}")
            
            if not raw_candles:
                return []
                
            historical_data = []
            seen_ts = set()
            for c in raw_candles:
                # c = [timestamp, open, high, low, close, volume, oi]
                # Upstox timestamp: "2023-11-20T09:15:00+05:30"
                ts_str = c[0]
                if "+" in ts_str:
                     ts_str = ts_str.split("+")[0]
                ts = datetime.fromisoformat(ts_str)
                
                # Deduplicate overlapping candles safely
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                
                candle = {
                    'timestamp': ts,
                    'symbol': symbol.upper(),
                    'open': float(c[1]), 'high': float(c[2]), 'low': float(c[3]), 'close': float(c[4]),
                    'volume': float(c[5]),
                    'open_time': ts,
                    'close_time': ts + timedelta(minutes=self.interval_minutes),
                    'is_final': True,
                    'action': '', 'confidence': 0.0, 'justification': ''
                }
                historical_data.append(candle)
            
            historical_data.sort(key=lambda x: x['timestamp'])
            return historical_data[-limit:]
        except Exception as e:
            print(f"Error fetching Upstox history for {symbol}: {e}")
            return []

    def initialize_with_historical_data(self, limit=250):
        print("🕐 Loading Upstox historical data...")
        for symbol in self.symbols:
            existing = self.load_existing_data(symbol)
            historical = self.fetch_historical_ohlcv(symbol, limit)
            
            # Merge logic
            combined = existing + historical
            if combined:
                 # Deduplicate by timestamp
                 seen = set()
                 unique = []
                 for item in combined:
                     ts = item['timestamp']
                     if ts not in seen:
                         seen.add(ts)
                         unique.append(item)
                 
                 # Sort
                 unique.sort(key=lambda x: x['timestamp'])
                 self.ohlcv_data[symbol] = unique[-self.max_candles:]
                 self.save_all_data_to_csv(symbol)
        print("✅ Upstox historical data loaded")

    def initialize_current_minute(self, symbol, price, volume, timestamp):
        minute_bucket = (timestamp.minute // self.interval_minutes) * self.interval_minutes
        open_time = timestamp.replace(minute=minute_bucket, second=0, microsecond=0)
        close_time = open_time + timedelta(minutes=self.interval_minutes)
        self.current_minute_data[symbol] = {
            'open': float(price), 'high': float(price), 'low': float(price), 'close': float(price),
            'volume': float(volume), 'timestamp': open_time, 'open_time': open_time, 'close_time': close_time
        }
        self.last_minute = open_time

    async def update_current_minute(self, symbol, price, volume, timestamp):
        price = float(price)
        minute_bucket = (timestamp.minute // self.interval_minutes) * self.interval_minutes
        current_minute = timestamp.replace(minute=minute_bucket, second=0, microsecond=0)
        
        current_data = self.current_minute_data[symbol]
        
        if current_data['open_time'] is None:
            self.initialize_current_minute(symbol, price, volume, timestamp)
            return

        if current_minute > current_data['open_time']:
            await self.finalize_candle(symbol)
            self.initialize_current_minute(symbol, price, volume, timestamp)
        else:
            current_data['close'] = price
            current_data['high'] = max(current_data['high'], price)
            current_data['low'] = min(current_data['low'], price)
            current_data['volume'] += volume 
            current_data['close_time'] = timestamp

    async def finalize_candle(self, symbol):
        current = self.current_minute_data[symbol]
        if current['open'] is not None:
             candle = {
                'timestamp': current['open_time'],
                'symbol': symbol.upper(),
                'open': current['open'], 'high': current['high'], 'low': current['low'], 'close': current['close'],
                'volume': current['volume'],
                'open_time': current['open_time'],
                'close_time': current['open_time'] + timedelta(minutes=self.interval_minutes),
                'is_final': True,
                'action': '', 'confidence': 0.0, 'justification': ''
            }
             self.ohlcv_data[symbol].append(candle)
             if len(self.ohlcv_data[symbol]) > self.max_candles:
                 self.ohlcv_data[symbol].pop(0)
             self.save_candle_to_csv(symbol, candle)
             
             print(f"\n🕯️ NEW CANDLE - {symbol.upper()} | Close: {candle['close']}")
             # Trigger analysis
             if self.trading_analyst:
                 await self.trigger_trading_analysis(symbol)

    async def trigger_trading_analysis(self, symbol):
         try:
             df = self.get_ohlcv_data(symbol)
             if len(df) >= 10: 
                 current_price = df['close'].iloc[-1]
                 self.trading_analyst.update_portfolio_history({symbol: current_price})
                 
                 current_time = datetime.now()
                 if self.last_analysis_time:
                     self.minutes_elapsed += (current_time - self.last_analysis_time).seconds // 60
                 else:
                     self.minutes_elapsed = 1
                 self.last_analysis_time = current_time
                 
                 decision = await self.trading_analyst.analyze_and_decide(df, self.minutes_elapsed, symbol)
                 
                 # Save decision
                 if "error" not in decision and self.ohlcv_data[symbol]:
                     latest = self.ohlcv_data[symbol][-1]
                     latest['action'] = decision.get('action', '')
                     latest['confidence'] = decision.get('confidence', 0.0)
                     latest['justification'] = decision.get('justification', '')
                     self.save_all_data_to_csv(symbol)
                     
                     print(f"🎯 TRADING DECISION: {decision.get('action', 'HOLD')}")
         except Exception as e:
             print(f"Error in analysis: {e}")

    def get_ohlcv_data(self, symbol):
        symbol = symbol.upper()
        if symbol in self.ohlcv_data:
            return pd.DataFrame(self.ohlcv_data[symbol])
        return pd.DataFrame()
        
    async def start_streaming(self, fetch_historical=True, historical_limit=250, interval="1m"):
        if fetch_historical:
            self.initialize_with_historical_data(limit=historical_limit)
        
        print(f"🚀 Starting Upstox Polling for: {self.symbols}")
        self.running = True
        
        while self.running:
            try:
                instrument_keys = ",".join([self.get_instrument_key(s) for s in self.symbols])
                url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={instrument_keys}"
                headers = {'Authorization': f'Bearer {self.access_token}', 'Accept': 'application/json'}
                
                response = requests.get(url, headers=headers)
                data = response.json()
                
                if data.get('status') == 'success':
                    timestamp = datetime.now()
                    # Response structure: { 'status': 'success', 'data': { 'NSE_EQ|RELIANCE': { 'last_price': ... } } }
                    for symbol in self.symbols:
                        key = self.get_instrument_key(symbol)
                        alt_key = f"NSE_EQ:{symbol.upper()}"
                        
                        ltp = None
                        if key in data['data']:
                            ltp = data['data'][key]['last_price']
                        elif alt_key in data['data']:
                            ltp = data['data'][alt_key]['last_price']
                            
                        if ltp is not None:
                            await self.update_current_minute(symbol, ltp, 0, timestamp)
                elif data.get('status') == 'error':
                    # Extremely loud warning for token expiration
                    error_msg = data.get('errors', [{'message': 'Unknown API Error'}])[0].get('message', 'Unknown API Error')
                    print(f"❌ UPSTOX API REJECTED POLLING: {error_msg}")
                    if 'token' in error_msg.lower():
                         print("⚠️ YOUR UPSTOX DAILY ACCESS TOKEN HAS EXPIRED!")
                         print("👉 Go to http://127.0.0.1:8000/upstox/login in your browser to instantly generate today's free token.")
                
                await asyncio.sleep(2) 
            except Exception as e:
                print(f"Polling error: {e}")
                await asyncio.sleep(5)

    def stop_streaming(self):
        self.running = False