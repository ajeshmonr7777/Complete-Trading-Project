import pandas as pd
import numpy as np
import websockets
import asyncio
import json
import time
import ssl
import certifi
import os
from datetime import datetime, timedelta
from typing import Dict
from binance.client import Client
from dotenv import load_dotenv

# Load env vars
load_dotenv()

class BinanceOHLCVStreamer:
    def __init__(self, symbols, api_key=None, secret_key=None, trading_analyst=None):
        self.symbols = [symbol.lower() for symbol in symbols]
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
        open_time = timestamp.replace(second=0, microsecond=0)
        close_time = open_time + timedelta(minutes=1) - timedelta(microseconds=1)
        
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
        current_minute = timestamp.replace(second=0, microsecond=0)
        
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
            final_close_time = current_data['open_time'] + timedelta(minutes=1) - timedelta(microseconds=1)
            
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
                            for symbol in self.symbols:
                                if (self.current_minute_data[symbol]['open_time'] is not None and 
                                    current_time_dt.minute != self.current_minute_data[symbol]['open_time'].minute):
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
    
    async def start_streaming(self, fetch_historical=True, historical_limit=250):
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