import pandas as pd
import numpy as np
import requests
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import os

# Import from separate modules (using live_ prefix)
from .indicators import calculate_technical_indicators
from .prompts import create_system_prompt
import csv
import glob
# Note: bybit_trading and paper_trading removed for simplified version


class TradingAnalyst:
    @property
    def lot_size(self):
        return self._lot_size
    
    @lot_size.setter
    def lot_size(self, value):
        if self._lot_size != value:
            print(f"⚙️ TradingAnalyst: Updating lot_size from {self._lot_size} to {value}")
            self._lot_size = value

    def __init__(self, deepseek_api_key: str, use_mock: bool = False, initial_capital: float = 1000.0, leverage: float = 10.0, enable_real_trading: bool = False):
        self.deepseek_api_key = deepseek_api_key
        self.base_url = "https://api.deepseek.com/chat/completions"
        self.use_mock = use_mock
        self.initial_capital = initial_capital
        self.enable_real_trading = enable_real_trading
        self.leverage = leverage  # Leverage multiplier (default: 10x)
        self._lot_size = 0.01  # Default lot size
        
        # Trading parameters
        self.interval = "1m"
        self.ema_period = 20
        self.rsi_period = 14
        self.prompt_strategy = "prompt3" # Default strategy
        self.prompt_builder = create_system_prompt
        
        # Portfolio state - PERSISTS between calls
        self.portfolio = {
            'initial_capital': self.initial_capital,
            'available_cash': self.initial_capital,
            'current_value': self.initial_capital,
            'positions': {},
            'total_trades': 0,
            'winning_trades': 0,
            'trade_history': []
        }
        
        # Add portfolio history tracking
        self.portfolio_history = []
        self._initialize_portfolio_history()
        
        # Decision history
        self.decision_history = []
        
        # Initialize CSV Logging
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.csv_filename = self._get_next_csv_filename()
        self._initialize_csv()


    def _get_next_csv_filename(self):
        """Find the next available summary_N.csv filename in data folder"""
        existing_files = glob.glob(os.path.join(self.data_dir, "summary_*.csv"))
        if not existing_files:
            return os.path.join(self.data_dir, "summary_1.csv")
        
        numbers = []
        for f in existing_files:
            try:
                # Extract number from data/summary_N.csv
                basename = os.path.basename(f)
                num = int(basename.replace("summary_", "").replace(".csv", ""))
                numbers.append(num)
            except ValueError:
                continue
        
        next_num = max(numbers) + 1 if numbers else 1
        return os.path.join(self.data_dir, f"summary_{next_num}.csv")

    def _initialize_csv(self):
        """Create CSV file with headers"""
        headers = [
            "Time", "Symbol", "Open", "High", "Low", "Close", "Volume", 
            "Prompt", "Portfolio Value", "Action", "Position Status", "Reason"
        ]
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"📝 Logging to {self.csv_filename}")


    def log_to_csv(self, timestamp, symbol, ohlcv_row, prompt_name, portfolio_value, action, position_status, reason):
        """Log trading step to CSV"""
        try:
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    symbol,
                    ohlcv_row['open'], ohlcv_row['high'], ohlcv_row['low'], ohlcv_row['close'], ohlcv_row['volume'],
                    prompt_name,
                    f"{portfolio_value:.2f}",
                    action,
                    position_status,
                    reason
                ])
        except Exception as e:
            print(f"❌ Failed to log to CSV: {e}")
    
    def _initialize_portfolio_history(self):
        """Initialize portfolio history with starting point"""
        initial_entry = {
            'timestamp': datetime.now(),
            'total_value': self.portfolio['initial_capital'],
            'available_cash': self.portfolio['initial_capital'],
            'positions_value': 0.0,
            'return_percent': 0.0,
            'total_trades': 0,
            'win_rate': 0.0
        }
        self.portfolio_history.append(initial_entry)
    
    def update_portfolio_history(self, current_prices: Dict[str, float] = None):
        """Update portfolio history with current values - With Leverage Support"""
        total_unrealized_pnl = 0.0
        total_margin_in_use = 0.0
        
        for symbol, position in self.portfolio['positions'].items():
            if current_prices and symbol in current_prices:
                current_price = current_prices[symbol]
            else:
                current_price = position['entry_price']
            
            quantity = position['quantity']
            entry_price = position['entry_price']
            margin_used = position.get('margin_used', position.get('position_value', 0))  # Fallback for old positions
            
            # Calculate unrealized PnL (based on full position size)
            if position['position_type'] == 'LONG':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:  # SHORT
                unrealized_pnl = (entry_price - current_price) * quantity
                
            total_unrealized_pnl += unrealized_pnl
            total_margin_in_use += margin_used  # Use margin, not cost basis
        
        # CORRECT portfolio value calculation for Leveraged account:
        # Total Value = Available Cash + Margin in Positions + Unrealized PnL
        # This ensures: initial_capital = available_cash + margin_in_use (when no PnL)
        total_value = self.portfolio['available_cash'] + total_margin_in_use + total_unrealized_pnl
        
        total_return_percent = ((total_value - self.portfolio['initial_capital']) / self.portfolio['initial_capital']) * 100
        
        win_rate = (self.portfolio['winning_trades'] / self.portfolio['total_trades'] * 100) if self.portfolio['total_trades'] > 0 else 0
        
        history_entry = {
            'timestamp': datetime.now(),
            'total_value': total_value,
            'available_cash': self.portfolio['available_cash'],
            'positions_value': total_margin_in_use + total_unrealized_pnl,  # Margin + PnL
            'return_percent': total_return_percent,
            'total_trades': self.portfolio['total_trades'],
            'win_rate': win_rate
        }
        
        self.portfolio_history.append(history_entry)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.portfolio_history) > 1000:
            self.portfolio_history.pop(0)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        return pd.DataFrame(self.portfolio_history)
    
    def get_portfolio_stats(self):
        """Get comprehensive portfolio statistics - With Leverage Support"""
        # Calculate current value
        total_unrealized_pnl = 0.0
        total_margin_in_use = 0.0
        
        for symbol, position in self.portfolio['positions'].items():
            # For stats, use entry price if generic call, or last known
            current_price = position['entry_price'] 
            quantity = position['quantity']
            entry_price = position['entry_price']
            margin_used = position.get('margin_used', position.get('position_value', 0))  # Fallback for old positions
            
            # No unrealized PnL if using entry price (price hasn't changed)
            if position['position_type'] == 'LONG':
                unrealized_pnl = 0.0
            else:
                unrealized_pnl = 0.0
                
            total_unrealized_pnl += unrealized_pnl
            total_margin_in_use += margin_used  # Use margin, not cost basis
        
        # Use simple calculation if no history or for quick check
        current_value = self.portfolio['available_cash'] + total_margin_in_use + total_unrealized_pnl
        return_percent = ((current_value - self.portfolio['initial_capital']) / self.portfolio['initial_capital']) * 100
        
        # Prefer latest history if available
        if self.portfolio_history:
            latest = self.portfolio_history[-1]
            current_value = latest['total_value']
            return_percent = latest['return_percent']

        total_pnl = current_value - self.portfolio['initial_capital']
        win_rate = (self.portfolio['winning_trades'] / self.portfolio['total_trades'] * 100) if self.portfolio['total_trades'] > 0 else 0
        
        return {
            'initial_capital': self.portfolio['initial_capital'],
            'current_value': current_value,
            'available_cash': self.portfolio['available_cash'],
            'total_return': total_pnl,
            'total_return_percent': return_percent,
            'total_trades': self.portfolio['total_trades'],
            'winning_trades': self.portfolio['winning_trades'],
            'win_rate': win_rate,
            'open_positions': len(self.portfolio['positions']),
            'trade_history': self.portfolio['trade_history'],
            'portfolio_history': self.portfolio_history,
             # Debug fields
            'unrealized_pnl': total_unrealized_pnl, 
        }
        
    def get_portfolio_summary(self, symbol: str, current_price: float) -> Dict:
        """Get portfolio summary for a specific symbol - With Leverage Support"""
        position = self.portfolio['positions'].get(symbol, {})
        
        if position:
            position_type = position['position_type']
            quantity = position['quantity']
            entry_price = position['entry_price']
            margin_used = position.get('margin_used', position.get('position_value', 0))  # Fallback
            leverage = position.get('leverage', 1.0)
            
            # Calculate unrealized PnL
            if position_type == 'LONG':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:  # SHORT
                unrealized_pnl = (entry_price - current_price) * quantity
                
            pnl_percent = (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0  # ROI on margin
            
            # Calculate liquidation price
            if leverage > 1.0:
                if position_type == 'LONG':
                    liquidation_price = entry_price * (1 - 0.95 / leverage)
                else:  # SHORT
                    liquidation_price = entry_price * (1 + 0.95 / leverage)
            else:
                liquidation_price = 0.0
        else:
            position_type = 'NONE'
            quantity = 0.0
            entry_price = 0.0
            unrealized_pnl = 0.0
            pnl_percent = 0.0
            margin_used = 0.0
            leverage = 1.0
            liquidation_price = 0.0
        
        # Calculate Total Portfolio Value
        total_margin_in_use_all = 0.0
        total_unrealized_pnl_all = 0.0
        
        for sym, pos in self.portfolio['positions'].items():
            pos_margin = pos.get('margin_used', pos.get('position_value', 0))  # Use margin
            total_margin_in_use_all += pos_margin
            
            # Use current price for analyzed symbol, entry for others (or cache if available)
            if sym == symbol:
                pos_current_price = current_price
            else:
                pos_current_price = pos['entry_price']
                
            if pos['position_type'] == 'LONG':
                pos_pnl = (pos_current_price - pos['entry_price']) * pos['quantity']
            else:
                pos_pnl = (pos['entry_price'] - pos_current_price) * pos['quantity']
                
            total_unrealized_pnl_all += pos_pnl
        
        total_value = self.portfolio['available_cash'] + total_margin_in_use_all + total_unrealized_pnl_all
        total_return_percent = ((total_value - self.portfolio['initial_capital']) / self.portfolio['initial_capital']) * 100
        
        return {
            'total_return_percent': round(total_return_percent, 2),
            'available_cash': round(self.portfolio['available_cash'], 2),
            'account_value': round(total_value, 2),
            'sharpe_ratio': 1.2, # Placeholder
            'position': {
                'symbol': symbol,
                'position_type': position_type,
                'quantity': round(quantity, 6),
                'entry_price': round(entry_price, 2),
                'current_price': round(current_price, 2),
                'unrealized_pnl': round(unrealized_pnl, 2),
                'pnl_percent': round(pnl_percent, 2),
                'leverage': leverage,
                'liquidation_price': round(liquidation_price, 2),
                'exit_plan': {
                    'profit_target': round(current_price * 1.03, 2) if position_type == 'LONG' else round(current_price * 0.97, 2),
                    'stop_loss': round(current_price * 0.97, 2) if position_type == 'LONG' else round(current_price * 1.03, 2),
                    'invalidation_condition': "RSI > 75 and price below EMA20" if position_type == 'LONG' else "RSI < 25 and price above EMA20"
                },
                'risk_usd': margin_used * 0.02  # 2% of margin at risk
            }
        }
    
    def execute_trade(self, action: str, symbol: str, current_price: float, position_size: str) -> Dict:
        """Execute a trade and update portfolio - No Leverage"""
        result = {
            'executed': False,
            'message': '',
            'position_size': 0.0,
            'quantity': 0.0,
            'position_type': None
        }
        
        # SAFETY CHECK
        current_position = self.portfolio['positions'].get(symbol, {})
        has_position = bool(current_position)
        current_position_type = current_position.get('position_type') if has_position else None
        
        # Block invalid actions
        if action == "BUY" and has_position and current_position_type == "LONG":
            return {**result, 'message': f"❌ Already LONG {symbol}. Close first."}
        
        if action == "SELL" and has_position and current_position_type == "SHORT":
            return {**result, 'message': f"❌ Already SHORT {symbol}. Close first."}
        
        # Execute OPENING trades
        if not has_position:
            # Use lot_size as exact quantity to buy/sell
            quantity = self._lot_size
            position_value = quantity * current_price
            
            # Calculate margin required with leverage
            margin_required = position_value / self.leverage
            available = self.portfolio['available_cash']
            
            # Check if we have enough margin
            if margin_required > available:
                return {**result, 'message': f"❌ Insufficient margin. Need ${margin_required:.2f}, have ${available:.2f} (leverage: {self.leverage}x)"}
            
            # Register Trade
            if action == "BUY": # LONG
                self.portfolio['positions'][symbol] = {
                    'position_type': 'LONG',
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'position_value': position_value,
                    'margin_used': margin_required,
                    'leverage': self.leverage
                }
                self.portfolio['available_cash'] -= margin_required  # Deduct margin, not full position value
                result['executed'] = True
                result['message'] = f"🟢 OPENED LONG: Bought {quantity:.6f} {symbol} at ${current_price:.2f} ({self.leverage}x leverage, margin: ${margin_required:.2f})"
                
                # Real Trading Hook
                if self.enable_real_trading:
                    print(f"🚀 [REAL TRADE REQUEST] BUY {quantity:.6f} {symbol} @ {current_price}")
                    # Broker specific API call would go here
                    
            elif action == "SELL": # SHORT
                 self.portfolio['positions'][symbol] = {
                    'position_type': 'SHORT',
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'position_value': position_value,
                    'margin_used': margin_required,
                    'leverage': self.leverage
                }
                 self.portfolio['available_cash'] -= margin_required  # Deduct margin, not full position value
                 result['executed'] = True
                 result['message'] = f"🔴 OPENED SHORT: Sold {quantity:.6f} {symbol} at ${current_price:.2f} ({self.leverage}x leverage, margin: ${margin_required:.2f})"
                 
                 # Real Trading Hook
                 if self.enable_real_trading:
                    print(f"🚀 [REAL TRADE REQUEST] SELL-SHORT {quantity:.6f} {symbol} @ {current_price}")
        
        # Execute CLOSING trades
        elif has_position:
            position = self.portfolio['positions'][symbol]
            quantity = position['quantity']
            entry_price = position['entry_price']
            position_type = position['position_type']
            margin_used = position.get('margin_used', position.get('position_value', 0))  # Fallback for old positions
            
            # Calculate PnL
            if position_type == 'LONG':
                pnl = (current_price - entry_price) * quantity
                close_action = 'SELL'
            else: # SHORT
                pnl = (entry_price - current_price) * quantity
                close_action = 'BUY'
            
            # Check validity
            if (position_type == 'LONG' and action == 'SELL') or (position_type == 'SHORT' and action == 'BUY'):
                # Return margin + PnL to available cash
                self.portfolio['available_cash'] += margin_used + pnl
                
                if pnl > 0:
                    self.portfolio['winning_trades'] += 1
                
                # Calculate return % based on margin (not position value)
                return_pct = (pnl / margin_used) * 100 if margin_used > 0 else 0
                
                # Add history
                self.portfolio['trade_history'].append({
                    'symbol': symbol,
                    'action': close_action,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now()
                })
                
                del self.portfolio['positions'][symbol]
                self.portfolio['total_trades'] += 1
                
                result['executed'] = True
                result['message'] = f"📊 CLOSED {position_type}: {close_action} {quantity:.6f} {symbol} | PnL: ${pnl:.2f} ({return_pct:+.2f}% ROI)"
                
                # Real Trading Hook
                if self.enable_real_trading:
                    print(f"🚀 [REAL TRADE REQUEST] CLOSE {position_type} {symbol}")
        
        elif action == "HOLD":
             result['message'] = f"HOLDING {symbol}"
             
        if result['executed']:
             print(f"💰 PORTFOLIO: {result['message']}")
             self.update_portfolio_history({symbol: current_price})
             
        return result
    
    def check_liquidation(self, symbol: str, current_price: float) -> bool:
        """Check if a position should be liquidated due to price hitting liquidation level"""
        if symbol not in self.portfolio['positions']:
            return False
        
        position = self.portfolio['positions'][symbol]
        entry_price = position['entry_price']
        position_type = position['position_type']
        leverage = position.get('leverage', 1.0)
        
        # If no leverage, no liquidation risk
        if leverage <= 1.0:
            return False
        
        # Calculate liquidation price
        # Liquidation happens when loss equals margin (100% of margin lost)
        if position_type == 'LONG':
            # For LONG: liquidation_price = entry_price * (1 - 1/leverage)
            # Example: 10x leverage, entry at $100 → liquidation at $90 (10% drop)
            liquidation_price = entry_price * (1 - 0.95 / leverage)  # 95% to give small buffer
            if current_price <= liquidation_price:
                print(f"⚠️ LIQUIDATION triggered for LONG {symbol} at ${current_price:.2f} (liq price: ${liquidation_price:.2f})")
                return True
        else:  # SHORT
            # For SHORT: liquidation_price = entry_price * (1 + 1/leverage)
            # Example: 10x leverage, entry at $100 → liquidation at $110 (10% rise)
            liquidation_price = entry_price * (1 + 0.95 / leverage)  # 95% to give small buffer
            if current_price >= liquidation_price:
                print(f"⚠️ LIQUIDATION triggered for SHORT {symbol} at ${current_price:.2f} (liq price: ${liquidation_price:.2f})")
                return True
        
        return False
    
    def should_force_close(self, symbol: str, current_price: float, indicators: Dict) -> bool:
        """Check if we should force close based on stop-loss or conditions"""
        if symbol not in self.portfolio['positions']:
            return False
        
        position = self.portfolio['positions'][symbol]
        entry_price = position['entry_price']
        position_type = position['position_type']
        
        # Fixed stop-loss for Spot Trading (no leverage)
        base_stop_loss = 0.03  # 3% hard stop-loss
        
        # Stop-loss check
        if position_type == 'LONG':
            stop_loss_price = entry_price * (1 - base_stop_loss)
            if current_price <= stop_loss_price:
                print(f"🛑 STOP-LOSS triggered for LONG {symbol} at ${current_price:.2f}")
                return True
        else:  # SHORT
            stop_loss_price = entry_price * (1 + base_stop_loss)
            if current_price >= stop_loss_price:
                print(f"🛑 STOP-LOSS triggered for SHORT {symbol} at ${current_price:.2f}")
                return True
        
        # Emergency close if extreme conditions
        if position_type == 'LONG' and indicators.get('rsi', 50) > 85 and current_price > entry_price * 1.05:
            print(f"🚨 Emergency close LONG: RSI extremely overbought at {indicators['rsi']:.1f}")
            return True
        elif position_type == 'SHORT' and indicators.get('rsi', 50) < 15 and current_price < entry_price * 0.95:
            print(f"🚨 Emergency close SHORT: RSI extremely oversold at {indicators['rsi']:.1f}")
            return True
        
        return False
    

    def mock_deepseek_response(self, indicators: Dict) -> Dict:
        """Mock DeepSeek API response for testing - improved logic"""
        
        # Only used if use_mock=True, but you're setting it to False
        action = "HOLD"
        confidence = 0.60
        justification = "Using real API - mock response should not be used"
        
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": f"""
REASONING TRACE:
Mock response - Real API should be used instead

MODEL OUTPUT:
ACTION: {action}
POSITION_SIZE: Maintain current position
CONFIDENCE: {confidence}
JUSTIFICATION: {justification}
RISK_MANAGEMENT: Using real API analysis
"""
                    }
                }
            ]
        }
        
        return mock_response
    
    def call_deepseek_api(self, prompt: str, indicators: Dict, symbol: str, timestamp: datetime) -> Dict:
        """Call DeepSeek API for trading analysis with better error handling"""
        
        if self.use_mock:
            print("⚠️ Using MOCK API response")
            return self.mock_deepseek_response(indicators)
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
            "stream": False
        }
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"🌐 Calling DeepSeek API (attempt {attempt + 1}/{max_retries})...")
                
                # Save the API request payload for debugging - DISABLED per user request (only on trade)
                # self.save_response_to_file({"request_payload": payload}, symbol, timestamp)
                
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=45)  # Increased timeout
                
                # Check status code first
                if response.status_code == 429:  # Rate limited
                    print(f"⏳ Rate limited, waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                    
                if response.status_code != 200:
                    print(f"❌ API Error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return {"error": f"API returned status {response.status_code}"}
                
                # Parse JSON response only if status is 200
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON response: {e}")
                    print(f"Response text: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return {"error": "Invalid JSON response from API"}
                
                # Save the API response - DISABLED per user request (only on trade)
                # self.save_response_to_file(response_data, symbol, timestamp)
                
                print("✅ API call successful")
                return response_data
                
            except requests.exceptions.Timeout:
                print(f"❌ API request timed out (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"error": "API request timed out"}
                
            except requests.exceptions.ConnectionError:
                print(f"❌ Connection error (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"error": "Connection error"}
                
            except Exception as e:
                print(f"❌ Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    def parse_trading_decision(self, api_response: Dict) -> Dict:
        """Parse the API response to extract trading decision"""
        try:
            if "error" in api_response:
                return {"error": api_response["error"]}
            
            if "choices" not in api_response or not api_response["choices"]:
                return {"error": "No choices in API response"}
            
            content = api_response["choices"][0]["message"]["content"]
            
            decision = {
                "raw_response": content,
                "reasoning": "",
                "action": "",
                "position_size": "",
                "confidence": 0.0,
                "justification": "",
                "risk_management": ""
            }
            
            # Extract reasoning trace
            reasoning_match = re.search(r'REASONING TRACE:(.*?)(?=MODEL OUTPUT:|$)', content, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                decision["reasoning"] = reasoning_match.group(1).strip()
            
            # Extract model output components
            # Improved regex to handle **BOLD** or [BRACKETS] and mixed case
            action_match = re.search(r'ACTION:\s*[*\[]*([a-zA-Z/]+)[*\]]*', content, re.IGNORECASE)
            if action_match:
                decision["action"] = action_match.group(1).strip().upper()
            
            size_match = re.search(r'POSITION_SIZE:\s*(.+)', content, re.IGNORECASE)
            if size_match:
                decision["position_size"] = size_match.group(1).strip()
            
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', content, re.IGNORECASE)
            if confidence_match:
                decision["confidence"] = float(confidence_match.group(1))
            
            justification_match = re.search(r'JUSTIFICATION:\s*(.+)', content, re.IGNORECASE)
            if justification_match:
                decision["justification"] = justification_match.group(1).strip()
            
            risk_match = re.search(r'RISK_MANAGEMENT:\s*(.+)', content, re.IGNORECASE)
            if risk_match:
                decision["risk_management"] = risk_match.group(1).strip()
            
            return decision
            
        except Exception as e:
            return {"error": f"Failed to parse response: {e}"}
    
    async def analyze_and_decide(self, ohlcv_data: pd.DataFrame, minutes_elapsed: int, symbol: str) -> Dict:
        """Main method to analyze data and get trading decision"""
        
        # Check if we have enough data
        if ohlcv_data is None or len(ohlcv_data) < 200:
            return {"error": f"Insufficient data: {len(ohlcv_data) if ohlcv_data is not None else 0} rows, need at least 200"}
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(ohlcv_data, self.ema_period, self.rsi_period)
        if not indicators:
            return {"error": "Failed to calculate technical indicators"}
        
        current_price = indicators['current_price']
        
        # Check for liquidation first (for leveraged positions)
        if self.check_liquidation(symbol, current_price):
            # Execute forced liquidation
            position = self.portfolio['positions'][symbol]
            if position['position_type'] == 'LONG':
                action = "SELL"
            else:
                action = "BUY"
                
            trade_result = self.execute_trade(action, symbol, current_price, "100%")
            return {
                "action": action,
                "confidence": 1.0,
                "justification": f"LIQUIDATED at ${current_price:.2f} - price hit liquidation level",
                "position_size": "100%",
                "risk_management": "Auto-liquidation due to insufficient margin",
                "forced_exit": True,
                "liquidation": True,
                "trade_executed": trade_result['executed'],
                "trade_message": trade_result['message']
            }
        
        # Check if we should force close (stop-loss, etc.)
        if self.should_force_close(symbol, current_price, indicators):
            # Execute forced close based on position type
            position = self.portfolio['positions'][symbol]
            if position['position_type'] == 'LONG':
                action = "SELL"
            else:
                action = "BUY"
                
            trade_result = self.execute_trade(action, symbol, current_price, "100%")
            return {
                "action": action,
                "confidence": 0.95,
                "justification": f"Stop-loss triggered at ${current_price:.2f}",
                "position_size": "100%",
                "risk_management": "Emergency exit due to stop-loss",
                "forced_exit": True,
                "trade_executed": trade_result['executed'],
                "trade_message": trade_result['message']
            }
        
        # Get account info with REAL current price and portfolio state
        account_info = self.get_portfolio_summary(symbol, current_price)
        
        # Create and save prompt using selected strategy
        # For prompt3, pass additional parameters
        # For prompt3 and prompt4, pass additional parameters
        if self.prompt_strategy in ['prompt3', 'prompt4']:
            system_prompt = self.prompt_builder(
                minutes_elapsed=minutes_elapsed,
                indicators=indicators,
                account_info=account_info,
                symbol=symbol,
                interval=self.interval,
                ema_period=self.ema_period,
                rsi_period=self.rsi_period,
                leverage=self.leverage,
                ohlcv_data=ohlcv_data,
                decision_history=self.decision_history
            )
        else:
            system_prompt = self.prompt_builder(
                minutes_elapsed=minutes_elapsed,
                indicators=indicators,
                account_info=account_info,
                symbol=symbol,
                interval=self.interval,
                ema_period=self.ema_period,
                rsi_period=self.rsi_period,
                leverage=self.leverage
            )
        timestamp = datetime.now()
        # Prompt saving moved to after decision
        
        # Call DeepSeek API (or mock)
        api_response = self.call_deepseek_api(system_prompt, indicators, symbol, timestamp)
        
        # Parse response
        decision = self.parse_trading_decision(api_response)
        
        # Save decision to history for prompt3
        # Save decision to history for prompt3 and prompt4
        if self.prompt_strategy in ['prompt3', 'prompt4']:
            decision_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': decision.get('action', 'UNKNOWN'),
                'confidence': decision.get('confidence', 0.0),
                'price': current_price,
                'reasoning': decision.get('justification', ''),  # Full REASONING TRACE for detailed analysis
            }
            self.decision_history.append(decision_record)
            
            if len(self.decision_history) > 20:
                self.decision_history.pop(0)
        
        # Handle CLOSE action mapping
        if decision.get('action') == 'CLOSE':
            if symbol in self.portfolio['positions']:
                pos_type = self.portfolio['positions'][symbol]['position_type']
                decision['action'] = 'SELL' if pos_type == 'LONG' else 'BUY'
                decision['justification'] += " (Mapped CLOSE to " + decision['action'] + ")"
            else:
                decision['action'] = 'HOLD'
                decision['justification'] += " (Ignored CLOSE - No position)"

        # Execute trade if it's BUY/SELL and not an error
        if "error" not in decision and decision.get('action') in ['BUY', 'SELL']:
            trade_result = self.execute_trade(
                decision['action'], 
                symbol, 
                current_price, 
                decision.get('position_size', '20%')
            )
            
            decision['trade_executed'] = trade_result['executed']
            decision['trade_message'] = trade_result['message']
            
            # Update decision history with outcome
            # Update decision history with outcome
            if self.prompt_strategy in ['prompt3', 'prompt4'] and self.decision_history:
                self.decision_history[-1]['outcome'] = {
                    'executed': trade_result['executed'],
                    'result': 'SUCCESS' if trade_result['executed'] else 'BLOCKED',
                    'message': trade_result['message']
                }
            
            # If trade was blocked due to safety check, override the action to HOLD
            if not trade_result['executed']:
                decision['action'] = "HOLD"
                decision['justification'] = f"Safety blocked {decision['action']}: {trade_result['message']}"
        
        # --- LOGGING & SAVING ---
        
        # 1. CSV Logging (Every Step)
        last_candle = ohlcv_data.iloc[-1]
        pos_status = "OPEN" if symbol in self.portfolio['positions'] else "NONE"
        if symbol in self.portfolio['positions']:
            pos_status = f"{self.portfolio['positions'][symbol]['position_type']} ({self.portfolio['positions'][symbol]['quantity']:.4f})"
            
        self.log_to_csv(
            timestamp=timestamp,
            symbol=symbol,
            ohlcv_row=last_candle,
            prompt_name=self.prompt_strategy,
            portfolio_value=self.get_portfolio_stats()['current_value'],
            action=decision.get('action', 'UNKNOWN'),
            position_status=pos_status,
            reason=decision.get('justification', '').replace('\n', ' ')
        )
        
        # 2. Prompt/Response Saving - DISABLED COMPLETELY
        # if decision.get('action') in ['BUY', 'SELL']:
        #     print(f"💾 Saving prompt/response for {decision['action']} decision...")
        #     self.save_prompt_to_file(system_prompt, "", symbol, timestamp)
        #     self.save_response_to_file(api_response, symbol, timestamp)

        return decision

    
    
    def print_portfolio_status(self):
        """Print current portfolio status"""
        stats = self.get_portfolio_stats()
        print(f"\n📊 PORTFOLIO STATUS:")
        print(f"   Initial Capital: ${stats['initial_capital']:.2f}")
        print(f"   Current Value: ${stats['current_value']:.2f}")
        print(f"   Total Return: ${stats['total_return']:.2f} ({stats['total_return_percent']:.2f}%)")
        print(f"   Available Cash: ${stats['available_cash']:.2f}")
        print(f"   Open Positions: {stats['open_positions']}")
        print(f"   Total Trades: {stats['total_trades']}")
        if stats['total_trades'] > 0:
            print(f"   Win Rate: {stats['win_rate']:.1f}%")
        
        if self.portfolio['positions']:
            print(f"\n   📈 OPEN POSITIONS:")
            for symbol, position in self.portfolio['positions'].items():
                current_value = position['quantity'] * position['entry_price']  # Simplified
                pnl = (position['entry_price'] - position['entry_price']) * position['quantity']  # Should use current price
                print(f"      {symbol}: {position['quantity']:.6f} @ ${position['entry_price']:.2f}")