"""
Live AI Trading Server - FastAPI endpoints for AI-powered crypto trading
Adapted from pnrl_finrl_project/server.py for integration with backtesting dashboard
"""

import asyncio
import threading
import time
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from .analyst import TradingAnalyst
from .streamer import BinanceOHLCVStreamer, UpstoxOHLCVStreamer
from .prompt_loader import PromptLoader

from dotenv import load_dotenv

# Load env vars
load_dotenv()

# --- Data Models ---
class InitConfig(BaseModel):
    deepseek_api_key: Optional[str] = None
    broker: str  # "binance", "zerodha", "ibkr", "paper"
    trading_mode: str  # "real" or "paper"
    amount: float = 1000.0
    lot_size: Optional[float] = 0.01  # Position size multiplier
    leverage: Optional[float] = 10.0  # Leverage multiplier (default: 10x)
    prompt_strategy: Optional[str] = "default_strategy"
    interval: Optional[str] = "1m"
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None
    zerodha_api_key: Optional[str] = None
    zerodha_api_secret: Optional[str] = None
    upstox_api_key: Optional[str] = None
    upstox_api_secret: Optional[str] = None
    upstox_redirect_uri: Optional[str] = None
    upstox_access_token: Optional[str] = None
    ibkr_username: Optional[str] = None
    ibkr_password: Optional[str] = None
    symbols: List[str]

class GlobalState:
    def __init__(self):
        self.trading_analyst: Optional[TradingAnalyst] = None
        self.streamer: Optional[BinanceOHLCVStreamer] = None
        self.running: bool = False
        self.stream_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None

state = GlobalState()

# Create FastAPI app
live_app = FastAPI(title="Live AI Trading API")

# Enable CORS
live_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def run_stream_in_thread(streamer):
    """Helper to run the async streamer in a separate thread"""
    asyncio.run(streamer.start_streaming(fetch_historical=True, historical_limit=250))

# --- API Endpoints ---

@live_app.post("/start")
async def start_trading(config: InitConfig):
    """Initialize and start the trading system"""
    if state.running:
        return {"status": "already_running"}
    
    try:
        # data validation
        if not config.symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")

        if not config.deepseek_api_key:
            config.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not config.deepseek_api_key:
                 raise HTTPException(status_code=400, detail="DeepSeek API Key not provided and not found in .env")

        # Initialize Trading Analyst
        state.trading_analyst = TradingAnalyst(
            deepseek_api_key=config.deepseek_api_key,
            use_mock=False,
            initial_capital=config.amount,
            leverage=config.leverage if hasattr(config, 'leverage') else 10.0,
            enable_real_trading=(config.trading_mode == "real"),
            broker_config={
                "broker": config.broker,
                "binance": {
                    "api_key": config.binance_api_key or os.getenv("BINANCE_API_KEY"),
                    "secret_key": config.binance_secret_key or os.getenv("BINANCE_SECRET_KEY")
                },
                "zerodha": {
                    "api_key": config.zerodha_api_key or os.getenv("ZERODHA_API_KEY"),
                    "api_secret": config.zerodha_api_secret or os.getenv("ZERODHA_API_SECRET")
                },
                "upstox": {
                    "api_key": config.upstox_api_key or os.getenv("UPSTOX_API_KEY"),
                    "api_secret": config.upstox_api_secret or os.getenv("UPSTOX_API_SECRET"),
                    "redirect_uri": config.upstox_redirect_uri or os.getenv("UPSTOX_REDIRECT_URI"),
                    "access_token": config.upstox_access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
                }
            }
        )
        
        # Set prompt strategy
        if hasattr(config, 'prompt_strategy') and config.prompt_strategy:
            state.trading_analyst.prompt_strategy = config.prompt_strategy
            
        # Set interval
        if hasattr(config, 'interval') and config.interval:
            state.trading_analyst.interval = config.interval

        # Set lot size if provided
        if hasattr(config, 'lot_size') and config.lot_size:
            state.trading_analyst.lot_size = config.lot_size
        
        # Initialize Binance Streamer
        # Use config keys first, fall back to environment variables
        binance_api_key = config.binance_api_key or os.getenv("BINANCE_API_KEY")
        binance_secret_key = config.binance_secret_key or os.getenv("BINANCE_SECRET_KEY")

        # Initialize Streamer based on broker
        if config.broker == "upstox":
            upstox_access_token = config.upstox_access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
            state.streamer = UpstoxOHLCVStreamer(
                symbols=config.symbols,
                api_key=config.upstox_api_key,
                secret_key=config.upstox_api_secret,
                trading_analyst=state.trading_analyst,
                access_token=upstox_access_token,
                interval=config.interval
            )
        else:
            state.streamer = BinanceOHLCVStreamer(
                symbols=config.symbols,
                api_key=binance_api_key,
                secret_key=binance_secret_key,
                trading_analyst=state.trading_analyst,
                interval=config.interval
            )
        
        # Start streaming in background thread
        def run_stream():
            asyncio.run(state.streamer.start_streaming(fetch_historical=True, interval=config.interval))
            
        state.stream_thread = threading.Thread(target=run_stream, daemon=True)
        state.stream_thread.start()
        
        state.running = True
        state.start_time = time.time()
        
        return {"status": "started", "symbols": config.symbols, "mode": config.trading_mode}
        
    except Exception as e:
        state.running = False
        raise HTTPException(status_code=500, detail=str(e))

@live_app.post("/stop")
async def stop_trading():
    """Stop the trading system"""
    if not state.running:
        return {"status": "not_running"}
    
    try:
        if state.streamer:
            state.streamer.stop_streaming()
        
        if state.stream_thread:
            state.stream_thread.join(timeout=2.0)
        
        # Get final stats before stopping
        final_stats = None
        if state.trading_analyst:
            final_stats = state.trading_analyst.get_portfolio_stats()
        
        state.running = False
        
        return {
            "status": "stopped",
            "message": "Trading system stopped successfully",
            "final_stats": final_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop trading system: {str(e)}")

@live_app.get("/prompts")
async def get_prompts():
    """Get list of available live trading prompts"""
    loader = PromptLoader()
    return {"prompts": list(loader.prompts.keys())}

@live_app.get("/status")
def get_status():
    """Get current trading status and metrics"""
    if not state.running or not state.trading_analyst:
        return {"status": "stopped", "running": False}
    
    try:
        # Get current prices from streamer
        current_prices = {}
        if state.streamer:
            for symbol in state.streamer.symbols:
                ohlcv_data = state.streamer.get_ohlcv_data(symbol)
                if not ohlcv_data.empty:
                    current_prices[symbol] = float(ohlcv_data.iloc[-1]['close'])
        
        # Update portfolio history with current prices
        if state.trading_analyst:
            state.trading_analyst.update_portfolio_history(current_prices)

        stats = state.trading_analyst.get_portfolio_stats()
        
        # Get positions as dict (not array) for frontend compatibility
        current_positions = {}
        for symbol, position in state.trading_analyst.portfolio['positions'].items():
            current_price = current_prices.get(symbol, position['entry_price'])
            
            # Calculate unrealized PnL
            if position['position_type'] == 'LONG':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:  # SHORT
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            cost_basis = position['quantity'] * position['entry_price']
            
            current_positions[symbol] = {
                "position_type": position['position_type'],
                "quantity": round(position['quantity'], 6),
                "entry_price": round(position['entry_price'], 2),
                "current_price": round(current_price, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "cost_basis": round(cost_basis, 2)
            }
        
        # Get portfolio history with timestamp and active position check
        portfolio_history = []
        for entry in stats['portfolio_history']:
            portfolio_history.append({
                'value': entry['total_value'],
                'timestamp': entry['timestamp'].isoformat() if hasattr(entry['timestamp'], 'isoformat') else str(entry['timestamp']),
                'in_position': bool(entry.get('positions_value', 0) != 0)
            })
        
        # Get recent decisions from decision_history
        recent_decisions = []
        for decision in state.trading_analyst.decision_history[-10:]:  # Last 10 decisions
            recent_decisions.append({
                "timestamp": decision['timestamp'].isoformat(),
                "symbol": decision['symbol'],
                "action": decision['action'],
                "price": round(decision['price'], 2),
                "confidence": round(decision['confidence'], 2),
                "reasoning": decision['reasoning'] if decision['reasoning'] else "N/A"
            })
            
        # Get recent completed trades
        recent_trades = []
        for trade in state.trading_analyst.portfolio['trade_history'][-10:]:
            recent_trades.append({
                "symbol": trade['symbol'],
                "position_type": trade['action'], # CLOSE_LONG -> SELL, CLOSE_SHORT -> BUY usually, but UI can parse
                "entry_time": trade['entry_time'].isoformat(),
                "entry_price": round(trade.get('entry_price', 0), 2),
                "exit_time": trade['exit_time'].isoformat(),
                "exit_price": round(trade.get('exit_price', 0), 2),
                "pnl": round(trade['pnl'], 2),
                "roi_percent": round(trade.get('return_pct', 0), 2)
            })
        
        # Get recent market data for candlestick chart (last 50 candles)
        market_data = {"candles": [], "timestamps": []}
        if state.streamer and state.streamer.symbols:
            symbol = list(state.streamer.symbols)[0]  # Use first symbol
            ohlcv_data = state.streamer.get_ohlcv_data(symbol)
            if not ohlcv_data.empty:
                # Get last 50 data points with OHLC
                recent_data = ohlcv_data.tail(50)
                market_data["timestamps"] = [ts.isoformat() for ts in recent_data['timestamp']]
                # Send OHLC data for candlestick chart
                market_data["candles"] = [
                    {
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close'])
                    }
                    for _, row in recent_data.iterrows()
                ]
        
        # Get recent trades for chart markers (all completed trades)
        trade_markers = []
        for trade in state.trading_analyst.portfolio['trade_history']:
            # Add entry marker
            trade_markers.append({
                "timestamp": trade['entry_time'].isoformat(),
                "price": float(trade.get('entry_price', 0)),
                "action": "BUY" if trade['action'] == 'CLOSE_LONG' else "SHORT_ENTRY",
                "symbol": trade['symbol']
            })
            # Add exit marker
            trade_markers.append({
                "timestamp": trade['exit_time'].isoformat(),
                "price": float(trade.get('exit_price', 0)),
                "action": "SELL" if trade['action'] == 'CLOSE_LONG' else "SHORT_EXIT",
                "symbol": trade['symbol']
            })
        
        return {
            "running": state.running,
            "uptime_seconds": int(time.time() - state.start_time) if state.start_time else 0,
            "portfolio_value": round(stats['current_value'], 2),
            "cash_balance": round(stats['available_cash'], 2),
            "initial_value": stats['initial_capital'],
            "trades_count": stats['total_trades'],
            "portfolio_history": portfolio_history,
            "current_positions": current_positions,
            "recent_decisions": recent_decisions,
            "recent_trades": recent_trades,
            "market_data": market_data,
            "trade_markers": trade_markers,
            "metrics": {
                "portfolio_value": round(stats['current_value'], 2),
                "available_cash": round(stats['available_cash'], 2),
                "total_return_percent": round(stats['total_return_percent'], 2),
                "total_trades": stats['total_trades'],
                "win_rate": round(stats['win_rate'], 2)
            },
            "symbols": list(state.streamer.symbols) if state.streamer else [],
            "currency_symbol": state.trading_analyst.currency_symbol if state.trading_analyst else "$"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@live_app.get("/history")
async def get_history():
    """Get portfolio value history"""
    if not state.trading_analyst:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    try:
        history_df = state.trading_analyst.get_portfolio_history()
        
        # Convert to list of dicts for JSON response
        history = []
        for _, row in history_df.iterrows():
            history.append({
                "timestamp": row['timestamp'].isoformat(),
                "total_value": round(row['total_value'], 2),
                "available_cash": round(row['available_cash'], 2),
                "positions_value": round(row.get('positions_value', 0), 2),
                "return_percent": round(row['return_percent'], 2)
            })
        
        return {"history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@live_app.get("/trades")
async def get_trades():
    """Get trade history"""
    if not state.trading_analyst:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    try:
        trades = []
        for trade in state.trading_analyst.portfolio['trade_history']:
            # Calculate ROI based on cost basis (margin_used was leveraged)
            # trade['pnl'] is absolute
            # trade['return_pct'] should be available
            
            trades.append({
                "symbol": trade['symbol'],
                "position_type": trade['action'], # CLOSE BUY/SELL
                "leverage": 1.0,
                "entry_price": round(trade.get('entry_price', 0), 2),
                "exit_price": round(trade.get('exit_price', 0), 2),
                "pnl": round(trade['pnl'], 2),
                "roi_percent": round(trade.get('return_pct', 0), 2),
                "entry_time": trade['entry_time'].isoformat(),
                "exit_time": trade['exit_time'].isoformat(),
                "holding_period_minutes": round(trade.get('holding_period', 0), 1)
            })
        
        return {"trades": trades}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trades: {str(e)}")

@live_app.post("/close_all")
async def close_all_positions():
    """Force close all positions"""
    if not state.trading_analyst or not state.streamer:
        raise HTTPException(status_code=400, detail="Trading system not initialized")
    
    try:
        closed_positions = []
        
        # Get current prices
        current_prices = {}
        for symbol in state.streamer.symbols:
            ohlcv_data = state.streamer.get_ohlcv_data(symbol)
            if not ohlcv_data.empty:
                current_prices[symbol] = float(ohlcv_data.iloc[-1]['close'])
        
        # Close all positions
        for symbol in list(state.trading_analyst.portfolio['positions'].keys()):
            position = state.trading_analyst.portfolio['positions'][symbol]
            current_price = current_prices.get(symbol, position['entry_price'])
            
            # Determine close action based on position type
            if position['position_type'] == 'LONG':
                action = 'SELL'
            else:
                action = 'BUY'
            
            # Execute close trade
            result = state.trading_analyst.execute_trade(action, symbol, current_price, "100%")
            
            if result['executed']:
                closed_positions.append({
                    "symbol": symbol,
                    "position_type": position['position_type'],
                    "message": result['message']
                })
        
        return {
            "status": "success",
            "closed_count": len(closed_positions),
            "closed_positions": closed_positions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to close positions: {str(e)}")

@live_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "running": state.running,
        "has_analyst": state.trading_analyst is not None,
        "has_streamer": state.streamer is not None
    }
