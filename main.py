from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import importlib.util
from backtesting.engine import BacktestEngine
from shared.converters import PineScriptParser
import datetime
import shutil

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import live trading server
from live_trading.server import live_app


app = FastAPI(title="Backtesting API")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend_static"), name="static")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount live trading API as sub-application
app.mount("/api/live", live_app)

@app.get("/")
def read_root():
    return FileResponse('frontend_static/index.html')

@app.get("/app")
def read_app():
    return FileResponse('frontend_static/app.html')

class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    interval: str
    strategy_type: str  # "saved" or "uploaded" (handled via separate flow usually, but we can simplify)
    strategy_file: str # Filename if saved
    stop_loss_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    max_active_trades: Optional[int] = 1 # Default 1 (Single Mode)
    exit_on_signal: Optional[bool] = True # Default True (Standard Logic)

import ast

class StrategyUpdate(BaseModel):
    code: str

def extract_docstring(file_path):
    """Extract docstring from a python file"""
    try:
        with open(file_path, "r") as f:
            node = ast.parse(f.read())
            return ast.get_docstring(node) or "No description available."
    except Exception:
        return "Description unavailable."

@app.get("/strategies")
def list_strategies():
    strategies_dir = os.path.join("backtesting", "strategies")
    os.makedirs(strategies_dir, exist_ok=True)
    strategies = []
    
    files = [f for f in os.listdir(strategies_dir) if f.endswith((".py", ".pine", ".txt"))]
    
    for f in files:
        # Generate clean name (remove extension and underscores)
        clean_name = f.rsplit('.', 1)[0].replace('_', ' ').title()
        description = "No description available."
        
        # Extract docstring for Python files
        if f.endswith('.py'):
            path = os.path.join(strategies_dir, f)
            description = extract_docstring(path)
            
        strategies.append({
            "filename": f,
            "name": clean_name,
            "description": description
        })
        
    return {"strategies": strategies}

@app.get("/strategies/{filename}")
def get_strategy_content(filename: str):
    strategies_dir = os.path.join("backtesting", "strategies")
    file_path = os.path.join(strategies_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Strategy not found")
        
    with open(file_path, "r") as f:
        content = f.read()
        
    return {"filename": filename, "code": content}

@app.put("/strategies/{filename}")
def update_strategy_content(filename: str, update: StrategyUpdate):
    strategies_dir = os.path.join("backtesting", "strategies")
    file_path = os.path.join(strategies_dir, filename)
    
    # Security check: prevent directory traversal
    if ".." in filename or "/" in filename:
         raise HTTPException(status_code=400, detail="Invalid filename")
         
    try:
        with open(file_path, "w") as f:
            f.write(update.code)
        return {"message": "Strategy updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- AI Strategy Generator Endpoints ---
from backtesting.generator import generate_strategy_from_text_stream, generate_strategy_from_youtube_stream, save_strategy_file

from fastapi.responses import StreamingResponse
import json
import asyncio

class GenerateStrategyRequest(BaseModel):
    type: str  # "text" or "youtube"
    input: str # Description or URL

class SaveStrategyRequest(BaseModel):
    name: str
    code: str

@app.post("/api/generate-strategy")
async def generate_strategy_endpoint(request: GenerateStrategyRequest):
    """Generate strategy code from text or YouTube - Streaming"""
    
    # Use generator function based on type
    if request.type == 'text':
        generator = generate_strategy_from_text_stream(request.input)
    elif request.type == 'youtube':
        generator = generate_strategy_from_youtube_stream(request.input)
    else:
        raise HTTPException(status_code=400, detail="Invalid strategy type")

    # Wrapper to stream JSON lines
    async def stream_generator():
        try:
             # Iterate over the sync generator
             for item in generator:
                 yield json.dumps(item) + "\n"
                 await asyncio.sleep(0.01) 
                 
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.post("/api/save-strategy")
async def save_strategy_endpoint(request: SaveStrategyRequest):
    try:
        if not request.name.endswith('.py'):
            request.name += '.py'
            
        filename = save_strategy_file(request.code, request.name.replace('.py', ''))
        return {"filename": filename, "message": "Strategy saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategies/upload")
async def upload_strategy(file: UploadFile = File(...)):
    strategies_dir = os.path.join("backtesting", "strategies")
    os.makedirs(strategies_dir, exist_ok=True)
    file_path = os.path.join(strategies_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.post("/backtest")
def run_backtest(request: BacktestRequest):
    try:
        print(f"Received Backtest Request: Tickers={len(request.tickers)}, SL={request.stop_loss_pct}, MaxTrades={request.max_active_trades}, ExitOnSignal={request.exit_on_signal}")
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        strategies_dir = os.path.join("backtesting", "strategies")
        file_path = os.path.join(strategies_dir, request.strategy_file)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Strategy file not found")
            
        # Strategy Loading (Load once)
        strategy_func = None
        if file_path.endswith(".py"):
            spec = importlib.util.spec_from_file_location("strategy_module", file_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            if hasattr(strategy_module, "strategy"):
                strategy_func = strategy_module.strategy
            else:
                 raise HTTPException(status_code=400, detail="Python file must contain a 'strategy(data)' function.")
        else:
             # Pine
             with open(file_path, "r") as f:
                 pine_code = f.read()
             parser = PineScriptParser(pine_code)
             strategy_func = parser.parse()

        if strategy_func is None:
             raise HTTPException(status_code=400, detail="Could not load strategy function.")

        # Portfolio Variables
        num_tickers = len(request.tickers)
        if num_tickers == 0:
             raise HTTPException(status_code=400, detail="No tickers provided.")
             
        # "Independent Analysis": Run each with FULL capital
        capital_per_ticker = request.initial_capital
        
        individual_results = {}
        all_equity_curves = []
        all_trades = []
        
        for ticker in request.tickers:
            try:
                engine = BacktestEngine(ticker, start_date, end_date, capital_per_ticker, request.interval)
                data = engine.load_data()
                
                if data is None or data.empty:
                    # Capture failure for this ticker but continue
                    individual_results[ticker] = {"error": "No Data"}
                    continue
                    
                results = engine.run_strategy(
                    strategy_func, 
                    stop_loss_pct=request.stop_loss_pct, 
                    risk_reward_ratio=request.risk_reward_ratio,
                    max_active_trades=request.max_active_trades,
                    exit_on_signal=request.exit_on_signal
                )
                
                if results:
                    # Process Individual Logic
                    eq = results['equity_curve']
                    tr = results['trades']
                    
                    # Store Equity for Aggregation
                    all_equity_curves.append(eq)
                    
                    # Process Trades for Aggregation
                    if not tr.empty:
                        tr['Symbol'] = ticker # Add Symbol column
                        all_trades.append(tr)
                        
                    # Format individual for response
                    metrics = engine.calculate_metrics(results)
                    
                    # Date formatting helper
                    def safer_date(x):
                         return str(x) if isinstance(x, (pd.Timestamp, datetime.date, datetime.datetime)) else x
                         
                    ind_trades_list = []
                    if not tr.empty:
                         ind_raw = tr.to_dict('records')
                         for t in ind_raw:
                             clean_t = {k: safer_date(v) for k, v in t.items()}
                             ind_trades_list.append(clean_t)
                             
                    ind_equity_data = [{"time": str(t), "value": v} for t, v in eq.items()]
                    
                    individual_results[ticker] = {
                        "metrics": metrics,
                        "equity_curve": ind_equity_data,
                        "trades": ind_trades_list
                    }
                else:
                    individual_results[ticker] = {"error": "Strategy returned None"}
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                individual_results[ticker] = {"error": str(e)}

        # Aggregation Logic
        if not all_equity_curves:
             raise HTTPException(status_code=500, detail="No simulation succeeded for any ticker.")
             
        # Generate Summary List for Table
        summary_list = []
        for ticker, res in individual_results.items():
            if "error" not in res:
                metrics = res['metrics']
                summary_list.append({
                    "Symbol": ticker,
                    "Total Return": metrics['total_return'],
                    "Final Equity": metrics['final_equity'],
                    "Trades": metrics['num_trades']
                })
        
        # Sort by Return (High to Low)
        summary_list.sort(key=lambda x: x['Total Return'], reverse=True)

        # Combine equity curves
        portfolio_equity_df = pd.concat(all_equity_curves, axis=1)
        
        # Forward fill missing data and fill failures/gaps with initial capital
        # ensuring consistent timestamps across the average
        portfolio_equity_df.ffill(inplace=True)
        portfolio_equity_df.fillna(request.initial_capital, inplace=True)
        
        # Calculate AVERAGE Equity Curve (Average Performance)
        portfolio_equity_curve = portfolio_equity_df.mean(axis=1)
        
        # Portfolio Metrics (Based on the Average Curve)
        total_final_equity = portfolio_equity_curve.iloc[-1]
        total_return = ((total_final_equity - request.initial_capital) / request.initial_capital) * 100
        
        # Aggregate Trades
        portfolio_trades_list = []
        portfolio_win_rate = 0.0
        
        if all_trades:
            combined_trades = pd.concat(all_trades)
            # Sort by entry time
            combined_trades.sort_values('Entry Time', inplace=True)
            
            # Calculate Portfolio Win Rate
            total_p_trades = len(combined_trades)
            if total_p_trades > 0:
                p_wins = len(combined_trades[combined_trades['PnL'] > 0])
                portfolio_win_rate = (p_wins / total_p_trades) * 100
            
            # Format
            raw_pt = combined_trades.to_dict('records')
            for t in raw_pt:
                 clean_t = {k: safer_date(v) for k, v in t.items()}
                 portfolio_trades_list.append(clean_t)
                 
        portfolio_equity_data = [{"time": str(t), "value": v} for t, v in portfolio_equity_curve.items()]
        
        response_data = {
            "portfolio": {
                "metrics": {
                    "total_return": total_return,
                    "final_equity": total_final_equity,
                    "num_trades": len(portfolio_trades_list),
                    "win_rate": portfolio_win_rate
                },
                "summary": summary_list, # New Field
                "equity_curve": portfolio_equity_data,
                "trades": portfolio_trades_list
            },
            "individual": individual_results
        }
        
        return response_data

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
