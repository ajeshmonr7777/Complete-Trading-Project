"""
Prompt Builder for Trading AI
Creates system prompts for DeepSeek API based on market data and portfolio state
"""

from typing import Dict


def create_system_prompt(
    minutes_elapsed: int,
    indicators: Dict,
    account_info: Dict,
    symbol: str,
    interval: str = "1m",
    ema_period: int = 20,
    rsi_period: int = 14,
    leverage: float = 1.0,
    **kwargs
) -> str:
    """
    Create the system prompt for DeepSeek API
    
    Args:
        minutes_elapsed: Minutes since trading started
        indicators: Dictionary of technical indicators
        account_info: Portfolio and position information
        symbol: Trading symbol (e.g., BTCUSDT)
        interval: Timeframe interval (default: 1m)
        ema_period: EMA period used (default: 20)
        rsi_period: RSI period used (default: 14)
        leverage: Leverage multiplier (default: 1.0)
    
    Returns:
        Formatted system prompt string
    """
    
    # Create market structure summary
    market_structure = []
    if indicators.get('above_ema20'):
        market_structure.append("Above EMA20")
    if indicators.get('above_ema50'):
        market_structure.append("Above EMA50")
    if indicators.get('above_ema200'):
        market_structure.append("Above EMA200")
    
    market_structure_summary = " | ".join(market_structure) if market_structure else "Below key EMAs"
    
    # Determine position context
    has_position = account_info['position']['quantity'] > 0
    position_type = account_info['position']['position_type']
    current_pnl = account_info['position']['unrealized_pnl']
    current_pnl_percent = account_info['position']['pnl_percent']
    
    if has_position:
        if position_type == 'LONG':
            position_context = f"""
🚨 **CURRENT STATUS: HOLDING LONG POSITION** 🚨
You currently hold LONG {account_info['position']['quantity']:.6f} {symbol}
- Entry Price: ${account_info['position']['entry_price']:.2f}
- Current Price: ${account_info['position']['current_price']:.2f}
- Unrealized PnL: ${current_pnl:.2f} ({current_pnl_percent:.2f}%)
- Profit when price INCREASES

**TRADING CONSTRAINT: Since you already have an open LONG position, you CANNOT open another position.**
**Available actions: SELL (close long for profit/loss) or HOLD (maintain long)**
"""
        else:  # SHORT
            position_context = f"""
🚨 **CURRENT STATUS: HOLDING SHORT POSITION** 🚨  
You currently hold SHORT {account_info['position']['quantity']:.6f} {symbol}
- Entry Price: ${account_info['position']['entry_price']:.2f}
- Current Price: ${account_info['position']['current_price']:.2f}  
- Unrealized PnL: ${current_pnl:.2f} ({current_pnl_percent:.2f}%)
- Profit when price DECREASES

**TRADING CONSTRAINT: Since you already have an open SHORT position, you CANNOT open another position.**
**Available actions: BUY (close short for profit/loss) or HOLD (maintain short)**
"""
    else:
        position_context = f"""
**POSITION STATUS: No open position in {symbol}**
**Mode: Spot Trading (1x Leverage)**
**Available actions: BUY (open LONG position) or SELL (open SHORT position) or HOLD (wait)**
**LONG**: Profit when price increases | **SHORT**: Profit when price decreases
"""

    system_prompt = f"""
You are a quantitative trading analyst. Your task is to analyze live market data for {symbol} and your current portfolio to make disciplined, data-driven trading decisions.

It has been {minutes_elapsed} minutes since you started trading. You are being provided with state data, price data, and predictive signals for {symbol}.

**CRITICAL DATA CONTEXT:**
*   **ALL PRICE AND SIGNAL DATA IS ORDERED: OLDEST → NEWEST**
*   **Timeframes note:** Intraday series are provided at **{interval} intervals**.
*   **POSITION CONSTRAINT: You can only hold ONE position (LONG or SHORT) per symbol at a time**
*   **TRADING MODE: Spot Trading (No Leverage)**

{position_context}

--- 
### CURRENT MARKET STATE FOR {symbol} ### 

**Price & Trend Analysis:**
Current Price: ${indicators['current_price']:.2f}
Market Structure: {market_structure_summary}
RSI Level: {indicators.get('rsi_level', 'NEUTRAL')}
MACD Trend: {indicators.get('macd_trend', 'NEUTRAL')}

**Technical Indicators:**
20-Period EMA: ${indicators['ema_20']:.2f}
50-Period EMA: ${indicators['ema_50']:.2f}
200-Period EMA: ${indicators['ema_200']:.2f}
MACD: {indicators['macd']:.4f} (Signal: {indicators['macd_signal']:.4f})
RSI ({rsi_period} period): {indicators['rsi']:.2f}
Bollinger Bands Position: {indicators.get('bb_position', 0):.2%}

**Market Metrics:**
Open Interest - Latest: {indicators['oi_latest']}, Average: {indicators['oi_average']}
Funding Rate: {indicators['funding_rate']:.6f}

**Intraday Series ({interval} intervals, oldest → latest):**
Mid Prices (last 10): {[f'${x:.1f}' for x in indicators['price_series'][-10:]]}
EMA{ema_period} Series (last 10): {[f'${x:.1f}' for x in indicators['ema_series'][-10:]]}
MACD Series (last 10): {[f'{x:.4f}' for x in indicators['macd_series'][-10:]]}
RSI Series (last 10): {[f'{x:.2f}' for x in indicators['rsi_series'][-10:]]}

--- 
### ACCOUNT INFORMATION & PERFORMANCE ### 
Current Total Return: {account_info['total_return_percent']}%
Available Cash: ${account_info['available_cash']:.2f}
Current Account Value: ${account_info['account_value']:.2f}
Sharpe Ratio: {account_info['sharpe_ratio']:.2f}

**Current {symbol} Position:**
Position Type: {account_info['position']['position_type']}
Quantity: {account_info['position']['quantity']:.6f}
Entry Price: ${account_info['position']['entry_price']:.2f}
Current Price: ${account_info['position']['current_price']:.2f}
Unrealized PnL: ${account_info['position']['unrealized_pnl']:.2f} ({account_info['position']['pnl_percent']:.2f}%)
Risk: ${account_info['position']['risk_usd']:.2f}

**TRADING DECISION FRAMEWORK:**
1. **FIRST: Check current position type (LONG/SHORT/NONE)**
   - If LONG: Consider SELL (close long) or HOLD (maintain long)
   - If SHORT: Consider BUY (close short) or HOLD (maintain short)  
   - If NONE: Consider BUY (open long) or SELL (open short) or HOLD (wait)
2. Analyze trend direction using EMA alignment and MACD
3. Assess momentum with RSI and volume
4. Evaluate risk-reward
5. Consider market structure and key levels
6. Provide clear position sizing (~20% of capital for high conviction)

**IMPORTANT: You cannot add to existing positions - only one position per symbol allowed.**

Generate the Reasoning Trace and final Model Output (action) based on the data above.

**REQUIRED OUTPUT FORMAT:**
REASONING TRACE:
[Your step-by-step analysis]

MODEL OUTPUT:
ACTION: [BUY/SELL/HOLD/CLOSE]
POSITION_SIZE: [Percentage e.g. 20%]
CONFIDENCE: [0.0-1.0]
JUSTIFICATION: [Clear explanation]
RISK_MANAGEMENT: [Stop-loss, take-profit]
"""
    return system_prompt
