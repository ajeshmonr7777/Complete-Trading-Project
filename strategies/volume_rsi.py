import pandas as pd
import numpy as np

def strategy(df):
    """
    Volume + RSI Reversal Strategy
    
    Logic:
    1. BUY when:
       - RSI is Oversold (< 35)
       - Volume is spiking (> 1.5x of 20-period average)
       - Limit: Only buy if price closed higher (Green candle) indicating reversal start.
       
    2. SELL when:
       - RSI becomes Overbought (> 70)
       
    Returns:
        DataFrame with 'Signal' column (1=Buy, -1=Sell, 0=Hold)
    """
    df = df.copy()
    
    # --- 1. RSI Calculation (14 Period) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Wilder's Smoothing for RSI
    avg_gain = gain.ewm(com=13, min_periods=13).mean()
    avg_loss = loss.ewm(com=13, min_periods=13).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- 2. Volume Analysis ---
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Spike'] = df['Volume'] > (df['Vol_MA20'] * 1.5)
    
    # --- 3. Signal Generation ---
    df['Signal'] = 0
    
    # Buy Condition
    # RSI low + High Volume + Price Up (Green Candle)
    buy_cond = (
        (df['RSI'] < 35) & 
        (df['Vol_Spike']) & 
        (df['Close'] > df['Open']) 
    )
    
    # Sell Condition
    # RSI Overbought
    sell_cond = (df['RSI'] > 70)
    
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    
    return df
