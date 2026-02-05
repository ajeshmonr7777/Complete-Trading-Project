def strategy(df):
    """
    Simple Moving Average Crossover Strategy
    """
    import pandas as pd
    
    # Calculate Indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Generate Signals
    df['Signal'] = 0
    
    # Buy when SMA_50 crosses above SMA_200
    # Use int conversion for vectorized logical operations if needed, or simple applying
    
    # Logic: 
    # 1 where (SMA50 > SMA200) AND (SMA50.shift(1) <= SMA200.shift(1)) -> Golden Cross (Buy)
    # -1 where (SMA50 < SMA200) AND (SMA50.shift(1) >= SMA200.shift(1)) -> Death Cross (Sell)
    
    buy_signal = ((df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))).fillna(False)
    sell_signal = ((df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))).fillna(False)
    
    import numpy as np
    df['Signal'] = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return df
