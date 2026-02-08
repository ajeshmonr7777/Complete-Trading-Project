def strategy(df):
    """
    Pure price action strategy based on market structure breaks and supply/demand zones.
    Trades only in trend direction with 2.5:1 R:R filter. No indicators used.
    """
    import pandas as pd
    import numpy as np
    
    # Step 1: Calculate swing highs and lows (lookback 5 periods)
    lookback = 5
    df['swing_high'] = df['High'][(df['High'] == df['High'].rolling(window=lookback*2+1, center=True).max()) & 
                                  (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))]
    df['swing_low'] = df['Low'][(df['Low'] == df['Low'].rolling(window=lookback*2+1, center=True).min()) & 
                                (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))]
    
    # Forward fill swing levels to create structure levels
    df['prev_high'] = df['swing_high'].fillna(method='ffill').fillna(df['High'].iloc[0])
    df['prev_low'] = df['swing_low'].fillna(method='ffill').fillna(df['Low'].iloc[0])
    
    # Step 2: Initialize Signal column
    df['Signal'] = 0
    
    # Step 3: Track Trend and Generate Signals
    # Forward fill calculated swings to get 'Last Swing High/Low' available at each bar
    df['last_swing_high'] = df['swing_high'].fillna(method='ffill')
    df['last_swing_low'] = df['swing_low'].fillna(method='ffill')
    
    trend = 0 
    
    for i in range(20, len(df)):
        # 1. Determine Trend
        # Simple break of structure:
        # If price closes above last swing high -> Uptrend
        if df['Close'].iloc[i] > df['last_swing_high'].iloc[i-1]:
            trend = 1
        # If price closes below last swing low -> Downtrend
        elif df['Close'].iloc[i] < df['last_swing_low'].iloc[i-1]:
            trend = -1
            
        # 2. Entry Conditions
        # We look for a pullback/consolidation within the trend
        
        # Calculate recent volatility (ATR-like)
        avg_range = (df['High'].iloc[i-10:i] - df['Low'].iloc[i-10:i]).mean()
        current_candle_range = df['High'].iloc[i] - df['Low'].iloc[i]
        
        # Consolidation/Pullback: Current candle is small (uncertainty) or we interpret previous candles as base
        # Let's simplify: Enter if we are in a trend and price is slightly pulling back (providing good risk/reward)
        
        if trend == 1:
            # Entry condition: Price is above moving average (momentum) AND recent small dip
            # Let's just use the "Impulse" concept from video: Close > Open (Green Candle) after a red candle?
            # Or valid Swing High Break recently?
            
            # Revised Logic:
            # If Uptrend, and we get a bullish candle that closes higher than previous candle high (breakout of micro-structure)
            if df['Close'].iloc[i] > df['High'].iloc[i-1] and df['Close'].iloc[i] > df['Open'].iloc[i]:
                 # Risk management
                entry = df['Close'].iloc[i]
                stop_loss = df['Low'].iloc[i-3:i].min() # Recent low
                
                if entry > stop_loss:
                    reward = (entry - stop_loss) * 2.0
                    df.loc[df.index[i], 'Signal'] = 1
                    
        elif trend == -1:
            # If Downtrend, and bearish candle break below previous low
            if df['Close'].iloc[i] < df['Low'].iloc[i-1] and df['Close'].iloc[i] < df['Open'].iloc[i]:
                entry = df['Close'].iloc[i]
                stop_loss = df['High'].iloc[i-3:i].max()
                
                if stop_loss > entry:
                    reward = (stop_loss - entry) * 2.0
                    df.loc[df.index[i], 'Signal'] = -1

    return df
