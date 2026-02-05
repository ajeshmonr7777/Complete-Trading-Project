import pandas as pd

def strategy(df):
    """
    TJR-style Strategy: Liquidity Sweeps + Fair Value Gaps (FVG)
    
    Concept:
    1. Identify Swing Highs/Lows (Liquidity Points).
    2. Wait for a 'Sweep' (Price pierces a swing point but closes back inside or reverses).
    3. Look for 'Displacement' (Strong move in opposite direction).
    4. Enter on FVG (Fair Value Gap) creation or retest.
    
    Simplified Logic for Backtester:
    - Bullish: 
        1. Price sweeps a recent 5-period Low (Low[i] < Low[i-n]), but Close[i] > Low[i-n] (Rejection).
        2. Next candle is Green.
        3. Enter.
    - Bearish:
        1. Price sweeps a recent 5-period High (High[i] > High[i-n]), but Close[i] < High[i-n] (Rejection).
        2. Next candle is Red.
        3. Enter.
    """
    
    # Needs pandas_ta for some helpers if desired, but we can stick to native pandas for speed
    df = df.copy()
    
    # 1. Identify Swing Points (Fractals)
    # A simple 5-bar fractal: High is higher than 2 bars left and 2 bars right
    # For backtesting causality, we can only know it's a fractal AFTER 2 bars. 
    # So at bar [i], we check if [i-2] was a high.
    
    window = 5
    
    # We will simply track rolling min/max to simulate "Recent Liquidity"
    df['Rolling_Min'] = df['Low'].rolling(window=10).min().shift(1) # Previous 10 candles low
    df['Rolling_Max'] = df['High'].rolling(window=10).max().shift(1) # Previous 10 candles high
    
    # Initialize Signal
    df['Signal'] = 0
    
    for i in range(window, len(df)):
        # --- Bullish Setup ---
        # 1. Liquidity Sweep: Current Low took out the recent rolling low
        sweep_low = df['Low'].iloc[i] < df['Rolling_Min'].iloc[i]
        
        # 2. Rejection: Price closed ABOVE the rolled low (Fakeout / Turtle Soup)
        rejection_low = df['Close'].iloc[i] > df['Rolling_Min'].iloc[i]
        
        # 3. Validation: Candle is Green (Close > Open) indicating buying pressure
        green_candle = df['Close'].iloc[i] > df['Open'].iloc[i]
        
        if sweep_low and rejection_low and green_candle:
            df.loc[df.index[i], 'Signal'] = 1
            
        # --- Bearish Setup ---
        # 1. Liquidity Sweep: Current High took out recent rolling high
        sweep_high = df['High'].iloc[i] > df['Rolling_Max'].iloc[i]
        
        # 2. Rejection: Price closed BELOW the rolled high
        rejection_high = df['Close'].iloc[i] < df['Rolling_Max'].iloc[i]
        
        # 3. Validation: Candle is Red
        red_candle = df['Close'].iloc[i] < df['Open'].iloc[i]
        
        # Note: Strategy is Long-Only currently in backtester.py, but Signal -1 closes positions.
        # If we enable shorting later, this will trigger shorts.
        if sweep_high and rejection_high and red_candle:
            df.loc[df.index[i], 'Signal'] = -1
            
    return df
