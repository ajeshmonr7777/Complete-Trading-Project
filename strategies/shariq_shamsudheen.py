import pandas as pd

def strategy(df):
    """
    Implements a swing trading strategy based on:
    1. 50-Day EMA for trend confirmation.
    2. CPR (Central Pivot Range) for level identification.
    3. Consolidation at support with low volume.
    4. Breakout confirmation.
    """
    
    df = df.copy()
    
    # --- 1. Technical Indicators ---
    
    # 50-Day Exponential Moving Average [1, 4]
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Central Pivot Range (CPR) Calculation [3, 4]
    # Standard CPR uses previous day's High, Low, Close
    df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['BC'] = (df['High'].shift(1) + df['Low'].shift(1)) / 2
    df['TC'] = (df['Pivot'] - df['BC']) + df['Pivot']
    
    # --- 2. Consolidation & Volume Analysis ---
    
    # Define consolidation: Price stays within a 2.5% range over the last 5 days [1, 2, 5]
    df['Rolling_Max'] = df['High'].rolling(window=5).max()
    df['Rolling_Min'] = df['Low'].rolling(window=5).min()
    
    # Safe division to avoid ZeroDivisionError if Low is 0 (bad data)
    # Using a small epsilon or replacing 0 with NaN/small number
    denom = df['Rolling_Min'].replace(0, 0.01) # Avoid div by zero
    df['Consolidating'] = ((df['Rolling_Max'] - df['Rolling_Min']) / denom) < 0.025
    
    # Low Volume Check: Current volume is lower than the 20-day average [1]
    df['Avg_Volume_20'] = df['Volume'].rolling(window=20).mean()
    # Use shift(1) for volume check to match "was consolidating and low volume" prior to breakout 
    # OR check if *during* consolidation volume was low. 
    # The original code: df['Low_Volume'] = df['Volume'] < df['Avg_Volume_20']
    # Then buy_condition checks (df['Low_Volume'].shift(1)) -> so clearly checking previous candle's volume.
    df['Low_Volume'] = df['Volume'] < df['Avg_Volume_20']
    
    # --- 3. Signal Logic ---
    
    # The strategy enters when:
    # - Price is above 50 EMA (Trend Confirmation) [1, 4]
    # - Price is above the CPR range [4]
    # - There was a recent consolidation period with low volume [1, 6]
    # - A breakout occurs (Close > previous consolidation high) [2, 7]

    df['Signal'] = 0
    
    # Note: 'TC' is typically Top Central Pivot, but TC can be below BC depending on calculation. 
    # Usually CPR is Pivot, BC, TC. Range is between BC and TC.
    # We check if Close > TC (assuming TC is the upper bound or at least part of the range).
    # Ideally should check Close > max(TC, BC). But using user logic strictly for now.
    
    buy_condition = (
        (df['Close'] > df['EMA_50']) &              # Above 50 EMA
        (df['Close'] > df['TC']) &                 # Above CPR Top Central
        (df['Consolidating'].shift(1)) &           # Was consolidating yesterday
        (df['Low_Volume'].shift(1)) &              # Had low volume yesterday (end of consolidation)
        (df['Close'] > df['Rolling_Max'].shift(1)) # Breakout above previous consolidated high
    )
    
    df.loc[buy_condition, 'Signal'] = 1 # 1 represents a BUY signal
    
    # Optional: Exit Logic (not specified, but backtester triggers Stop Loss/Take Profit)
    # The backtester handles stops if signals are just '1'. 
    # If we wanted an indicator-based exit, we'd set Signal = -1.
    
    return df
