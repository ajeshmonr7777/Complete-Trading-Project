def strategy(df):
    """
    Swing Breakout Strategy derived from YouTube transcript.
    
    Logic:
    1. Price must be above EMA20 for 3+ days (long bias) or below for shorts
    2. Price retraces back to EMA20
    3. Draw least steep trend line covering multiple candles
    4. Wait for breakout candle (price breaks above trend line)
    5. Confirm with Relative Volume > 2.0 and Relative Strength (RS) strength
    6. Calculate Stop Loss = Trend Line Intersection - ATR
    7. Set Take Profit with 1:2 Risk:Reward minimum
    8. Entry at close of breakout candle, trade only 1-hour timeframe
    """
    import pandas as pd
    import numpy as np
    
    # Step 1: Calculate Indicators
    # EMA 20 - trend filter
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # ATR - volatility for stop loss calculation
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Relative Volume - volume confirmation
    df['Volume_MA50'] = df['Volume'].rolling(window=50).mean()
    df['RelativeVolume'] = df['Volume'] / df['Volume_MA50']
    
    # Relative Strength Line - compare stock strength vs market
    # Simplified: use RSI as proxy for relative strength
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Step 2: Identify Long Bias (price above EMA20 for 3+ days)
    df['Above_EMA20'] = df['Close'] > df['EMA20']
    df['Days_Above_EMA20'] = df['Above_EMA20'].astype(int).rolling(window=3).sum()
    df['Long_Bias'] = df['Days_Above_EMA20'] >= 3
    
    # Step 3: Identify Retracement to EMA20
    # Price pulls back within 2% of EMA20 after being above it
    df['Near_EMA20'] = abs(df['Close'] - df['EMA20']) / df['EMA20'] < 0.02
    df['Retracement'] = df['Long_Bias'] & df['Near_EMA20']
    
    # Step 4: Identify Breakout Candle
    # Price breaks above the 20-period high (trend line approximation)
    df['Highest_20'] = df['High'].rolling(window=20).max().shift(1)
    df['Breakout'] = df['High'] > df['Highest_20']
    
    # Step 5: Confirm Breakout with Volume and Strength
    df['Volume_Confirmed'] = df['RelativeVolume'] > 2.0
    df['Strength_Confirmed'] = df['RSI'] > 50  # RSI > 50 indicates strength
    
    # Step 6: Calculate Stop Loss and Take Profit
    # Stop Loss = Breakout High - ATR (approximation of trend line intersection)
    df['StopLoss'] = df['High'] - df['ATR']
    
    # Risk = Entry Price - Stop Loss
    df['Risk'] = df['Close'] - df['StopLoss']
    
    # Take Profit = Entry Price + (Risk * 2) for 1:2 R:R
    df['TakeProfit'] = df['Close'] + (df['Risk'] * 2)
    
    # Step 7: Generate Trading Signal
    df['Signal'] = 0
    
    # Buy Signal: Long bias + Retracement + Breakout + Volume + Strength confirmed
    buy_condition = (
        df['Long_Bias'] & 
        df['Retracement'] & 
        df['Breakout'] & 
        df['Volume_Confirmed'] & 
        df['Strength_Confirmed'] &
        (df['ATR'] > 0) &  # Ensure ATR is calculated
        (df['Risk'] > 0)   # Ensure valid risk calculation
    )
    
    df.loc[buy_condition, 'Signal'] = 1
    
    # Sell Signal: Short bias (price below EMA20 for 3+ days) with similar logic
    df['Below_EMA20'] = df['Close'] < df['EMA20']
    df['Days_Below_EMA20'] = df['Below_EMA20'].astype(int).rolling(window=3).sum()
    df['Short_Bias'] = df['Days_Below_EMA20'] >= 3
    
    df['Lowest_20'] = df['Low'].rolling(window=20).min().shift(1)
    df['Breakdown'] = df['Low'] < df['Lowest_20']
    
    sell_condition = (
        df['Short_Bias'] & 
        df['Breakdown'] & 
        df['Volume_Confirmed'] & 
        (df['RSI'] < 50) &  # RSI < 50 indicates weakness
        (df['ATR'] > 0) &
        (df['Risk'] > 0)
    )
    
    df.loc[sell_condition, 'Signal'] = -1
    
    # Step 8: Position Management
    # Shift signal to avoid look-ahead bias (enter on next candle)
    df['Position'] = df['Signal'].shift(1)
    
    # Calculate returns
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Position'] * df['Daily_Return']
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    
    return df
