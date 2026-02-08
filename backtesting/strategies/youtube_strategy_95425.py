def strategy(df):
    """
    Strategy derived from YouTube video.
    Logic: Swing breakout strategy using 1H timeframe.
    1. Price above EMA20 for 3+ days (long bias) or below (short bias)
    2. Price retraces to EMA20
    3. Breakout above recent swing low (long) or below swing high (short) 
    4. Relative volume > 2x average (approximated with volume ratio)
    5. Enter at close, SL = breakout_level +/- ATR, TP = 2x risk
    """
    import pandas as pd
    import numpy as np
    
    # Step 1: Indicators
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['ATR'] = df[['High', 'Low', 'Close']].apply(lambda x: pd.Series.max(x) - pd.Series.min(x), axis=1).rolling(14).mean()
    df['Volume_MA'] = df['Volume'].rolling(50).mean()
    df['Rel_Volume'] = df['Volume'] / df['Volume_MA']
    
    # Approximate swing lows/highs with rolling min/max (lookback 10 periods)
    lookback = 10
    df['Swing_Low'] = df['Low'].rolling(2*lookback+1, center=True).min()
    df['Swing_High'] = df['High'].rolling(2*lookback+1, center=True).max()
    
    # Step 2: Signal (1=Buy, -1=Sell, 0=Hold)
    df['Signal'] = 0
    df['Bias'] = 0
    
    # Long bias: Close > EMA20 for 3+ consecutive days
    df['Above_EMA'] = df['Close'] > df['EMA20']
    df['Long_Bias'] = df['Above_EMA'].rolling(3).sum() >= 3
    df['Below_EMA'] = df['Close'] < df['EMA20']
    df['Short_Bias'] = df['Below_EMA'].rolling(3).sum() >= 3
    
    # Retracement condition: Close near EMA20 (within 0.5 ATR)
    atr_prox = 0.5
    df['Near_EMA'] = np.abs(df['Close'] - df['EMA20']) < (atr_prox * df['ATR'])
    
    # Breakout conditions (no future leakage - use previous swing levels)
    df['Breakout_Up'] = df['Close'] > df['Swing_Low'].shift(1)
    df['Breakout_Down'] = df['Close'] < df['Swing_High'].shift(1)
    
    # High relative volume (>2x)
    df['High_Vol'] = df['Rel_Volume'] > 2.0
    
    # Generate signals
    long_condition = (
        df['Long_Bias'] & 
        df['Near_EMA'].shift(1) &  # Was retracing previous bar
        df['Breakout_Up'] &
        df['High_Vol']
    )
    
    short_condition = (
        df['Short_Bias'] & 
        df['Near_EMA'].shift(1) &  # Was retracing previous bar
        df['Breakout_Down'] &
        df['High_Vol']
    )
    
    df.loc[long_condition, 'Signal'] = 1
    df.loc[short_condition, 'Signal'] = -1
    
    return df
