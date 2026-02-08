def strategy(df):
    """
    EMA 200 + MACD Crossover Strategy:
    BUY: Close > EMA200 AND MACD crosses above Signal
    SELL: MACD crosses below Signal
    """
    import pandas as pd
    import numpy as np
    
    # Step 1: Calculate indicators from OHLC data
    # 200-period EMA
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Step 2: Initialize Signal column
    df['Signal'] = 0
    
    # Step 3: Generate trading signals
    # MACD crossover conditions (avoid bitwise NOT operator error)
    macd_cross_above = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    macd_cross_below = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    
    # Buy: Close > EMA200 AND MACD bullish crossover
    buy_condition = (df['Close'] > df['EMA_200']) & macd_cross_above
    df.loc[buy_condition, 'Signal'] = 1
    
    # Sell: MACD bearish crossover
    sell_condition = macd_cross_below
    df.loc[sell_condition, 'Signal'] = -1
    
    return df
