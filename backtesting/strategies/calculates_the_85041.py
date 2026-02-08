def strategy(df):
    """
    EMA 200 + MACD Crossover Strategy.
    Buy: Close > EMA200 AND MACD crosses above Signal.
    Sell: MACD crosses below Signal.
    """
    import pandas as pd
    import numpy as np
    
    # Step 1: Calculate indicators from OHLC data
    # 200-period EMA
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Step 2: Initialize Signal column
    df['Signal'] = 0
    
    # Step 3: Generate trading signals
    # MACD crossover conditions
    macd_cross_up = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    macd_cross_down = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    
    # Buy: Close > EMA200 AND MACD crosses above Signal
    df.loc[(df['Close'] > df['EMA200']) & macd_cross_up, 'Signal'] = 1
    
    # Sell: MACD crosses below Signal
    df.loc[macd_cross_down, 'Signal'] = -1
    
    return df
