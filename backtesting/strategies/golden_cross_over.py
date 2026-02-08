def strategy(df):
    """
    Golden Cross strategy: Buy when 50-period SMA crosses above 200-period SMA.
    Sell when 50-period SMA crosses below 200-period SMA.
    """
    import pandas as pd
    import numpy as np
    
    # Step 1: Calculate indicators from OHLC data
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Step 2: Initialize Signal column
    df['Signal'] = 0
    
    # Step 3: Generate trading signals
    # 1 = BUY, -1 = SELL, 0 = HOLD
    df['prev_SMA_50'] = df['SMA_50'].shift(1)
    df['prev_SMA_200'] = df['SMA_200'].shift(1)
    
    # Buy signal: 50 SMA crosses above 200 SMA
    df.loc[(df['prev_SMA_50'] <= df['prev_SMA_200']) & 
           (df['SMA_50'] > df['SMA_200']), 'Signal'] = 1
    
    # Sell signal: 50 SMA crosses below 200 SMA
    df.loc[(df['prev_SMA_50'] >= df['prev_SMA_200']) & 
           (df['SMA_50'] < df['SMA_200']), 'Signal'] = -1
    
    return df
