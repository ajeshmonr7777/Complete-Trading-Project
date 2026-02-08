def strategy(df):
    """
    Golden Cross strategy: Buy when 50-period SMA crosses above 200-period SMA,
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
    # Golden Cross: SMA_50 crosses above SMA_200 -> BUY (1)
    # Death Cross: SMA_50 crosses below SMA_200 -> SELL (-1)
    df['Signal'] = np.where(
        (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1)), 1,
        np.where(
            (df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1)), -1, 0
        )
    )
    
    return df
