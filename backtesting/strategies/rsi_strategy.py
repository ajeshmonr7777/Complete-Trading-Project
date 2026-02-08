def strategy(df):
    """
    RSI (Relative Strength Index) Strategy
    """
    import pandas as pd
    import numpy as np

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Generate Signals
    df['Signal'] = 0
    
    # Buy when RSI < 30
    buy_signal = (df['RSI'] < 30)
    
    # Sell when RSI > 70
    sell_signal = (df['RSI'] > 70)
    
    df['Signal'] = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    
    return df
