"""
Fixed Technical Indicators Calculator
Proper handling of ranges, volumes, and edge cases
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional


def calculate_technical_indicators(df: pd.DataFrame, ema_period: int = 20, rsi_period: int = 14) -> Dict:
    """
    Calculate technical indicators with proper error handling and realistic values
    """
    if len(df) < max(ema_period, rsi_period, 50):  # Reduced minimum to 50
        return {}
    
    try:
        # Make a copy to avoid modifying original
        df_analysis = df.copy()
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_analysis.columns:
                df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        
        # === CORE INDICATORS ===
        # Trend Indicators
        df_analysis['ema_20'] = ta.trend.EMAIndicator(close=df_analysis['close'], window=20).ema_indicator()
        df_analysis['ema_50'] = ta.trend.EMAIndicator(close=df_analysis['close'], window=50).ema_indicator()
        df_analysis['ema_200'] = ta.trend.EMAIndicator(close=df_analysis['close'], window=200).ema_indicator()
        
        # MACD
        macd_indicator = ta.trend.MACD(close=df_analysis['close'])
        df_analysis['macd'] = macd_indicator.macd()
        df_analysis['macd_signal'] = macd_indicator.macd_signal()
        df_analysis['macd_hist'] = macd_indicator.macd_diff()
        
        # Momentum Indicators
        df_analysis['rsi'] = ta.momentum.RSIIndicator(close=df_analysis['close'], window=14).rsi()
        
        # Volatility Indicators
        df_analysis['bb_upper'] = ta.volatility.BollingerBands(close=df_analysis['close']).bollinger_hband()
        df_analysis['bb_lower'] = ta.volatility.BollingerBands(close=df_analysis['close']).bollinger_lband()
        df_analysis['bb_middle'] = ta.volatility.BollingerBands(close=df_analysis['close']).bollinger_mavg()
        
        # Volume Indicators
        if 'volume' in df_analysis.columns:
            df_analysis['volume_sma'] = df_analysis['volume'].rolling(window=20, min_periods=1).mean()
            df_analysis['volume_ratio'] = df_analysis['volume'] / df_analysis['volume_sma'].replace(0, 1)  # Avoid division by zero
        
        # === ENHANCED INDICATORS WITH PROPER HANDLING ===
        # Volatility - ATR (with validation)
        try:
            df_analysis['atr_14'] = ta.volatility.AverageTrueRange(
                high=df_analysis['high'], 
                low=df_analysis['low'], 
                close=df_analysis['close'], 
                window=14
            ).average_true_range()
        except:
            df_analysis['atr_14'] = df_analysis['close'].rolling(window=14).std()
        
        # Recent Price Ranges (with proper window sizing)
        available_periods = min(240, len(df_analysis))  # Max 4 hours for 1m data
        if available_periods > 50:
            df_analysis['recent_high_4h'] = df_analysis['high'].rolling(window=available_periods, min_periods=1).max()
            df_analysis['recent_low_4h'] = df_analysis['low'].rolling(window=available_periods, min_periods=1).min()
            df_analysis['recent_range'] = df_analysis['recent_high_4h'] - df_analysis['recent_low_4h']
        else:
            df_analysis['recent_high_4h'] = df_analysis['high'].max()
            df_analysis['recent_low_4h'] = df_analysis['low'].min()
            df_analysis['recent_range'] = df_analysis['high'].max() - df_analysis['low'].min()
        
        # Enhanced Volume Analysis
        if 'volume' in df_analysis.columns:
            df_analysis['volume_trend'] = df_analysis['volume'] > df_analysis['volume_sma']
            df_analysis['volume_spike'] = df_analysis['volume'] > (df_analysis['volume_sma'] * 1.5)
            df_analysis['volume_very_high'] = df_analysis['volume'] > (df_analysis['volume_sma'] * 2.0)
            
            # Volume classification with safe handling
            conditions = [
                df_analysis['volume_ratio'] > 2.0,
                df_analysis['volume_ratio'] > 1.5,
                df_analysis['volume_ratio'] > 1.0
            ]
            choices = ['VERY_HIGH', 'HIGH', 'AVERAGE']
            df_analysis['volume_class'] = np.select(conditions, choices, default='LOW')
        
        # Key Levels (with safe rolling)
        df_analysis['support_level'] = df_analysis['low'].rolling(window=min(50, len(df_analysis)), min_periods=1).min()
        df_analysis['resistance_level'] = df_analysis['high'].rolling(window=min(50, len(df_analysis)), min_periods=1).max()
        
        # Get latest values with safety checks
        current_price = float(df_analysis['close'].iloc[-1]) if len(df_analysis) > 0 else 0
        
        # Calculate position in range safely
        if 'recent_range' in df_analysis and df_analysis['recent_range'].iloc[-1] > 0:
            position_in_range = (current_price - df_analysis['recent_low_4h'].iloc[-1]) / df_analysis['recent_range'].iloc[-1]
        else:
            position_in_range = 0.5
        
        # === INDICATORS DICTIONARY WITH SAFE DEFAULTS ===
        indicators = {
            # === CORE FIELDS ===
            'current_price': current_price,
            'ema_20': float(df_analysis['ema_20'].iloc[-1]) if not pd.isna(df_analysis['ema_20'].iloc[-1]) else current_price,
            'ema_50': float(df_analysis['ema_50'].iloc[-1]) if not pd.isna(df_analysis['ema_50'].iloc[-1]) else current_price,
            'ema_200': float(df_analysis['ema_200'].iloc[-1]) if not pd.isna(df_analysis['ema_200'].iloc[-1]) else current_price,
            'macd': float(df_analysis['macd'].iloc[-1]) if not pd.isna(df_analysis['macd'].iloc[-1]) else 0,
            'macd_signal': float(df_analysis['macd_signal'].iloc[-1]) if not pd.isna(df_analysis['macd_signal'].iloc[-1]) else 0,
            'macd_hist': float(df_analysis['macd_hist'].iloc[-1]) if not pd.isna(df_analysis['macd_hist'].iloc[-1]) else 0,
            'rsi': float(df_analysis['rsi'].iloc[-1]) if not pd.isna(df_analysis['rsi'].iloc[-1]) else 50,
            'bb_upper': float(df_analysis['bb_upper'].iloc[-1]) if not pd.isna(df_analysis['bb_upper'].iloc[-1]) else current_price,
            'bb_lower': float(df_analysis['bb_lower'].iloc[-1]) if not pd.isna(df_analysis['bb_lower'].iloc[-1]) else current_price,
            'bb_middle': float(df_analysis['bb_middle'].iloc[-1]) if not pd.isna(df_analysis['bb_middle'].iloc[-1]) else current_price,
            'bb_position': (current_price - float(df_analysis['bb_lower'].iloc[-1])) / 
                          (float(df_analysis['bb_upper'].iloc[-1]) - float(df_analysis['bb_lower'].iloc[-1])) 
                          if not pd.isna(df_analysis['bb_upper'].iloc[-1]) and not pd.isna(df_analysis['bb_lower'].iloc[-1]) and 
                             (float(df_analysis['bb_upper'].iloc[-1]) - float(df_analysis['bb_lower'].iloc[-1])) > 0 
                          else 0.5,
            
            # Series data
            'price_series': [float(x) for x in df_analysis['close'].tail(50)],
            'ema_series': [float(x) for x in df_analysis['ema_20'].tail(50)],
            'macd_series': [float(x) for x in df_analysis['macd'].tail(50)],
            'rsi_series': [float(x) for x in df_analysis['rsi'].tail(50)],
            'volume_series': [float(x) for x in df_analysis['volume'].tail(50)] if 'volume' in df_analysis.columns else [],
            
            # Additional metrics
            'oi_latest': 100000,
            'oi_average': 95000,
            'funding_rate': 0.0001,
            
            # Market structure
            'above_ema20': current_price > float(df_analysis['ema_20'].iloc[-1]) if not pd.isna(df_analysis['ema_20'].iloc[-1]) else False,
            'above_ema50': current_price > float(df_analysis['ema_50'].iloc[-1]) if not pd.isna(df_analysis['ema_50'].iloc[-1]) else False,
            'above_ema200': current_price > float(df_analysis['ema_200'].iloc[-1]) if not pd.isna(df_analysis['ema_200'].iloc[-1]) else False,
            'rsi_level': 'OVERSOLD' if float(df_analysis['rsi'].iloc[-1]) < 30 else 'OVERBOUGHT' if float(df_analysis['rsi'].iloc[-1]) > 70 else 'NEUTRAL',
            'macd_trend': 'BULLISH' if float(df_analysis['macd'].iloc[-1]) > float(df_analysis['macd_signal'].iloc[-1]) else 'BEARISH',
            
            # === ENHANCED FIELDS WITH SAFE DEFAULTS ===
            # Volume Analysis
            'volume_trend': 'INCREASING' if 'volume_trend' in df_analysis.columns and bool(df_analysis['volume_trend'].iloc[-1]) else 'DECREASING',
            'volume_class': str(df_analysis['volume_class'].iloc[-1]) if 'volume_class' in df_analysis.columns else 'UNKNOWN',
            'volume_spike': bool(df_analysis['volume_spike'].iloc[-1]) if 'volume_spike' in df_analysis.columns else False,
            'volume_ratio_value': float(df_analysis['volume_ratio'].iloc[-1]) if 'volume_ratio' in df_analysis.columns else 1.0,
            'current_volume': float(df_analysis['volume'].iloc[-1]) if 'volume' in df_analysis.columns else 0,
            'volume_sma_20': float(df_analysis['volume_sma'].iloc[-1]) if 'volume_sma' in df_analysis.columns else 0,
            
            # Volatility (FIXED - no more NaN)
            'atr_14': float(df_analysis['atr_14'].iloc[-1]) if 'atr_14' in df_analysis.columns and not pd.isna(df_analysis['atr_14'].iloc[-1]) else max(0.01 * current_price, 1.0),
            'atr_percent': (float(df_analysis['atr_14'].iloc[-1]) / current_price * 100) if 'atr_14' in df_analysis.columns and not pd.isna(df_analysis['atr_14'].iloc[-1]) and current_price > 0 else 1.0,
            'daily_range': float(df_analysis['recent_range'].iloc[-1]) if 'recent_range' in df_analysis.columns and not pd.isna(df_analysis['recent_range'].iloc[-1]) else current_price * 0.02,
            'daily_range_percent': (float(df_analysis['recent_range'].iloc[-1]) / current_price * 100) if 'recent_range' in df_analysis.columns and not pd.isna(df_analysis['recent_range'].iloc[-1]) and current_price > 0 else 2.0,
            'recent_high_24h': float(df_analysis['recent_high_4h'].iloc[-1]) if 'recent_high_4h' in df_analysis.columns and not pd.isna(df_analysis['recent_high_4h'].iloc[-1]) else current_price * 1.01,
            'recent_low_24h': float(df_analysis['recent_low_4h'].iloc[-1]) if 'recent_low_4h' in df_analysis.columns and not pd.isna(df_analysis['recent_low_4h'].iloc[-1]) else current_price * 0.99,
            
            # Key Levels
            'support_level': float(df_analysis['support_level'].iloc[-1]) if 'support_level' in df_analysis.columns and not pd.isna(df_analysis['support_level'].iloc[-1]) else current_price * 0.99,
            'resistance_level': float(df_analysis['resistance_level'].iloc[-1]) if 'resistance_level' in df_analysis.columns and not pd.isna(df_analysis['resistance_level'].iloc[-1]) else current_price * 1.01,
            'distance_to_resistance': ((float(df_analysis['resistance_level'].iloc[-1]) - current_price) / current_price * 100) if 'resistance_level' in df_analysis.columns and not pd.isna(df_analysis['resistance_level'].iloc[-1]) and current_price > 0 else 1.0,
            'distance_to_support': ((current_price - float(df_analysis['support_level'].iloc[-1])) / current_price * 100) if 'support_level' in df_analysis.columns and not pd.isna(df_analysis['support_level'].iloc[-1]) and current_price > 0 else 1.0,
            'position_in_daily_range': position_in_range,
            
            # Market Context
            'volatility_assessment': 'HIGH' if (float(df_analysis['atr_14'].iloc[-1]) / current_price) > 0.02 else 'MEDIUM' if (float(df_analysis['atr_14'].iloc[-1]) / current_price) > 0.01 else 'LOW',
            'volume_confidence': 'HIGH' if 'volume_very_high' in df_analysis.columns and bool(df_analysis['volume_very_high'].iloc[-1]) else 'MEDIUM' if 'volume_spike' in df_analysis.columns and bool(df_analysis['volume_spike'].iloc[-1]) else 'LOW'
        }
        
        return indicators
        
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return {}


# ... (keep the same helper functions as before)
def get_available_indicators() -> Dict[str, List[str]]:
    return {
        'core_technical': [
            'current_price', 'ema_20', 'ema_50', 'ema_200', 'macd', 'macd_signal', 
            'macd_hist', 'rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'bb_position'
        ],
        'volume_analysis': [
            'volume_trend', 'volume_class', 'volume_spike', 'volume_ratio_value',
            'current_volume', 'volume_sma_20', 'volume_confidence'
        ],
        'volatility_context': [
            'atr_14', 'atr_percent', 'daily_range', 'daily_range_percent',
            'recent_high_24h', 'recent_low_24h', 'volatility_assessment'
        ],
        'key_levels': [
            'support_level', 'resistance_level', 'distance_to_resistance',
            'distance_to_support', 'position_in_daily_range'
        ]
    }


def create_custom_indicator_subset(indicators: Dict, categories: List[str]) -> Dict:
    available = get_available_indicators()
    custom_indicators = {}
    
    for category in categories:
        if category in available:
            for indicator in available[category]:
                if indicator in indicators:
                    custom_indicators[indicator] = indicators[indicator]
    
    return custom_indicators