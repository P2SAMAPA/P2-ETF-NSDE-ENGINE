import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for NSDE model - same logic as original for compatibility."""
    df = df.copy()
    
    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility (20-day)
    df['vol_20'] = df['log_return'].rolling(20).std() * np.sqrt(252)
    
    # Momentum
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['mom_60'] = df['close'] / df['close'].shift(60) - 1
    
    # Macro features (if present)
    if 'vix' in df.columns:
        df['vix_change'] = df['vix'].pct_change()
    if 't10y2y' in df.columns:
        df['yield_curve'] = df['t10y2y']
    
    # Fill NaNs (pandas 2.0+ compatible)
    df = df.ffill().fillna(0)
    
    return df
