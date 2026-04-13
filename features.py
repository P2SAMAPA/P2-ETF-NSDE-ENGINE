import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
    """Engineer features for NSDE model from ETF prices, optionally merge macro data."""
    df = df.copy()
    # Price-derived features
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_20'] = df['log_return'].rolling(20).std() * np.sqrt(252)
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['mom_60'] = df['close'] / df['close'].shift(60) - 1

    # Select core feature columns
    feature_cols = ['vol_20', 'mom_10', 'mom_60']
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        df['log_return_feat'] = df['log_return']
        available = ['log_return_feat']
    X = df[available].copy()

    # Merge macro variables if provided
    if macro_df is not None:
        # Reindex macro to same datetime index as ETF (forward fill)
        macro_aligned = macro_df.reindex(df.index, method='ffill')
        for col in macro_aligned.columns:
            X[f'macro_{col}'] = macro_aligned[col]
        # Add macro momentum (5-day change)
        for col in macro_aligned.columns:
            X[f'macro_{col}_chg5'] = macro_aligned[col].pct_change(5)

    # Fill NaNs – pandas 2.0+ compatible (use ffill then fillna)
    X = X.ffill().fillna(0)
    return X
