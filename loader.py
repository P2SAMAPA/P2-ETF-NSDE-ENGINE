from huggingface_hub import snapshot_download
import pandas as pd
import os

from config import (
    HF_DATASET_INPUT,
    OPTION_A_ETFS,
    OPTION_B_ETFS
)

def load_dataset(option: str = "both"):
    """Load pre-processed data from HF dataset (combined parquet files)."""
    print(f"Downloading dataset: {HF_DATASET_INPUT}")
    
    local_dir = snapshot_download(
        repo_id=HF_DATASET_INPUT, 
        repo_type="dataset",
        allow_patterns=["*.parquet"]
    )
    
    # Load the master parquet file (contains all ETF price columns)
    master_path = os.path.join(local_dir, "master.parquet")
    if not os.path.exists(master_path):
        # Fallback to etf_ohlcv.parquet if master is missing
        master_path = os.path.join(local_dir, "etf_ohlcv.parquet")
    
    if not os.path.exists(master_path):
        raise FileNotFoundError("No master.parquet or etf_ohlcv.parquet found in dataset.")
    
    df = pd.read_parquet(master_path)
    
    # Ensure index is datetime (may already be, but convert if needed)
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse a date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            raise ValueError("Dataframe index is not datetime and no 'date' column found.")
    
    # Determine which tickers to load
    if option in ["a", "both"]:
        tickers_to_load = ["AGG"] + OPTION_A_ETFS
    elif option == "b":
        tickers_to_load = ["SPY"] + OPTION_B_ETFS
    else:
        tickers_to_load = ["AGG", "SPY"] + OPTION_A_ETFS + OPTION_B_ETFS
    
    data = {}
    for ticker in tickers_to_load:
        # Try different column naming patterns: e.g., "AGG_Close", "AGG_Close", or "Close_AGG"
        possible_cols = [f"{ticker}_Close", f"{ticker}_close", f"Close_{ticker}"]
        close_col = None
        for col in possible_cols:
            if col in df.columns:
                close_col = col
                break
        
        if close_col is None:
            print(f"⚠️ No close price column found for {ticker}")
            continue
        
        # Extract the series, drop NaNs, and create a DataFrame with a 'close' column
        series = df[close_col].dropna()
        if len(series) == 0:
            print(f"⚠️ No valid data for {ticker}")
            continue
        
        ticker_df = pd.DataFrame({'close': series})
        data[ticker] = ticker_df
        print(f"✅ Loaded {ticker}: {len(ticker_df)} rows")
    
    print(f"Total loaded: {len(data)} tickers")
    return data
