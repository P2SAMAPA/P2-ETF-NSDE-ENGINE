from huggingface_hub import hf_hub_download
import pandas as pd
import os

from config import (
    HF_DATASET_INPUT,
    OPTION_A_ETFS,
    OPTION_B_ETFS
)

def load_dataset(option: str = "both"):
    """Load pre-processed data from HF dataset (master.parquet)."""
    print(f"Downloading dataset: {HF_DATASET_INPUT}")
    
    master_path = hf_hub_download(
        repo_id=HF_DATASET_INPUT,
        filename="data/master.parquet",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    
    df = pd.read_parquet(master_path)
    
    # Convert index to datetime if it's numeric (Unix timestamp)
    if pd.api.types.is_numeric_dtype(df.index):
        # Try to infer if timestamps are in seconds or milliseconds
        # Typical Unix seconds are ~1.6e9, milliseconds ~1.6e12
        sample = df.index[0]
        if sample > 1e12:  # milliseconds
            df.index = pd.to_datetime(df.index, unit='ms')
        else:  # seconds
            df.index = pd.to_datetime(df.index, unit='s')
        print(f"Converted numeric index to datetime: {df.index[0]} to {df.index[-1]}")
    
    # If index is still not datetime, try to find a date column
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'datetime', 'timestamp', 'ds', 'time']:
                date_col = col
                break
        if date_col is None:
            # Check for any datetime column
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = col
                    break
        
        if date_col:
            print(f"Using '{date_col}' as date column")
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            raise ValueError("Could not convert index to datetime and no date column found.")
    
    # Determine which tickers to load
    if option in ["a", "both"]:
        tickers_to_load = ["AGG"] + OPTION_A_ETFS
    elif option == "b":
        tickers_to_load = ["SPY"] + OPTION_B_ETFS
    else:
        tickers_to_load = ["AGG", "SPY"] + OPTION_A_ETFS + OPTION_B_ETFS
    
    data = {}
    for ticker in tickers_to_load:
        # Try common column patterns for close prices
        possible_cols = [f"{ticker}_Close", f"{ticker}_close", f"Close_{ticker}", f"{ticker}_Close_adj"]
        close_col = None
        for col in possible_cols:
            if col in df.columns:
                close_col = col
                break
        
        if close_col is None:
            print(f"⚠️ No close price column found for {ticker}")
            continue
        
        series = df[close_col].dropna()
        if len(series) == 0:
            print(f"⚠️ No valid data for {ticker}")
            continue
        
        ticker_df = pd.DataFrame({'close': series})
        data[ticker] = ticker_df
        print(f"✅ Loaded {ticker}: {len(ticker_df)} rows")
    
    print(f"Total loaded: {len(data)} tickers")
    return data
