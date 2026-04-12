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
    
    # Convert 'Date' column from milliseconds to datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
    else:
        raise KeyError("Column 'Date' not found in master.parquet")
    
    df.sort_index(inplace=True)
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Determine which tickers to load (FIXED)
    if option == "a":
        tickers_to_load = ["AGG"] + OPTION_A_ETFS
    elif option == "b":
        tickers_to_load = ["SPY"] + OPTION_B_ETFS
    elif option == "both":
        tickers_to_load = ["AGG", "SPY"] + OPTION_A_ETFS + OPTION_B_ETFS
    else:
        raise ValueError(f"Invalid option: {option}. Use 'a', 'b', or 'both'.")
    
    data = {}
    for ticker in tickers_to_load:
        # Try multiple column patterns for close prices
        possible_cols = [
            f"{ticker}_Close",
            f"{ticker}_close",
            f"Close_{ticker}",
            f"{ticker}_Close_adj",
            f"{ticker}_PRICE"
        ]
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
