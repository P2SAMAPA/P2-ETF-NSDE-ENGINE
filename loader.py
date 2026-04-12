from huggingface_hub import snapshot_download
import pandas as pd
import os
import glob

from config import (
    HF_DATASET_INPUT,
    OPTION_A_ETFS,
    OPTION_B_ETFS
)

def load_dataset(option: str = "both"):
    """Load pre-processed data from HF dataset (master.parquet)."""
    print(f"Downloading dataset: {HF_DATASET_INPUT}")
    
    # Download the entire repo (no filtering to see all files)
    local_dir = snapshot_download(
        repo_id=HF_DATASET_INPUT,
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),  # Optional but helps with rate limits
    )
    
    # List all downloaded files for debugging
    print(f"Downloaded to: {local_dir}")
    all_files = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
            print(f"Found: {full_path}")
    
    # Find any .parquet file (we'll use the first one that contains price data)
    parquet_files = [f for f in all_files if f.endswith('.parquet')]
    if not parquet_files:
        raise FileNotFoundError("No .parquet files found in the downloaded dataset.")
    
    # Use the largest parquet file (likely the main data file)
    parquet_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    master_path = parquet_files[0]
    print(f"Using parquet file: {master_path}")
    
    df = pd.read_parquet(master_path)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
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
        # Try common column patterns
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
