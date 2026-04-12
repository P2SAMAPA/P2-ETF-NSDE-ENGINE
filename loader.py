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
    
    # Directly download the master.parquet file (or fallback to etf_ohlcv.parquet)
    try:
        master_path = hf_hub_download(
            repo_id=HF_DATASET_INPUT,
            filename="master.parquet",
            repo_type="dataset"
        )
    except Exception as e:
        print(f"master.parquet not found, trying etf_ohlcv.parquet: {e}")
        try:
            master_path = hf_hub_download(
                repo_id=HF_DATASET_INPUT,
                filename="etf_ohlcv.parquet",
                repo_type="dataset"
            )
        except Exception as e2:
            raise FileNotFoundError("Neither master.parquet nor etf_ohlcv.parquet found in dataset.") from e2
    
    df = pd.read_parquet(master_path)
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            # Try to infer from first column
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
        # Try different column naming patterns
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
