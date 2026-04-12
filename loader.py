from huggingface_hub import snapshot_download
import pandas as pd
import os

# Import the ETF lists from config
from config import (
    HF_DATASET_INPUT,
    OPTION_A_ETFS,
    OPTION_B_ETFS
)

def load_dataset(option: str = "both"):
    """Load pre-processed data from HF dataset."""
    print(f"Downloading dataset: {HF_DATASET_INPUT}")
    
    local_dir = snapshot_download(
        repo_id=HF_DATASET_INPUT, 
        repo_type="dataset",
        allow_patterns=["*.parquet"]   # Only download parquet files to speed up
    )
    
    data = {}
    
    if option in ["a", "both"]:
        etfs_to_load = ["AGG"] + OPTION_A_ETFS
    elif option == "b":
        etfs_to_load = ["SPY"] + OPTION_B_ETFS
    else:
        etfs_to_load = ["AGG", "SPY"] + OPTION_A_ETFS + OPTION_B_ETFS

    for etf in etfs_to_load:
        file_path = os.path.join(local_dir, f"{etf}.parquet")
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                data[etf] = df
                print(f"✅ Loaded {etf}: {len(df)} rows")
            except Exception as e:
                print(f"⚠️ Failed to load {etf}: {e}")
        else:
            print(f"⚠️ File not found: {etf}.parquet")

    print(f"Total loaded: {len(data)} tickers")
    return data
