from huggingface_hub import snapshot_download
import pandas as pd
import os
from config import HF_DATASET_INPUT

def load_dataset(option: str = "both"):
    """Load pre-processed data from HF dataset."""
    print(f"Downloading dataset: {HF_DATASET_INPUT}")
    
    local_dir = snapshot_download(repo_id=HF_DATASET_INPUT, repo_type="dataset")
    
    data = {}
    etfs = []
    
    if option in ["a", "both"]:
        etfs.extend(["AGG"] + OPTION_A_ETFS)  # benchmark + etfs
    if option in ["b", "both"]:
        etfs.extend(["SPY"] + OPTION_B_ETFS)
    
    for etf in set(etfs):
        file_path = os.path.join(local_dir, f"{etf}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            data[etf] = df
            print(f"Loaded {etf}: {len(df)} rows")
    
    return data
