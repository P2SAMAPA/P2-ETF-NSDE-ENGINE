from huggingface_hub import hf_hub_download
import pandas as pd
import os

from config import (
    HF_DATASET_INPUT,
    OPTION_A_ETFS,
    OPTION_B_ETFS
)

def load_dataset(option: str = "both", include_benchmarks: bool = True):
    """
    Load ETF price data.
    option: 'a', 'b', or 'both'
    include_benchmarks: if True, include AGG for option 'a' and SPY for option 'b'/'both'.
    """
    print(f"Downloading dataset: {HF_DATASET_INPUT}")
    master_path = hf_hub_download(
        repo_id=HF_DATASET_INPUT,
        filename="data/master.parquet",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    df = pd.read_parquet(master_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    if option == "a":
        tickers_to_load = OPTION_A_ETFS.copy()
        if include_benchmarks:
            tickers_to_load.insert(0, "AGG")
    elif option == "b":
        tickers_to_load = OPTION_B_ETFS.copy()
        if include_benchmarks:
            tickers_to_load.insert(0, "SPY")
    else:  # both
        tickers_to_load = OPTION_A_ETFS + OPTION_B_ETFS
        if include_benchmarks:
            tickers_to_load = ["AGG", "SPY"] + tickers_to_load

    data = {}
    for ticker in tickers_to_load:
        possible_cols = [f"{ticker}_Close", f"{ticker}_close", f"Close_{ticker}"]
        close_col = None
        for col in possible_cols:
            if col in df.columns:
                close_col = col
                break
        if close_col is None:
            print(f"⚠️ No close price column for {ticker}")
            continue
        series = df[close_col].dropna()
        if len(series) == 0:
            continue
        data[ticker] = pd.DataFrame({'close': series})
        print(f"✅ Loaded {ticker}: {len(series)} rows")
    print(f"Total loaded ETF tickers: {len(data)}")
    return data

def load_macro_data():
    """Load macro variables (VIX, T10Y2Y, HY spread) – unchanged."""
    # ... (same as previous version)
    # (keep the existing implementation)
