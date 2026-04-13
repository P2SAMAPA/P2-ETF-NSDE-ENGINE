from huggingface_hub import hf_hub_download
import pandas as pd
import os

from config import (
    HF_DATASET_INPUT,
    OPTION_A_ETFS,
    OPTION_B_ETFS
)

def load_dataset(option: str = "both"):
    """Load ETF price data from master.parquet."""
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

    if option in ["a", "both"]:
        tickers_to_load = ["AGG"] + OPTION_A_ETFS
    elif option == "b":
        tickers_to_load = ["SPY"] + OPTION_B_ETFS
    else:
        tickers_to_load = ["AGG", "SPY"] + OPTION_A_ETFS + OPTION_B_ETFS

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
    """Load macro variables (VIX, T10Y2Y, HY spread) from master.parquet or macro_fred.parquet."""
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
    # Look for common macro column names
    macro_cols = {}
    for col in df.columns:
        if 'VIX' in col.upper():
            macro_cols['VIX'] = df[col]
        if 'T10Y2Y' in col.upper() or 'yield' in col.lower():
            macro_cols['T10Y2Y'] = df[col]
        if 'HY' in col.upper() and 'SPREAD' in col.upper():
            macro_cols['HY_SPREAD'] = df[col]
    if not macro_cols:
        # Fallback: try to load macro_fred.parquet
        try:
            macro_path = hf_hub_download(
                repo_id=HF_DATASET_INPUT,
                filename="data/macro_fred.parquet",
                repo_type="dataset",
                token=os.getenv("HF_TOKEN")
            )
            macro_df = pd.read_parquet(macro_path)
            if 'date' in macro_df.columns:
                macro_df['date'] = pd.to_datetime(macro_df['date'])
                macro_df.set_index('date', inplace=True)
            for col in macro_df.columns:
                if 'vix' in col.lower():
                    macro_cols['VIX'] = macro_df[col]
                if 't10y2y' in col.lower():
                    macro_cols['T10Y2Y'] = macro_df[col]
                if 'hy_spread' in col.lower():
                    macro_cols['HY_SPREAD'] = macro_df[col]
        except:
            pass
    if macro_cols:
        macro_df = pd.DataFrame(macro_cols).sort_index()
        macro_df = macro_df.ffill().bfill()  # fill missing
        print(f"Loaded macro variables: {list(macro_cols.keys())}")
        return macro_df
    else:
        print("⚠️ No macro variables found; using zeros placeholder.")
        # Return a dummy macro series with same index as first ETF (will be aligned later)
        return None
