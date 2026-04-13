import json
import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
from huggingface_hub import hf_hub_download
from config import HF_DATASET_OUTPUT, OPTION_A_ETFS, OPTION_B_ETFS
from loader import load_dataset
from features import engineer_features
from model import NSDEModel

def load_macro_data():
    """Load latest macro indicators from the input dataset."""
    from huggingface_hub import hf_hub_download
    macro_path = hf_hub_download(
        repo_id="P2SAMAPA/p2-etf-deepm-data",
        filename="data/master.parquet",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    df = pd.read_parquet(macro_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
    # Look for VIX, T10Y2Y, HY spread columns (adjust names as needed)
    vix_col = next((c for c in df.columns if 'vix' in c.lower()), None)
    t10y2y_col = next((c for c in df.columns if 't10y2y' in c.lower() or 'yield' in c.lower()), None)
    hy_col = next((c for c in df.columns if 'hy' in c.lower() and 'spread' in c.lower()), None)
    latest = df.iloc[-1]
    return {
        "VIX": round(latest[vix_col], 2) if vix_col else "N/A",
        "T10Y2Y": round(latest[t10y2y_col], 2) if t10y2y_col else "N/A",
        "HY_SPREAD": round(latest[hy_col], 0) if hy_col else "N/A"
    }

def prepare_inference_data(data_dict, lookback=20):
    X_dict = {}
    for ticker, df in data_dict.items():
        df = engineer_features(df)
        feature_cols = ['vol_20', 'mom_10', 'mom_60']
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            df['log_return_feat'] = df['log_return']
            available = ['log_return_feat']
        features = df[available].values
        if len(features) < lookback:
            continue
        X = features[-lookback:]
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        X_dict[ticker] = {
            'tensor': X_tensor,
            'feature_dim': X.shape[1]
        }
    return X_dict

def generate_signals(option, model, device, lookback=20):
    print(f"Generating signals for Option {option}...")
    if option == 'A':
        tickers = ['AGG'] + OPTION_A_ETFS
    else:
        tickers = ['SPY'] + OPTION_B_ETFS
    raw_data = load_dataset(option.lower())
    inf_data = prepare_inference_data({t: raw_data[t] for t in tickers if t in raw_data}, lookback)
    forecasts = {}
    for ticker, info in inf_data.items():
        X = info['tensor'].to(device)
        t_span = torch.linspace(0, 1, steps=X.shape[1], device=device)
        with torch.no_grad():
            mu, log_sigma = model(X, t_span)
        sigma = torch.exp(log_sigma)
        # Probability of positive return (confidence)
        prob_pos = norm.cdf(mu.item() / sigma.item()) if sigma.item() > 0 else 0.5
        forecasts[ticker] = {
            'mu': mu.item(),
            'sigma': sigma.item(),
            'confidence': prob_pos
        }
    if forecasts:
        top_pick = max(forecasts.items(), key=lambda x: x[1]['mu'])[0]
        top_mu = forecasts[top_pick]['mu']
    else:
        top_pick = None
        top_mu = 0.0
    # Load macro regime
    regime_context = load_macro_data()
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts": forecasts,
        "top_pick": top_pick,
        "top_mu": top_mu,
        "regime_context": regime_context
    }

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print("Downloading trained model...")
        model_path = hf_hub_download(
            repo_id=HF_DATASET_OUTPUT,
            filename="nsde_model.pth",
            repo_type="dataset",
            token=os.getenv("HF_TOKEN")
        )
        # Determine feature dimension
        sample_data = load_dataset("both")
        sample_ticker = next(iter(sample_data.keys()))
        sample_df = engineer_features(sample_data[sample_ticker])
        feature_cols = ['vol_20', 'mom_10', 'mom_60']
        available = [c for c in feature_cols if c in sample_df.columns]
        if not available:
            available = ['log_return_feat']
        feature_dim = len(available)
        model = NSDEModel(feature_dim=feature_dim, hidden_dim=64)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        signal_A = generate_signals('A', model, device)
        signal_B = generate_signals('B', model, device)
        os.makedirs("signals", exist_ok=True)
        with open("signals/signal_A.json", "w") as f:
            json.dump(signal_A, f, indent=2)
        with open("signals/signal_B.json", "w") as f:
            json.dump(signal_B, f, indent=2)
        print("Signals saved.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
