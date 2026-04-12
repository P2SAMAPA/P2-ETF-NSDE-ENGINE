import json
import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download
from config import HF_DATASET_OUTPUT, OPTION_A_ETFS, OPTION_B_ETFS
from loader import load_dataset
from features import engineer_features
from model import NSDEModel

print("=== Starting update_daily.py ===")

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
        last_close = df['close'].iloc[-1]
        X_dict[ticker] = {
            'tensor': X_tensor,
            'last_close': last_close,
            'feature_dim': X.shape[1]
        }
    return X_dict

def generate_signals(option, model, device, lookback=20):
    print(f"--- Generating signals for Option {option} ---")
    if option == 'A':
        tickers = ['AGG'] + OPTION_A_ETFS
    else:
        tickers = ['SPY'] + OPTION_B_ETFS
    print(f"Loading data for option {option}...")
    raw_data = load_dataset(option.lower())
    inf_data = prepare_inference_data({t: raw_data[t] for t in tickers if t in raw_data}, lookback)
    forecasts = {}
    for ticker, info in inf_data.items():
        X = info['tensor'].to(device)
        t_span = torch.linspace(0, 1, steps=X.shape[1], device=device)
        with torch.no_grad():
            mu, log_sigma = model(X, t_span)
        sigma = torch.exp(log_sigma)
        forecasts[ticker] = {
            'mu': mu.item(),
            'sigma': sigma.item(),
            'confidence': 1 - 2 * (sigma.item() / (abs(mu.item()) + sigma.item() + 1e-8))
        }
        print(f"  {ticker}: mu={mu.item():.4f}, sigma={sigma.item():.4f}")
    if forecasts:
        top_pick = max(forecasts.items(), key=lambda x: x[1]['mu'])[0]
        top_mu = forecasts[top_pick]['mu']
    else:
        top_pick = None
        top_mu = 0.0
    regime_context = {"VIX": "N/A", "T10Y2Y": "N/A", "HY_SPREAD": "N/A"}
    signal = {
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts": forecasts,
        "top_pick": top_pick,
        "top_mu": top_mu,
        "regime_context": regime_context
    }
    return signal

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print("Downloading trained model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id=HF_DATASET_OUTPUT,
            filename="nsde_model.pth",
            repo_type="dataset",
            token=os.getenv("HF_TOKEN")
        )
        print(f"Model downloaded to: {model_path}")

        # Determine feature dimension from a sample ticker
        print("Loading sample data to infer feature dimension...")
        sample_data = load_dataset("both")
        sample_ticker = next(iter(sample_data.keys()))
        sample_df = engineer_features(sample_data[sample_ticker])
        feature_cols = ['vol_20', 'mom_10', 'mom_60']
        available = [c for c in feature_cols if c in sample_df.columns]
        if not available:
            available = ['log_return_feat']
        feature_dim = len(available)
        print(f"Feature dimension: {feature_dim}")

        # Instantiate model and load weights
        model = NSDEModel(feature_dim=feature_dim, hidden_dim=64)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")

        # Generate signals
        signal_A = generate_signals('A', model, device)
        signal_B = generate_signals('B', model, device)

        # Save locally
        os.makedirs("signals", exist_ok=True)
        with open("signals/signal_A.json", "w") as f:
            json.dump(signal_A, f, indent=2)
        with open("signals/signal_B.json", "w") as f:
            json.dump(signal_B, f, indent=2)
        print("Signals saved to signals/ directory.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
