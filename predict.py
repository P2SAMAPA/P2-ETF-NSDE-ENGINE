import json
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download, upload_file
from config import HF_DATASET_INPUT, HF_DATASET_OUTPUT, OPTION_A_ETFS, OPTION_B_ETFS
from loader import load_dataset
from features import engineer_features
from model import NSDEModel

# ---------- Helper: Prepare features for inference ----------
def prepare_inference_data(data_dict, lookback=20):
    """
    For each ticker, extract the last `lookback` days of features.
    Returns a dict: ticker -> (feature_tensor, last_close, next_day_target_placeholder)
    """
    X_dict = {}
    for ticker, df in data_dict.items():
        df = engineer_features(df)
        # Use same feature columns as in training
        feature_cols = ['vol_20', 'mom_10', 'mom_60']
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            df['log_return_feat'] = df['log_return']
            available = ['log_return_feat']
        features = df[available].values
        if len(features) < lookback:
            print(f"⚠️ {ticker}: insufficient data (need {lookback} days), skipping")
            continue
        # Take last `lookback` rows
        X = features[-lookback:]   # shape (lookback, n_features)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, lookback, feat_dim)
        last_close = df['close'].iloc[-1]
        X_dict[ticker] = {
            'tensor': X_tensor,
            'last_close': last_close,
            'feature_dim': X.shape[1]
        }
    return X_dict

# ---------- Main prediction routine ----------
def generate_signals(option, model, device, lookback=20):
    """Generate forecasts for all tickers in one option universe."""
    print(f"\n--- Generating signals for Option {option} ---")
    # Load only the required option data (fast)
    raw_data = load_dataset(option.lower())
    # Filter tickers: for Option A we need AGG as benchmark; for Option B we need SPY
    if option == 'A':
        tickers = ['AGG'] + OPTION_A_ETFS
    else:
        tickers = ['SPY'] + OPTION_B_ETFS
    
    # Prepare inference inputs
    inf_data = prepare_inference_data({t: raw_data[t] for t in tickers if t in raw_data}, lookback)
    forecasts = {}
    for ticker, info in inf_data.items():
        X = info['tensor'].to(device)
        # Integration time span (same as training)
        t_span = torch.linspace(0, 1, steps=X.shape[1], device=device)
        with torch.no_grad():
            mu, log_sigma = model(X, t_span)
        sigma = torch.exp(log_sigma)
        forecasts[ticker] = {
            'mu': mu.item(),
            'sigma': sigma.item(),
            'confidence': 1 - 2 * (sigma.item() / (abs(mu.item()) + sigma.item()))  # heuristic
        }
    # Identify top pick based on mu
    top_pick = max(forecasts.items(), key=lambda x: x[1]['mu'])[0] if forecasts else None
    top_mu = forecasts[top_pick]['mu'] if top_pick else 0.0
    
    # Build regime context (optional: extract from last available macro)
    regime_context = {"VIX": "N/A", "T10Y2Y": "N/A", "HY_SPREAD": "N/A"}
    # Try to read macro from raw_data if present (e.g., from load_dataset full)
    # For simplicity, we leave as N/A; you can extend later.
    
    signal = {
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts": forecasts,
        "top_pick": top_pick,
        "top_mu": top_mu,
        "regime_context": regime_context
    }
    return signal

def main():
    # ---- Configuration ----
    lookback = 20
    hidden_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- Load trained model from HF dataset ----
    print("Downloading trained model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id=HF_DATASET_OUTPUT,
        filename="nsde_model.pth",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    # Determine feature dimension by loading a sample data point
    # We'll load both options to get feature_dim (should be same across all)
    sample_data = load_dataset("both")
    sample_ticker = next(iter(sample_data.keys()))
    sample_df = engineer_features(sample_data[sample_ticker])
    feature_cols = ['vol_20', 'mom_10', 'mom_60']
    available = [c for c in feature_cols if c in sample_df.columns]
    if not available:
        sample_df['log_return_feat'] = sample_df['log_return']
        available = ['log_return_feat']
    feature_dim = len(available)
    
    # Instantiate model and load weights
    model = NSDEModel(feature_dim=feature_dim, hidden_dim=hidden_dim)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
    # ---- Generate signals for Option A and Option B ----
    signal_A = generate_signals('A', model, device, lookback)
    signal_B = generate_signals('B', model, device, lookback)
    
    # ---- Save locally and upload to HF dataset ----
    os.makedirs("signals", exist_ok=True)
    for opt, signal in [("A", signal_A), ("B", signal_B)]:
        local_path = f"signals/signal_{opt}.json"
        with open(local_path, "w") as f:
            json.dump(signal, f, indent=2)
        print(f"Saved {local_path}")
        # Upload to HF dataset
        try:
            token = os.getenv("HF_TOKEN")
            if not token:
                # fallback to config if defined (not recommended)
                token = None
            if token:
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=local_path,   # signals/signal_X.json
                    repo_id=HF_DATASET_OUTPUT,
                    repo_type="dataset",
                    token=token,
                )
                print(f"✅ Uploaded {local_path} to {HF_DATASET_OUTPUT}")
            else:
                print("⚠️ HF_TOKEN not found, signals not uploaded.")
        except Exception as e:
            print(f"⚠️ Failed to upload {local_path}: {e}")
    
    print("\nPrediction complete. Signals are now available in the HF dataset.")

if __name__ == "__main__":
    main()
