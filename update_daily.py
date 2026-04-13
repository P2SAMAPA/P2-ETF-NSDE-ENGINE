import json
import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download
from config import HF_DATASET_OUTPUT, OPTION_A_ETFS, OPTION_B_ETFS
from loader import load_dataset, load_macro_data
from features import engineer_features
from model import NSDEModel

print("=== Starting update_daily.py with macro control path ===")

def prepare_inference_data(data_dict, macro_df, lookback=20):
    """
    For each ticker, extract the last `lookback` days of price features and macro features.
    Returns a dict: ticker -> (price_tensor, macro_tensor, last_close)
    """
    X_dict = {}
    for ticker, df in data_dict.items():
        df_feat = engineer_features(df, macro_df)
        # Separate price-derived columns (without 'macro_' prefix) and macro columns
        price_cols = [c for c in df_feat.columns if not c.startswith('macro_')]
        macro_cols = [c for c in df_feat.columns if c.startswith('macro_')]
        X_vals = df_feat[price_cols].values
        M_vals = df_feat[macro_cols].values
        if len(X_vals) < lookback:
            continue
        # Take last `lookback` rows
        X = X_vals[-lookback:]   # (lookback, price_feat_dim)
        M = M_vals[-lookback:]   # (lookback, macro_dim)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, lookback, price_dim)
        M_tensor = torch.tensor(M, dtype=torch.float32).unsqueeze(0)  # (1, lookback, macro_dim)
        last_close = df['close'].iloc[-1]
        X_dict[ticker] = {
            'price_tensor': X_tensor,
            'macro_tensor': M_tensor,
            'last_close': last_close,
            'price_dim': X.shape[1],
            'macro_dim': M.shape[1]
        }
    return X_dict

def generate_signals(option, model, device, lookback=20):
    print(f"--- Generating signals for Option {option} ---")
    if option == 'A':
        tickers = ['AGG'] + OPTION_A_ETFS
    else:
        tickers = ['SPY'] + OPTION_B_ETFS

    # Load ETF data for the option
    raw_data = load_dataset(option.lower())
    # Load macro data (same for both options)
    macro_df = load_macro_data()
    if macro_df is None:
        # Create dummy macro with one column of zeros using first ETF's index
        first_ticker = next(iter(raw_data.keys()))
        dummy_idx = raw_data[first_ticker].index
        macro_df = pd.DataFrame(index=dummy_idx, data={'dummy': 0.0})

    inf_data = prepare_inference_data(
        {t: raw_data[t] for t in tickers if t in raw_data},
        macro_df, lookback
    )
    if not inf_data:
        print(f"⚠️ No valid inference data for Option {option}")
        return None

    # Determine price_dim and macro_dim from first ticker (should be same across)
    sample = next(iter(inf_data.values()))
    price_dim = sample['price_dim']
    macro_dim = sample['macro_dim']

    forecasts = {}
    for ticker, info in inf_data.items():
        X = info['price_tensor'].to(device)
        M = info['macro_tensor'].to(device)
        t_span = torch.linspace(0, 1, steps=X.shape[1], device=device)
        with torch.no_grad():
            mu, log_sigma = model(X, M, t_span)
        sigma = torch.exp(log_sigma)
        # Heuristic confidence (avoid division by zero)
        confidence = 1 - 2 * (sigma.item() / (abs(mu.item()) + sigma.item() + 1e-8))
        forecasts[ticker] = {
            'mu': mu.item(),
            'sigma': sigma.item(),
            'confidence': confidence
        }
        print(f"  {ticker}: mu={mu.item():.4f}, sigma={sigma.item():.4f}")

    if forecasts:
        top_pick = max(forecasts.items(), key=lambda x: x[1]['mu'])[0]
        top_mu = forecasts[top_pick]['mu']
    else:
        top_pick = None
        top_mu = 0.0

    signal = {
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts": forecasts,
        "top_pick": top_pick,
        "top_mu": top_mu,
        "regime_context": {}   # macro not displayed, but internally used
    }
    return signal

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Download trained model from HF
    print("Downloading trained model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id=HF_DATASET_OUTPUT,
        filename="nsde_model.pth",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )

    # 2. Determine model dimensions by loading a sample of data
    print("Loading sample data to infer price_dim and macro_dim...")
    sample_data = load_dataset("both")
    sample_ticker = next(iter(sample_data.keys()))
    sample_macro = load_macro_data()
    if sample_macro is None:
        sample_macro = pd.DataFrame(index=sample_data[sample_ticker].index, data={'dummy': 0.0})
    sample_feat = engineer_features(sample_data[sample_ticker], sample_macro)
    price_cols = [c for c in sample_feat.columns if not c.startswith('macro_')]
    macro_cols = [c for c in sample_feat.columns if c.startswith('macro_')]
    price_dim = len(price_cols)
    macro_dim = len(macro_cols)
    print(f"Price feature dimension: {price_dim}, Macro dimension: {macro_dim}")

    # 3. Instantiate model and load weights
    model = NSDEModel(feature_dim=price_dim, macro_dim=macro_dim, hidden_dim=64)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 4. Generate signals for Option A and Option B
    signal_A = generate_signals('A', model, device)
    signal_B = generate_signals('B', model, device)

    if signal_A is None and signal_B is None:
        print("❌ No signals generated. Exiting.")
        sys.exit(1)

    # 5. Save signals locally
    os.makedirs("signals", exist_ok=True)
    if signal_A:
        with open("signals/signal_A.json", "w") as f:
            json.dump(signal_A, f, indent=2)
        print("✅ Signal A saved to signals/signal_A.json")
    if signal_B:
        with open("signals/signal_B.json", "w") as f:
            json.dump(signal_B, f, indent=2)
        print("✅ Signal B saved to signals/signal_B.json")

    # 6. (Optional) Upload directly to HF – the workflow will also do it, but we do it here for completeness.
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import upload_file
            for opt in ["A", "B"]:
                local_path = f"signals/signal_{opt}.json"
                if os.path.exists(local_path):
                    upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=local_path,
                        repo_id=HF_DATASET_OUTPUT,
                        repo_type="dataset",
                        token=token,
                    )
                    print(f"✅ Uploaded {local_path} to HF")
        except Exception as e:
            print(f"⚠️ Upload failed: {e}")
    else:
        print("⚠️ HF_TOKEN not set; signals not uploaded.")

    print("=== update_daily.py finished ===")

if __name__ == "__main__":
    main()
