import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import json
import pandas as pd
from datetime import datetime
from config import *
from loader import load_dataset, load_macro_data
from features import engineer_features
from model import NSDEModel
from huggingface_hub import upload_file

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_tensors(data_dict, macro_df, lookback=20):
    """
    Prepare tensors for training: price features X, macro features M, and target y.
    """
    X_list, M_list, y_list = [], [], []
    for ticker, df in data_dict.items():
        # Engineer features (includes macro alignment)
        df_feat = engineer_features(df, macro_df)
        # Separate price-derived and macro columns
        price_cols = [c for c in df_feat.columns if not c.startswith('macro_')]
        macro_cols = [c for c in df_feat.columns if c.startswith('macro_')]
        X_vals = df_feat[price_cols].values
        M_vals = df_feat[macro_cols].values
        # Target: next day's return
        targets = df['close'].pct_change().shift(-1).values
        for i in range(lookback, len(X_vals) - 1):
            X_list.append(X_vals[i - lookback:i])
            M_list.append(M_vals[i - lookback:i])
            y_list.append(targets[i])
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    M = torch.tensor(np.array(M_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    return X, M, y

def negative_log_likelihood(mu, log_sigma, y):
    """Gaussian negative log-likelihood loss."""
    sigma = torch.exp(log_sigma)
    return 0.5 * ((y - mu) / sigma).pow(2) + log_sigma

def generate_signals(option, model, device, macro_df, lookback=20):
    """Generate signals for a given option (used after training)."""
    from loader import load_dataset
    from features import engineer_features

    if option == 'A':
        tickers = ['AGG'] + OPTION_A_ETFS
    else:
        tickers = ['SPY'] + OPTION_B_ETFS

    raw_data = load_dataset(option.lower())
    inf_data = {}
    for ticker in tickers:
        if ticker not in raw_data:
            continue
        df_feat = engineer_features(raw_data[ticker], macro_df)
        price_cols = [c for c in df_feat.columns if not c.startswith('macro_')]
        macro_cols = [c for c in df_feat.columns if c.startswith('macro_')]
        X_vals = df_feat[price_cols].values
        M_vals = df_feat[macro_cols].values
        if len(X_vals) < lookback:
            continue
        X = X_vals[-lookback:]
        M = M_vals[-lookback:]
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        M_tensor = torch.tensor(M, dtype=torch.float32).unsqueeze(0)
        inf_data[ticker] = (X_tensor, M_tensor)

    forecasts = {}
    for ticker, (X, M) in inf_data.items():
        X = X.to(device)
        M = M.to(device)
        t_span = torch.linspace(0, 1, steps=X.shape[1], device=device)
        with torch.no_grad():
            mu, log_sigma = model(X, M, t_span)
        sigma = torch.exp(log_sigma)
        confidence = 1 - 2 * (sigma.item() / (abs(mu.item()) + sigma.item() + 1e-8))
        forecasts[ticker] = {'mu': mu.item(), 'sigma': sigma.item(), 'confidence': confidence}

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
        "regime_context": {}
    }
    return signal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, default="both", choices=["a", "b", "both"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    # Load ETF data
    print("Loading ETF data...")
    raw_data = load_dataset(args.option)

    # Load macro data
    print("Loading macro data...")
    macro_df = load_macro_data()
    if macro_df is None:
        # Create dummy macro if none available
        first_ticker = next(iter(raw_data.keys()))
        dummy_idx = raw_data[first_ticker].index
        macro_df = pd.DataFrame(index=dummy_idx, data={'dummy': 0.0})
        print("Using dummy macro data (zeros).")

    # Prepare tensors
    print("Preparing features and targets...")
    X, M, y = prepare_tensors(raw_data, macro_df, args.lookback)
    print(f"X shape: {X.shape}, M shape: {M.shape}, y shape: {y.shape}")

    dataset = TensorDataset(X, M, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model dimensions
    price_dim = X.shape[-1]
    macro_dim = M.shape[-1]
    print(f"Price feature dimension: {price_dim}, Macro dimension: {macro_dim}")

    model = NSDEModel(feature_dim=price_dim, macro_dim=macro_dim, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    t_span = torch.linspace(0, 1, steps=args.lookback, device=device)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_M, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_M = batch_M.to(device)
            batch_y = batch_y.to(device)
            mu, log_sigma = model(batch_X, batch_M, t_span)
            loss = negative_log_likelihood(mu, log_sigma, batch_y.squeeze()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f}")

    # Save model locally
    model_path = "nsde_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")

    # Upload model to HF
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=model_path,
                repo_id=HF_DATASET_OUTPUT,
                repo_type="dataset",
                token=token,
            )
            print("✅ Model uploaded to Hugging Face Hub")
        except Exception as e:
            print(f"⚠️ Model upload failed: {e}")
    else:
        print("⚠️ HF_TOKEN not set, model not uploaded.")

    # Generate and upload signals using the freshly trained model
    model.eval()
    print("Generating signals...")
    signal_A = generate_signals('A', model, device, macro_df, args.lookback)
    signal_B = generate_signals('B', model, device, macro_df, args.lookback)

    os.makedirs("signals", exist_ok=True)
    if signal_A:
        with open("signals/signal_A.json", "w") as f:
            json.dump(signal_A, f, indent=2)
        print("✅ Signal A saved")
    if signal_B:
        with open("signals/signal_B.json", "w") as f:
            json.dump(signal_B, f, indent=2)
        print("✅ Signal B saved")

    if token:
        try:
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
                    print(f"✅ Uploaded {local_path}")
        except Exception as e:
            print(f"⚠️ Signal upload failed: {e}")

    print("Training and signal generation complete.")

if __name__ == "__main__":
    main()
