import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np
import os
from config import *
from loader import load_dataset
from features import engineer_features
from model import NSDEModel
from huggingface_hub import upload_file

def prepare_tensors(data_dict, lookback=20):
    """
    Convert raw price data into (X, y) tensors.
    X: features over lookback window (log returns, volatility, momentum, macro)
    y: next‑day return (target)
    """
    X_list, y_list = [], []
    for ticker, df in data_dict.items():
        df = engineer_features(df)
        # Use close prices and engineered features
        # For simplicity, use log_return as target; for features use [vol_20, mom_10, mom_60]
        feature_cols = ['vol_20', 'mom_10', 'mom_60']
        # Ensure all required columns exist
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            # Fallback: use log_return and rolling mean
            df['mom_5'] = df['close'].pct_change(5)
            df['vol_10'] = df['log_return'].rolling(10).std()
            available = ['mom_5', 'vol_10']
        features = df[available].values
        targets = df['log_return'].shift(-1).values  # next day return

        # Create windows
        for i in range(lookback, len(features)-1):
            X_list.append(features[i-lookback:i])
            y_list.append(targets[i])
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    return X, y

def negative_log_likelihood(mu, log_sigma, y):
    """Gaussian NLL loss"""
    sigma = torch.exp(log_sigma)
    return 0.5 * ((y - mu) / sigma).pow(2) + log_sigma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, default="both", choices=["a", "b", "both"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback", type=int, default=20, help="Number of past days as input")
    args = parser.parse_args()

    print("Loading data...")
    raw_data = load_dataset(args.option)

    print("Preparing features and targets...")
    X, y = prepare_tensors(raw_data, lookback=args.lookback)
    print(f"Dataset shape: X {X.shape}, y {y.shape}")

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model: feature_dim = number of features per time step
    # In our prepare_tensors, we used available features; we need to fix dimension.
    # For simplicity, we'll compute feature_dim from X.
    feature_dim = X.shape[-1]
    model = NSDEModel(feature_dim=feature_dim, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Integration time span (t from 0 to 1)
    t_span = torch.linspace(0, 1, steps=X.shape[1], device=device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)      # (batch, lookback, feat_dim)
            batch_y = batch_y.to(device)      # (batch, 1)

            # NSDE forward pass: returns mu, log_sigma
            mu, log_sigma = model(batch_X, t_span)

            loss = negative_log_likelihood(mu, log_sigma, batch_y.squeeze()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f}")

    # Save trained model
    model_path = "nsde_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved locally as {model_path}")

    # Upload to Hugging Face Dataset
    try:
        token = os.getenv("HF_TOKEN")
        if not token:
            token = HF_TOKEN if 'HF_TOKEN' in globals() else None
        if token:
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=model_path,
                repo_id=HF_DATASET_OUTPUT,
                repo_type="dataset",
                token=token,
            )
            print(f"✅ Successfully uploaded trained model to: {HF_DATASET_OUTPUT}")
        else:
            print("⚠️ HF_TOKEN not found. Model not uploaded.")
    except Exception as e:
        print(f"⚠️ Failed to upload model: {e}")

if __name__ == "__main__":
    main()
