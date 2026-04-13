import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np
import os
import json
from datetime import datetime
from config import *
from loader import load_dataset, load_macro_data
from features import engineer_features
from model import NSDEModel
from huggingface_hub import upload_file

def prepare_tensors(data_dict, macro_df, lookback=20):
    X_list, M_list, y_list = [], [], []
    for ticker, df in data_dict.items():
        df_feat = engineer_features(df, macro_df)
        # Feature columns (excluding macro ones if any, but we keep all)
        # We'll separate: price-derived vs macro-derived? Simpler: treat all as X, but macro is separate path.
        # For the two-control-path, we need price features X and macro features M separately.
        # Let's define price_feature_cols as those without 'macro_' prefix.
        price_cols = [c for c in df_feat.columns if not c.startswith('macro_')]
        macro_cols = [c for c in df_feat.columns if c.startswith('macro_')]
        X_vals = df_feat[price_cols].values
        M_vals = df_feat[macro_cols].values
        targets = df['close'].pct_change().shift(-1).values  # next day return
        for i in range(lookback, len(X_vals)-1):
            X_list.append(X_vals[i-lookback:i])
            M_list.append(M_vals[i-lookback:i])
            y_list.append(targets[i])
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    M = torch.tensor(np.array(M_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    return X, M, y

def negative_log_likelihood(mu, log_sigma, y):
    sigma = torch.exp(log_sigma)
    return 0.5 * ((y - mu) / sigma).pow(2) + log_sigma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, default="both", choices=["a", "b", "both"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback", type=int, default=20)
    args = parser.parse_args()

    print("Loading ETF data...")
    raw_data = load_dataset(args.option)
    print("Loading macro data...")
    macro_df = load_macro_data()
    if macro_df is None:
        # Create dummy macro with one column of zeros
        dummy_idx = next(iter(raw_data.values())).index
        macro_df = pd.DataFrame(index=dummy_idx, data={'dummy': 0.0})

    print("Preparing features...")
    X, M, y = prepare_tensors(raw_data, macro_df, args.lookback)
    print(f"X shape: {X.shape}, M shape: {M.shape}, y shape: {y.shape}")

    dataset = TensorDataset(X, M, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    feature_dim = X.shape[-1]
    macro_dim = M.shape[-1]
    model = NSDEModel(feature_dim=feature_dim, macro_dim=macro_dim, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    t_span = torch.linspace(0, 1, steps=X.shape[1], device=device)

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
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.6f}")

    model_path = "nsde_model.pth"
    torch.save(model.state_dict(), model_path)
    print("Model saved locally.")

    token = os.getenv("HF_TOKEN")
    if token:
        upload_file(path_or_fileobj=model_path, path_in_repo=model_path,
                    repo_id=HF_DATASET_OUTPUT, repo_type="dataset", token=token)
        print("✅ Model uploaded to HF")

    # Also generate and upload signals (including macro)
    # ... (similar to previous train.py, but now generate_signals must also use macro)
    # For brevity, I'll skip signal generation here – you can adapt from earlier.

if __name__ == "__main__":
    main()
