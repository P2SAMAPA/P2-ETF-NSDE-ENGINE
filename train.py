import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np
import os
import json
from datetime import datetime
from config import *
from loader import load_dataset
from features import engineer_features
from model import NSDEModel
from huggingface_hub import upload_file

def prepare_tensors(data_dict, lookback=20):
    X_list, y_list = [], []
    for ticker, df in data_dict.items():
        df = engineer_features(df)
        feature_cols = ['vol_20', 'mom_10', 'mom_60']
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            df['log_return_feat'] = df['log_return']
            available = ['log_return_feat']
        features = df[available].values
        targets = df['log_return'].shift(-1).values
        for i in range(lookback, len(features)-1):
            X_list.append(features[i-lookback:i])
            y_list.append(targets[i])
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    return X, y

def negative_log_likelihood(mu, log_sigma, y):
    sigma = torch.exp(log_sigma)
    return 0.5 * ((y - mu) / sigma).pow(2) + log_sigma

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
        forecasts[ticker] = {
            'mu': mu.item(),
            'sigma': sigma.item(),
            'confidence': 1 - 2 * (sigma.item() / (abs(mu.item()) + sigma.item() + 1e-8))
        }
    if forecasts:
        top_pick = max(forecasts.items(), key=lambda x: x[1]['mu'])[0]
        top_mu = forecasts[top_pick]['mu']
    else:
        top_pick = None
        top_mu = 0.0
    regime_context = {"VIX": "N/A", "T10Y2Y": "N/A", "HY_SPREAD": "N/A"}
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts": forecasts,
        "top_pick": top_pick,
        "top_mu": top_mu,
        "regime_context": regime_context
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, default="both", choices=["a", "b", "both"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback", type=int, default=20)
    args = parser.parse_args()

    print("Loading data...")
    raw_data = load_dataset(args.option)

    print("Preparing features and targets...")
    X, y = prepare_tensors(raw_data, lookback=args.lookback)
    print(f"Dataset shape: X {X.shape}, y {y.shape}")

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    feature_dim = X.shape[-1]
    model = NSDEModel(feature_dim=feature_dim, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    t_span = torch.linspace(0, 1, steps=X.shape[1], device=device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            mu, log_sigma = model(batch_X, t_span)
            loss = negative_log_likelihood(mu, log_sigma, batch_y.squeeze()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.6f}")

    # Save trained model
    model_path = "nsde_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved as {model_path}")

    # Upload model
    token = os.getenv("HF_TOKEN")
    if token:
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path,
            repo_id=HF_DATASET_OUTPUT,
            repo_type="dataset",
            token=token,
        )
        print("✅ Model uploaded to HF")
    else:
        print("⚠️ HF_TOKEN missing, model not uploaded")

    # Generate and upload signals
    model.eval()
    signal_A = generate_signals('A', model, device, args.lookback)
    signal_B = generate_signals('B', model, device, args.lookback)

    os.makedirs("signals", exist_ok=True)
    with open("signals/signal_A.json", "w") as f:
        json.dump(signal_A, f, indent=2)
    with open("signals/signal_B.json", "w") as f:
        json.dump(signal_B, f, indent=2)

    if token:
        for opt in ["A", "B"]:
            upload_file(
                path_or_fileobj=f"signals/signal_{opt}.json",
                path_in_repo=f"signals/signal_{opt}.json",
                repo_id=HF_DATASET_OUTPUT,
                repo_type="dataset",
                token=token,
            )
        print("✅ Signals uploaded to HF")
    print("All done.")

if __name__ == "__main__":
    main()
