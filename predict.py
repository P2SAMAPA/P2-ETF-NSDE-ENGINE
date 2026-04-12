import torch
import json
import os
from datetime import datetime
from config import (
    HF_DATASET_OUTPUT, OPTION_A_ETFS, OPTION_B_ETFS,
    SIGNAL_DIR, HF_TOKEN
)
from model import NSDEModel
from huggingface_hub import HfApi, upload_file

def generate_signals(option: str = "both"):
    print(f"Generating NSDE signals for option: {option}")

    # Placeholder inference (replace with real model loading + forward pass later)
    model = NSDEModel(feature_dim=32)
    # model.load_state_dict(torch.load("nsde_model.pth", map_location="cpu"))  # uncomment when model is trained

    signals_a = {"option": "A", "generated_at": datetime.utcnow().isoformat(), "forecasts": {}, "top_pick": None}
    signals_b = {"option": "B", "generated_at": datetime.utcnow().isoformat(), "forecasts": {}, "top_pick": None}

    # Dummy forecasts (replace with real predictions)
    dummy_etfs_a = OPTION_A_ETFS
    dummy_etfs_b = OPTION_B_ETFS

    for etfs, signals_dict in [(dummy_etfs_a, signals_a), (dummy_etfs_b, signals_b)]:
        forecasts = {}
        top_mu = -999
        top_ticker = None

        for ticker in etfs:
            mu = round(0.005 + torch.rand(1).item() * 0.03, 4)   # random for demo
            sigma = round(0.01 + torch.rand(1).item() * 0.02, 4)
            confidence = round(0.6 + torch.rand(1).item() * 0.35, 2)

            forecasts[ticker] = {
                "mu": mu,
                "sigma": sigma,
                "confidence": confidence
            }

            if mu > top_mu:
                top_mu = mu
                top_ticker = ticker

        signals_dict["forecasts"] = forecasts
        signals_dict["top_pick"] = top_ticker
        signals_dict["top_mu"] = top_mu
        signals_dict["regime_context"] = {
            "VIX": 18.5,
            "T10Y2Y": 0.35,
            "HY_SPREAD": 420
        }

    # Save locally
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    with open(f"{SIGNAL_DIR}/signal_A.json", "w") as f:
        json.dump(signals_a, f, indent=2)
    with open(f"{SIGNAL_DIR}/signal_B.json", "w") as f:
        json.dump(signals_b, f, indent=2)

    print("Signals saved locally.")

    # Upload to HF dataset
    api = HfApi()
    for file_name in ["signal_A.json", "signal_B.json"]:
        upload_file(
            path_or_fileobj=f"{SIGNAL_DIR}/{file_name}",
            path_in_repo=f"signals/{file_name}",
            repo_id=HF_DATASET_OUTPUT,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    print(f"Uploaded signals to {HF_DATASET_OUTPUT}/signals/")

if __name__ == "__main__":
    generate_signals("both")
