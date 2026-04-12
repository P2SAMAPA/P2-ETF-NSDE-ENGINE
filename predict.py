import torch
from config import HF_DATASET_OUTPUT, OPTION_A_ETFS, OPTION_B_ETFS
from model import NSDEModel
from huggingface_hub import HfApi
import pandas as pd
import json
import os
from datetime import datetime

def generate_signals():
    print("Generating NSDE signals...")
    
    # Dummy prediction for structure
    signals = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "option": "both",
        "signals": []
    }
    
    # Example structure (replace with real inference later)
    for etf in OPTION_B_ETFS[:5]:
        signals["signals"].append({
            "ticker": etf,
            "mu": 0.012,
            "sigma": 0.018,
            "confidence": 0.75
        })
    
    # Save locally and push to HF
    os.makedirs("signals", exist_ok=True)
    with open("signals/latest_signals.json", "w") as f:
        json.dump(signals, f, indent=2)
    
    print("Signals generated and saved.")
    print(f"Upload to HF dataset: {HF_DATASET_OUTPUT}")

if __name__ == "__main__":
    generate_signals()
