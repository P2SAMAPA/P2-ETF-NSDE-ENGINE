from predict import generate_signals
from model import NSDEModel
from config import HF_DATASET_OUTPUT
from huggingface_hub import hf_hub_download
import torch
import os

def main():
    print("=== Daily NSDE Signal Update ===")
    
    # Load the trained model from HF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = hf_hub_download(
        repo_id=HF_DATASET_OUTPUT,
        filename="nsde_model.pth",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    
    # Determine feature dimension (same as in predict.py)
    # We'll load a sample data point to infer feature_dim
    from loader import load_dataset
    from features import engineer_features
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
    model = NSDEModel(feature_dim=feature_dim, hidden_dim=64)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Generate signals for both options
    print("Generating signals for Option A...")
    signal_A = generate_signals('A', model, device)
    print("Generating signals for Option B...")
    signal_B = generate_signals('B', model, device)
    
    # Save and upload signals (handled inside generate_signals already if predict.py includes upload)
    # But to be safe, we call the upload logic again? Actually predict.py's generate_signals returns dict but doesn't upload.
    # We'll let the workflow handle upload via the separate step. So we just need to save locally.
    import json
    import os
    os.makedirs("signals", exist_ok=True)
    with open("signals/signal_A.json", "w") as f:
        json.dump(signal_A, f, indent=2)
    with open("signals/signal_B.json", "w") as f:
        json.dump(signal_B, f, indent=2)
    print("Signals saved locally in signals/ directory.")
    # The workflow will upload them via the hf upload step.

if __name__ == "__main__":
    main()
