import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from config import *
from loader import load_dataset
from features import engineer_features
from model import NSDEModel
from huggingface_hub import HfApi, upload_file
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, default="both", choices=["a", "b", "both"])
    args = parser.parse_args()
    
    print("Loading data...")
    raw_data = load_dataset(args.option)
    
    # Simple training loop placeholder (full implementation would process paths)
    print("NSDE training not fully implemented in this skeleton.")
    print("Model architecture is ready for variance-preserving SDE.")
    print("Use this as base and extend the training logic as needed.")
    
    # Save dummy model for structure
    model = NSDEModel(feature_dim=32)
    model_path = "nsde_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved locally as {model_path}")

    # Upload the model to Hugging Face Dataset
    try:
        token = os.getenv("HF_TOKEN")  # Read from environment (set in GitHub Actions)
        if not token:
            # Fallback to config if defined, otherwise raise
            token = HF_TOKEN if 'HF_TOKEN' in globals() else None
        
        if token:
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=model_path,          # Root of the dataset repo
                repo_id=HF_DATASET_OUTPUT,
                repo_type="dataset",
                token=token,
            )
            print(f"✅ Successfully uploaded model to: {HF_DATASET_OUTPUT}")
        else:
            print("⚠️ HF_TOKEN not found. Model not uploaded.")
    except Exception as e:
        print(f"⚠️ Failed to upload model to Hugging Face: {e}")

if __name__ == "__main__":
    main()
