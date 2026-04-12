from loader import load_dataset

if __name__ == "__main__":
    print("Validating input dataset...")
    data = load_dataset("both")
    print(f"Successfully loaded {len(data)} tickers.")
    print("Dataset validation passed.")
