# Simple wrapper for daily update
from predict import generate_signals

if __name__ == "__main__":
    print("Running daily NSDE signal update...")
    generate_signals()
    print("Daily update completed.")
