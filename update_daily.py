from predict import generate_signals

if __name__ == "__main__":
    print("=== Daily NSDE Signal Update ===")
    generate_signals("both")
    print("Daily update completed successfully.")
