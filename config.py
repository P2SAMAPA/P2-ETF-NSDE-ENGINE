import os

# Hugging Face
HF_DATASET_INPUT = "P2SAMAPA/p2-etf-deepm-data"
HF_DATASET_OUTPUT = "P2SAMAPA/p2-etf-nsde-engine-results"

# ETF Universes (including XLB and XLRE)
OPTION_A_ETFS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "PFF", "MBB"]
OPTION_B_ETFS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "XME"]

BENCHMARK_A = "AGG"
BENCHMARK_B = "SPY"

# Model & Training
HIDDEN_DIM = 64
FEATURE_DIM = 32
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DT = 0.1
SOLVER = "euler"

# Signal paths (used by predict and app)
SIGNAL_DIR = "signals"
os.makedirs(SIGNAL_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
