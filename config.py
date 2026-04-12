import os

# Hugging Face Datasets
HF_DATASET_INPUT = "P2SAMAPA/p2-etf-deepm-data"
HF_DATASET_OUTPUT = "P2SAMAPA/p2-etf-nsde-engine-results"

# ETF Universes
OPTION_A_ETFS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "PFF", "MBB"]
OPTION_B_ETFS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "XME"]

BENCHMARK_A = "AGG"
BENCHMARK_B = "SPY"

# Training & Model Config
SEQUENCE_LENGTH = 252
HIDDEN_DIM = 64
DRIFT_HIDDEN = 128
DIFFUSION_HIDDEN = 64
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
SOLVER = "euler"          # or "milstein" for SDE
DT = 0.1

# Paths (used locally or in HF)
LOCAL_SIGNAL_DIR = "./signals"
os.makedirs(LOCAL_SIGNAL_DIR, exist_ok=True)

# Environment
HF_TOKEN = os.getenv("HF_TOKEN")
