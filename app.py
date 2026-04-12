import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import json
from datetime import datetime
from config import HF_DATASET_OUTPUT

st.set_page_config(page_title="P2 ETF NSDE Engine", layout="wide")
st.title("🧠 P2 ETF NSDE Engine")
st.markdown("**Neural Stochastic Differential Equations (Variance-Preserving SDE)** — Next-day ETF forecasts with uncertainty")

# Load latest signals from HF dataset
@st.cache_data
def load_signals():
    try:
        file_path = hf_hub_download(repo_id=HF_DATASET_OUTPUT, filename="latest_signals.json", repo_type="dataset")
        with open(file_path) as f:
            return json.load(f)
    except:
        return {"date": "N/A", "signals": []}

signals = load_signals()

col1, col2 = st.columns([2, 1])
with col1:
    st.metric("Latest Signal Date", signals.get("date", "N/A"))
with col2:
    st.metric("Model Type", "NSDE (Variance-Preserving)")

st.subheader("Top Forecasts")
if signals["signals"]:
    df = pd.DataFrame(signals["signals"])
    df = df.sort_values("mu", ascending=False)
    st.dataframe(df.style.format({
        "mu": "{:.2%}",
        "sigma": "{:.2%}",
        "confidence": "{:.1%}"
    }), use_container_width=True)
else:
    st.info("No signals available yet. Run predict.py first.")

st.caption("Built with Neural SDEs • Data from p2-etf-deepm-data • Results stored in p2-etf-nsde-engine-results")
