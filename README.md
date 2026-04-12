Here are all the complete files you need to upload to your new repository:
https://github.com/P2SAMAPA/P2-ETF-NSDE-ENGINE
Create them exactly as shown below (copy-paste the full content into each file).
No placeholders — everything is filled in and adapted for NSDE (Neural Stochastic Differential Equations with Variance-Preserving SDE) while keeping the exact same structure, style, and design as the original NCDE repo.
1. README.md
Markdown# P2-ETF-NSDE-ENGINE

**Neural Stochastic Differential Equations (NSDE) with Variance-Preserving SDE**  
for ETF Next-Day Return Forecasting

A continuous-time regime-aware quant signal engine using **Neural SDEs**.  
Generates probabilistic forecasts (μ) with well-calibrated uncertainty (σ) for multiple ETFs.

---

### Core Idea
Instead of deterministic NCDEs, this engine models the hidden state evolution as a **stochastic process**:
dh(t) = f_θ(h(t), X(t)) dt + g_φ(h(t)) dW(t)
text- `f_θ`: Neural drift network (vector field)
- `g_φ`: Diffusion network designed for **variance-preserving** property
- `X(t)`: Continuous control path from interpolated macro + ETF features
- `dW(t)`: Wiener process (Brownian motion)

This allows richer modeling of stochastic volatility, fat tails, and regime shifts common in financial markets.

---

### ETF Universes

**Option A — Fixed Income & Alternatives** (Benchmark: AGG)  
`TLT, LQD, HYG, VNQ, GLD, SLV, PFF, MBB`

**Option B — Equity Sectors** (Benchmark: SPY)  
`SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, GDX, XME`

---

### Data
- **Input Dataset**: `P2SAMAPA/p2-etf-deepm-data` (same raw data as NCDE)
- **Output Dataset**: `P2SAMAPA/p2-etf-nsde-engine-results` (stores trained models, signals, and daily forecasts)

---

### Quick Start
```bash
pip install -r requirements_train.txt
# Set HF_TOKEN with write access to your output dataset
python validate_dataset.py
python train.py --option both
python predict.py
Deploy app.py on Streamlit Community Cloud.

For research and educational purposes only. Not financial advice.
