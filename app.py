import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime
from huggingface_hub import hf_hub_download
import config as cfg
from trading_calendar import next_trading_day, format_next_trading_day
import numpy as np

st.set_page_config(
    page_title="P2 NSDE Engine",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .hero-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: #1e293b;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #3b82f6;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 P2 ETF NSDE Engine")
st.markdown("**Neural Stochastic Differential Equations (Variance-Preserving SDE)** — Next-day probabilistic ETF forecasts with calibrated uncertainty")

# Hero box with next trading day and top picks summary
nxt_date = format_next_trading_day()
st.markdown(f"""
<div class="hero-box">
    <div class="hero-title">Next Trading Day: {nxt_date}</div>
    <div class="hero-subtitle">Model: NSDE (VP-SDE) • Variance-Preserving Neural SDE</div>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_signal(opt: str):
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_OUTPUT,
            filename=f"signals/signal_{opt}.json",
            repo_type="dataset"
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load signal_{opt}.json — {str(e)[:100]}")
        return None

def compute_etf_metrics(ticker):
    """Compute annualized return, Sharpe ratio, max drawdown from historical data."""
    try:
        from loader import load_dataset
        data = load_dataset("both")
        if ticker not in data:
            return None
        prices = data[ticker]['close']
        returns = prices.pct_change().dropna()
        if len(returns) < 2:
            return None
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        return {
            "ann_return": ann_return,
            "sharpe": sharpe,
            "max_dd": max_dd
        }
    except:
        return None

signal_a = load_signal("A")
signal_b = load_signal("B")

tab_a, tab_b = st.tabs(["Option A — Fixed Income & Alternatives", "Option B — Equity Sectors"])

def render_tab(signal, label):
    if not signal:
        st.info("🚫 No signal data available yet.\n\nRun `python update_daily.py` or wait for the daily GitHub Action.")
        return

    fc = signal.get("forecasts", {})
    top = signal.get("top_pick")
    top_mu = signal.get("top_mu", 0.0)
    regime = signal.get("regime_context", {})
    gen_time = signal.get("generated_at", "")
    try:
        gen_time = datetime.fromisoformat(gen_time).strftime("%Y-%m-%d %H:%M UTC")
    except:
        gen_time = "Unknown"

    # Top pick hero card
    st.markdown(f"""
    <div style="background:#0f172a; border-left: 5px solid #3b82f6; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
        <div style="font-size: 0.8rem; color:#94a3b8; text-transform:uppercase;">Top Pick</div>
        <div style="font-size: 2.5rem; font-weight:bold; color:#3b82f6;">{top}</div>
        <div style="font-size: 1.2rem;">Expected Return: {top_mu:.4f}</div>
        <div style="font-size: 0.9rem; color:#94a3b8;">Generated: {gen_time}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row for the top pick (if available)
    metrics = compute_etf_metrics(top) if top else None
    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annualized Return", f"{metrics['ann_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_dd']:.2%}")

    # Regime context as small pills
    st.markdown("**Current Market Regime**")
    cols = st.columns(3)
    cols[0].metric("VIX", regime.get("VIX", "N/A"))
    cols[1].metric("T10Y2Y", regime.get("T10Y2Y", "N/A"))
    cols[2].metric("HY Spread (bp)", regime.get("HY_SPREAD", "N/A"))

    # Prepare data for bar chart
    tickers = cfg.OPTION_A_ETFS if label == "A" else cfg.OPTION_B_ETFS
    data = []
    for t in tickers:
        if t in fc:
            f = fc[t]
            data.append({
                "Ticker": t,
                "μ": f.get("mu", 0),
                "σ": f.get("sigma", 0.015),
                "Confidence": f.get("confidence", 0.5)
            })
    if not data:
        st.warning("No forecast data for this option.")
        return

    df = pd.DataFrame(data).sort_values("μ", ascending=False)

    # Plotly bar chart with error bars
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Ticker"],
        x=df["μ"],
        orientation='h',
        error_x=dict(type='data', array=df["σ"], visible=True, color="#94a3b8"),
        marker_color=['#3b82f6' if t == top else '#64748b' for t in df["Ticker"]],
        hovertemplate="<b>%{y}</b><br>μ = %{x:.4f}<br>σ = %{error_x.array:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title="Next-Day Return Forecasts (μ ± 1σ)",
        xaxis_title="Expected Return",
        height=620,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table with confidence as percentage
    df_display = df.copy()
    df_display["Confidence"] = df_display["Confidence"].apply(lambda x: f"{x:.1%}")
    df_display["μ"] = df_display["μ"].apply(lambda x: f"{x:.4f}")
    df_display["σ"] = df_display["σ"].apply(lambda x: f"{x:.4f}")
    st.subheader("All Forecasts")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.caption(f"Option {label} • {len(df)} ETFs • Generated via NSDE")

with tab_a:
    st.subheader("Option A — Fixed Income & Alternatives (Benchmark: AGG)")
    render_tab(signal_a, "A")

with tab_b:
    st.subheader("Option B — Equity Sectors (Benchmark: SPY)")
    render_tab(signal_b, "B")

st.divider()
st.caption(f"""
**P2-ETF-NSDE-ENGINE** • Variance-Preserving Neural SDE  
Input: `{cfg.HF_DATASET_INPUT}` • Output: `{cfg.HF_DATASET_OUTPUT}`  
Built for research & educational purposes only — Not financial advice.
""")
