import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime
import numpy as np
from huggingface_hub import hf_hub_download
import config as cfg
from trading_calendar import format_next_trading_day
from loader import load_dataset

st.set_page_config(
    page_title="P2 NSDE Engine",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# Custom CSS for hero box and metrics
st.markdown("""
<style>
.hero-box {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
}
.top-pick-label {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: 0.8;
}
.top-pick-ticker {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.25rem 0;
}
.expected-return {
    font-size: 1.8rem;
    font-weight: 600;
    margin: 0.5rem 0;
}
.small-date {
    font-size: 0.8rem;
    opacity: 0.8;
    margin-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.2);
    padding-top: 0.75rem;
}
.metrics-container {
    background: #f8fafc;
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem 0;
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    border: 1px solid #e2e8f0;
}
.metric-card {
    text-align: center;
    padding: 0.5rem 1rem;
}
.metric-label {
    font-size: 0.8rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 P2 ETF NSDE Engine")
st.markdown("**Neural Stochastic Differential Equations (Variance-Preserving SDE)** — Probabilistic ETF forecasts with calibrated uncertainty")

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

@st.cache_data(ttl=3600)
def load_historical_prices():
    """Load all historical close prices from the input dataset."""
    data = load_dataset("both")
    return {ticker: df['close'] for ticker, df in data.items()}

def compute_metrics(price_series):
    """Compute annualized return, Sharpe ratio, max drawdown over available period."""
    if price_series is None or len(price_series) < 2:
        return None, None, None
    daily_returns = price_series.pct_change().dropna()
    if len(daily_returns) == 0:
        return None, None, None
    # Annualized return (assuming 252 trading days)
    ann_return = (price_series.iloc[-1] / price_series.iloc[0]) ** (252 / len(daily_returns)) - 1
    # Annualized volatility
    ann_vol = daily_returns.std() * np.sqrt(252)
    # Sharpe ratio (risk-free rate = 0)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    # Max drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    return ann_return, sharpe, max_dd

# Preload historical prices
historical_prices = load_historical_prices()

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
    gen_time = signal.get("generated_at", "")
    try:
        gen_time = datetime.fromisoformat(gen_time).strftime("%Y-%m-%d %H:%M UTC")
    except:
        gen_time = "Unknown"

    # Hero box with top pick and next trading date inside
    st.markdown(f"""
    <div class="hero-box">
        <div class="top-pick-label">TOP PICK</div>
        <div class="top-pick-ticker">{top}</div>
        <div class="expected-return">Expected Return: {top_mu:.4f}</div>
        <div class="small-date">📅 US Markets Next Trading Day: {format_next_trading_day()} • Generated: {gen_time}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Metrics for the top pick (separate container) ----
    if top and top in historical_prices:
        ann_ret, sharpe, max_dd = compute_metrics(historical_prices[top])
        if ann_ret is not None:
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.markdown(f'<div class="metric-card"><div class="metric-label">Annual Return</div><div class="metric-value">{ann_ret:.2%}</div></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{sharpe:.2f}</div></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-value">{max_dd:.2%}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("⚠️ Historical price data not available for top pick to compute metrics.")

    # Regime Context as metrics row
    rc = signal.get("regime_context", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("VIX", rc.get('VIX', 'N/A'))
    col2.metric("T10Y2Y (Yield Curve)", rc.get('T10Y2Y', 'N/A'))
    col3.metric("HY Spread (bps)", rc.get('HY_SPREAD', 'N/A'))

    # Forecast bar chart
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
    df = pd.DataFrame(data).sort_values("μ", ascending=False)

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

    # Detailed table
    st.subheader("All Forecasts")
    styled_df = df.style.format({
        "μ": "{:.4f}",
        "σ": "{:.4f}",
        "Confidence": "{:.1%}"
    }).background_gradient(subset=["μ"], cmap="Blues")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.caption(f"Option {label} • {len(df)} ETFs • Model: NSDE (VP-SDE)")

with tab_a:
    render_tab(signal_a, "A")
with tab_b:
    render_tab(signal_b, "B")

st.divider()
st.caption("**P2-ETF-NSDE-ENGINE** • Variance-Preserving Neural SDE • Research only – Not financial advice.")
