import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime
from huggingface_hub import hf_hub_download
import config as cfg
from trading_calendar import format_next_trading_day

st.set_page_config(
    page_title="P2 NSDE Engine",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# Custom CSS for hero box
st.markdown("""
<style>
.hero-box {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 2rem;
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
