import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime
from huggingface_hub import hf_hub_download
import config as cfg
from calendar import next_trading_day, format_next_trading_day

st.set_page_config(
    page_title="P2 NSDE Engine",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

st.title("🧠 P2 ETF NSDE Engine")
st.markdown("**Neural Stochastic Differential Equations (Variance-Preserving SDE)** — Next-day probabilistic ETF forecasts with calibrated uncertainty")

# Prominent Next Trading Day
st.caption(f"**Signal For:** {format_next_trading_day()} • Model: NSDE (VP-SDE)")

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
        st.info("🚫 No signal data available yet.\n\nRun `python predict.py` or wait for the daily GitHub Action.")
        return

    fc = signal.get("forecasts", {})
    top = signal.get("top_pick")
    top_mu = signal.get("top_mu", 0.0)

    # Hero Card
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader(f"**Top Pick: {top}**")
        st.metric("Expected Return (μ)", f"{top_mu:.4f}", delta=None)
    with col2:
        confidence = fc.get(top, {}).get("confidence", 0)
        st.metric("Confidence", f"{confidence:.1%}")
    with col3:
        gen_time = signal.get("generated_at", "")
        try:
            gen_time = datetime.fromisoformat(gen_time).strftime("%Y-%m-%d %H:%M UTC")
        except:
            gen_time = "Unknown"
        st.metric("Generated", gen_time)

    # Regime Context Pills
    rc = signal.get("regime_context", {})
    st.markdown("**Current Market Regime**")
    pills = f"""
    <span style="background:#1e3a8a;color:white;padding:6px 12px;border-radius:20px;margin-right:8px;">
        VIX: {rc.get('VIX', '—')}
    </span>
    <span style="background:#1e3a8a;color:white;padding:6px 12px;border-radius:20px;margin-right:8px;">
        T10Y2Y: {rc.get('T10Y2Y', '—')}
    </span>
    <span style="background:#1e3a8a;color:white;padding:6px 12px;border-radius:20px;">
        HY Spread: {rc.get('HY_SPREAD', '—')} bp
    </span>
    """
    st.markdown(pills, unsafe_allow_html=True)

    # Forecast Bar Chart with Error Bars
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
        marker_color=['#3a5bd9' if t == top else '#64748b' for t in df["Ticker"]],
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

    # Detailed Table
    st.subheader("All Forecasts")
    styled_df = df.style.format({
        "μ": "{:.4f}",
        "σ": "{:.4f}",
        "Confidence": "{:.1%}"
    }).background_gradient(subset=["μ"], cmap="Blues")

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    # Footer info
    st.caption(f"Option {label} • {len(df)} ETFs • Generated via NSDE")

# Render both tabs
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
