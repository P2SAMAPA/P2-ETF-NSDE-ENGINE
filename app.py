# app.py — P2-ETF-NSDE-ENGINE Streamlit Dashboard
#
# Reads latest signals from P2SAMAPA/p2-etf-nsde-engine-results HF dataset.
# Tab layout: Option A | Option B
# Shows: top pick + mu/confidence hero, per-ETF forecast bar chart,
# regime context pills, signal history table.

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download
import pandas_market_calendars as mcal

import config as cfg

st.set_page_config(
    page_title="NSDE — ETF Signal Engine",
    page_icon="∂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🧠 P2 ETF NSDE Engine")
st.markdown("**Neural Stochastic Differential Equations (Variance-Preserving SDE)** — Next-day probabilistic ETF forecasts")

nyse = mcal.get_calendar("NYSE")

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_signals() -> dict:
    result = {}
    for opt in ["A", "B"]:
        try:
            path = hf_hub_download(
                repo_id=cfg.HF_DATASET_OUTPUT,
                filename=f"signals/signal_{opt}.json",
                repo_type="dataset",
                token=cfg.HF_TOKEN or None,
                force_download=True,
            )
            with open(path) as f:
                data = json.load(f)
            if "option" not in data:
                data["option"] = opt
            result[opt] = data
        except Exception as e:
            st.warning(f"Could not load signal_{opt}.json: {e}")
            result[opt] = {}
    return result

@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_INPUT,
            filename="master.parquet",   # adjust if your master file has different name
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        df = pd.read_parquet(path)
        for col in ["Date", "date"]:
            if col in df.columns:
                df = df.set_index(col)
                break
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()
    except Exception as e:
        st.error(f"Could not load master dataset: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_history(option: str) -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_OUTPUT,
            filename=f"signals/signal_history_{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    except Exception:
        return pd.DataFrame()

def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    sched = nyse.schedule(start_date=date, end_date=date + pd.Timedelta(days=10))
    days = sched.index[sched.index > date]
    return days[0] if len(days) > 0 else date + pd.Timedelta(days=1)

# ── UI helpers ─────────────────────────────────────────────────────────────────

def pill(label, val, lo, hi):
    cls = "pill-g" if val < lo else ("pill-r" if val > hi else "pill-a")
    return f'<span class="{cls}">{label}: {val:.2f}</span> '

def render_hero(signal: dict, master: pd.DataFrame):
    if not signal or "top_pick" not in signal:
        st.info("Signal not available yet — run predict.py first.")
        return

    tickers = cfg.OPTION_A_ETFS if signal.get("option") == "A" else cfg.OPTION_B_ETFS
    forecasts = signal.get("forecasts", {})

    ranked = sorted(
        [(t, forecasts[t]["mu"], forecasts[t].get("confidence", 0.5))
         for t in tickers if t in forecasts],
        key=lambda x: x[1], reverse=True,
    )

    t1 = ranked[0] if ranked else (signal["top_pick"], signal.get("top_mu", 0), signal.get("top_confidence", 0))
    t2 = ranked[1] if len(ranked) > 1 else None
    t3 = ranked[2] if len(ranked) > 2 else None

    next_day = str(next_trading_day(master.index[-1]).date()) if not master.empty else signal.get("signal_date", "—")

    gen = signal.get("generated_at", "")
    try:
        gen = datetime.fromisoformat(gen).strftime("%Y-%m-%d %H:%M UTC")
    except:
        pass

    runner = ""
    if t2:
        runner += f"2nd: **{t2[0]}** μ={t2[1]:.4f} "
    if t3:
        runner += f"3rd: **{t3[0]}** μ={t3[1]:.4f}"

    rc = signal.get("regime_context", {})
    pills = ""
    if rc.get("VIX"): pills += pill("VIX", rc["VIX"], 15, 25)
    if rc.get("T10Y2Y"): pills += pill("T10Y2Y", rc["T10Y2Y"], -0.5, 0.5)
    if rc.get("HY_SPREAD"): pills += pill("HY Spr", rc["HY_SPREAD"], 300, 500)

    st.markdown(f"""
    <div style="background: #1e1e2e; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h2 style="margin:0">{t1[0]} <span style="color:#3a5bd9">μ = {t1[1]:.4f}</span></h2>
        <p style="margin:8px 0">Signal for <strong>{next_day}</strong> · Generated {gen} · Confidence {t1[2]:.1%}</p>
        <p>{runner}</p>
        <div style="margin-top:12px">{pills}</div>
    </div>
    """, unsafe_allow_html=True)

def render_forecast_chart(signal: dict):
    if not signal or "forecasts" not in signal:
        return

    forecasts = signal["forecasts"]
    tickers = cfg.OPTION_A_ETFS if signal.get("option") == "A" else cfg.OPTION_B_ETFS
    tickers = [t for t in tickers if t in forecasts]

    mus = [forecasts[t]["mu"] for t in tickers]
    sigmas = [forecasts[t].get("sigma", 0.01) for t in tickers]
    colors = ["#3a5bd9" if t == signal.get("top_pick") else "#9ca3af" for t in tickers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mus,
        y=tickers,
        orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=sigmas, visible=True, color="#d1d5db"),
        hovertemplate="<b>%{y}</b><br>μ = %{x:.4f}<br>σ = %{error_x.array:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title="Next-Day Return Forecasts (μ ± σ)",
        xaxis_title="Expected Return",
        height=500,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Main UI ────────────────────────────────────────────────────────────────────

tabs = st.tabs(["Fixed Income & Alternatives (A)", "Equity Sectors (B)"])

signals = load_signals()
master = load_master()

with tabs[0]:
    st.subheader("Option A — Fixed Income & Alternatives")
    signal_a = signals.get("A", {})
    render_hero(signal_a, master)
    render_forecast_chart(signal_a)

with tabs[1]:
    st.subheader("Option B — Equity Sectors")
    signal_b = signals.get("B", {})
    render_hero(signal_b, master)
    render_forecast_chart(signal_b)

st.caption("NSDE Engine • Variance-Preserving Stochastic Differential Equations • "
           f"Input: {cfg.HF_DATASET_INPUT} • Output: {cfg.HF_DATASET_OUTPUT}")
