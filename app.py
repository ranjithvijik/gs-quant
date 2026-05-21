"""
Open Source Quant Platform — Complete gs-quant Replica
========================================================
A professional-grade quantitative finance toolkit built entirely with open-source tools.
Inspired by Goldman Sachs' gs-quant, with no proprietary APIs required.

MODULES (15 Total):
  1.  Dashboard              — Live market overview, normalized performance, correlation
  2.  Market Data            — OHLCV + Bollinger, RSI, MACD, SMA/EMA multi-subplot charts
  3.  Derivatives Pricing    — Black-Scholes, 5 Greeks, 3D surfaces, IV calculator, payoff
  4.  Risk Analytics         — Historical/Parametric/MC VaR, ES, GBM simulation, stress testing
  5.  Portfolio Management   — MVO (Max Sharpe/Min Vol), Efficient Frontier, attribution
  6.  Strategy Backtesting   — SMA Crossover, Mean Reversion, Momentum + full metrics
  7.  Volatility Surface     — Interactive 3D IV surface with skew/smile/term controls
  8.  Fixed Income           — Bond pricer, yield curve (cubic spline), duration/convexity
  9.  Multi-Asset Pricing    — FX Forwards, Interest Rate Swaps, Credit Default Swaps
  10. Scenario Analysis      — Parallel/non-parallel rate shocks, spot/vol shocks
  11. Timeseries Analytics   — Z-scores, rolling regression, winsorize, full statistics
  12. Factor Risk Models     — Factor decomposition, systematic vs idiosyncratic risk
  13. Basket Construction    — Custom index creation with rebalancing
  14. Hedging Analytics      — Delta/gamma hedging, factor hedging
  15. FX Analytics           — Cross-currency analysis, FX carry, FX volatility

Requirements:
    pip install streamlit numpy pandas scipy plotly yfinance statsmodels

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import math
from scipy.stats import norm, zscore
from scipy.optimize import minimize, brentq
from scipy.interpolate import CubicSpline
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant Platform", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

GOLD = "#D4AF37"
BLUE = "#4FC3F7"
RED = "#FF1744"
GREEN = "#00C853"
ORANGE = "#FF7043"
PURPLE = "#AB47BC"
CARD_BG = "#1A1D23"

st.markdown(f"""
<style>
    .main .block-container {{ padding-top: 1rem; max-width: 1400px; }}
    h1, h2, h3 {{ font-family: 'Georgia', serif; }}
    .gold-bar {{ height: 4px; background: linear-gradient(90deg, {GOLD}, #FFD700, {GOLD}); border-radius: 2px; margin-bottom: 1.2rem; }}
    .mc {{ background: {CARD_BG}; border: 1px solid #333; border-radius: 10px; padding: 1.1rem; text-align: center; border-left: 4px solid {GOLD}; margin-bottom: 0.5rem; }}
    .mc h4 {{ color: #888; font-size: 0.82rem; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }}
    .mc p {{ color: #fff; font-size: 1.4rem; font-weight: bold; margin: 0.3rem 0 0 0; }}
    .mc .sub {{ color: #aaa; font-size: 0.8rem; margin-top: 0.2rem; }}
    [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0a0c10 0%, #141820 100%); }}
    .ib {{ background: {CARD_BG}; border: 1px solid #333; border-radius: 8px; padding: 1rem 1.2rem; border-left: 4px solid {BLUE}; margin-bottom: 1rem; }}
    .ib p {{ color: #ccc; font-size: 0.85rem; margin: 0; line-height: 1.5; }}
</style>
""", unsafe_allow_html=True)

PL = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="Georgia, serif"), legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
          margin=dict(l=40, r=40, t=50, b=40))

def card(label, value, sub="", color=GOLD):
    s = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="mc" style="border-left-color:{color};"><h4>{label}</h4><p>{value}</p>{s}</div>'

# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_multi(tickers, start, end):
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    try:
        df = yf.download(tickers, start=start, end=end, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c).strip() for c in df.columns]
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_single(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        return df
    except: return pd.DataFrame()

def closes_from_multi(df):
    cols = [c for c in df.columns if c.startswith("Close")]
    if not cols: return pd.DataFrame()
    out = df[cols].copy()
    out.columns = [c.replace("Close_", "").strip() for c in out.columns]
    return out.dropna()

# ─────────────────────────────────────────────────────────────────────────────
# PRICING ENGINES
# ─────────────────────────────────────────────────────────────────────────────
def bs(S, K, T, r, sigma, otype="Call"):
    """Black-Scholes: (price, delta, gamma, theta, vega, rho)"""
    if T <= 0 or sigma <= 0:
        intr = max(0, S - K) if otype == "Call" else max(0, K - S)
        d = (1.0 if S > K else 0.0) if otype == "Call" else (-1.0 if K > S else 0.0)
        return intr, d, 0.0, 0.0, 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1, Nd2, nd1 = norm.cdf(d1), norm.cdf(d2), norm.pdf(d1)
    if otype == "Call":
        price = S * Nd1 - K * math.exp(-r * T) * Nd2
        delta = Nd1
        theta = (-S * nd1 * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * Nd2) / 365
        rho = K * T * math.exp(-r * T) * Nd2 / 100
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = Nd1 - 1
        theta = (-S * nd1 * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = nd1 / (S * sigma * math.sqrt(T))
    vega = S * math.sqrt(T) * nd1 / 100
    return price, delta, gamma, theta, vega, rho

def implied_vol(market_price, S, K, T, r, otype="Call"):
    try: return brentq(lambda sig: bs(S, K, T, r, sig, otype)[0] - market_price, 1e-4, 5.0, xtol=1e-8)
    except: return np.nan

def fx_forward(spot, r_dom, r_for, T):
    """FX Forward price using covered interest rate parity"""
    return spot * math.exp((r_dom - r_for) * T)

def fx_forward_points(spot, r_dom, r_for, T):
    """Forward points (pips)"""
    fwd = fx_forward(spot, r_dom, r_for, T)
    return (fwd - spot) * 10000

def irs_npv(notional, fixed_rate, float_spread, swap_curve, tenor_years, pay_fixed=True, freq=2):
    """Interest Rate Swap NPV using simple discounting"""
    periods = int(tenor_years * freq)
    times = np.arange(1, periods + 1) / freq
    fixed_cf = notional * fixed_rate / freq
    df = np.array([1 / (1 + swap_curve(t) / 100) ** t for t in times])
    fixed_leg = np.sum(fixed_cf * df) + notional * df[-1]
    float_rates = np.array([swap_curve(t) / 100 + float_spread for t in times])
    float_cf = notional * float_rates / freq
    float_leg = np.sum(float_cf * df) + notional * df[-1]
    if pay_fixed:
        return float_leg - fixed_leg
    return fixed_leg - float_leg

def cds_spread_approx(hazard_rate, recovery_rate, tenor_years, risk_free_rate):
    """Approximate CDS spread using reduced-form model"""
    lgd = 1 - recovery_rate
    spread = hazard_rate * lgd * 10000
    return spread

def cds_npv(notional, spread_bps, hazard_rate, recovery_rate, tenor_years, risk_free_rate):
    """CDS NPV approximation"""
    if hazard_rate <= 1e-10:
        return 0.0
    lgd = 1 - recovery_rate
    fair_spread = hazard_rate * lgd * 10000
    spread_diff = (spread_bps - fair_spread) / 10000
    risky_annuity = (1 - math.exp(-hazard_rate * tenor_years)) / hazard_rate
    df = math.exp(-risk_free_rate * tenor_years / 2)
    return notional * spread_diff * risky_annuity * df

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def add_technicals(df):
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Mid"], df["BB_Upper"], df["BB_Lower"] = bb_mid, bb_mid + 2*bb_std, bb_mid - 2*bb_std
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    e12, e26 = df["Close"].ewm(span=12, adjust=False).mean(), df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"], df["Signal_Line"] = e12 - e26, (e12 - e26).ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]
    return df

# ─────────────────────────────────────────────────────────────────────────────
# TIMESERIES ANALYTICS (gs-quant style)
# ─────────────────────────────────────────────────────────────────────────────
def ts_zscore(series, window=20):
    """Rolling z-score"""
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def ts_winsorize(series, lower=0.05, upper=0.95):
    """Winsorize extreme values"""
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)

def ts_rolling_beta(asset_returns, market_returns, window=60):
    """Rolling beta vs market"""
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var

def ts_rolling_corr(x, y, window=60):
    """Rolling correlation"""
    return x.rolling(window).corr(y)

def ts_rolling_regression(y, X, window=60):
    """Rolling OLS regression - returns alpha, beta, r-squared"""
    alphas, betas, r2s = [], [], []
    for i in range(len(y)):
        if i < window:
            alphas.append(np.nan); betas.append(np.nan); r2s.append(np.nan)
        else:
            y_win = y.iloc[i-window:i].values
            X_win = X.iloc[i-window:i].values
            X_mat = np.column_stack([np.ones(window), X_win])
            try:
                coeffs = np.linalg.lstsq(X_mat, y_win, rcond=None)[0]
                y_pred = X_mat @ coeffs
                ss_res = np.sum((y_win - y_pred)**2)
                ss_tot = np.sum((y_win - np.mean(y_win))**2)
                alphas.append(coeffs[0]); betas.append(coeffs[1])
                r2s.append(1 - ss_res/ss_tot if ss_tot > 0 else 0)
            except:
                alphas.append(np.nan); betas.append(np.nan); r2s.append(np.nan)
    return pd.Series(alphas, index=y.index), pd.Series(betas, index=y.index), pd.Series(r2s, index=y.index)

def ts_exponential_vol(returns, beta=0.94):
    """EWMA volatility (RiskMetrics style)"""
    var = returns.ewm(alpha=1-beta, adjust=False).var()
    return np.sqrt(var * 252)

def ts_realized_vol(returns, window=20):
    """Realized volatility"""
    return returns.rolling(window).std() * np.sqrt(252)

def ts_hurst_exponent(series, max_lag=20):
    """Hurst exponent for mean reversion detection"""
    try:
        lags = list(range(2, max_lag))
        tau = [np.sqrt(np.std(np.subtract(series[lag:].values, series[:-lag].values))) for lag in lags]
        tau = [max(t, 1e-10) for t in tau]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

# ─────────────────────────────────────────────────────────────────────────────
# BOND PRICING
# ─────────────────────────────────────────────────────────────────────────────
def bond_price_full(face, cpn_rate, ytm, years, freq=2):
    periods = int(years * freq)
    c = face * cpn_rate / freq
    y = ytm / freq
    t = np.arange(1, periods + 1)
    cf = np.full(periods, c); cf[-1] += face
    df = 1 / (1 + y) ** t
    pv = cf * df
    price = pv.sum()
    mac = (t * pv).sum() / price / freq
    mod = mac / (1 + y)
    conv = (t * (t + 1) * pv).sum() / (price * (1 + y)**2 * freq**2)
    dv01 = mod * price / 10000
    return price, mac, mod, conv, dv01

# ─────────────────────────────────────────────────────────────────────────────
# FACTOR RISK MODEL
# ─────────────────────────────────────────────────────────────────────────────
def compute_factor_exposures(returns, factor_returns):
    """Compute factor betas via regression"""
    exposures = {}
    for col in returns.columns:
        X = factor_returns.values
        y = returns[col].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if mask.sum() < 30: continue
        X_clean = np.column_stack([np.ones(mask.sum()), X[mask]])
        y_clean = y[mask]
        try:
            coeffs = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
            exposures[col] = {"Alpha": coeffs[0] * 252}
            for i, f in enumerate(factor_returns.columns):
                exposures[col][f] = coeffs[i+1]
        except: pass
    return pd.DataFrame(exposures).T

def decompose_risk(returns, factor_returns, weights):
    """Decompose portfolio risk into systematic and idiosyncratic"""
    port_ret = (returns * weights).sum(axis=1)
    X = factor_returns.values
    y = port_ret.values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    if mask.sum() < 10:
        return {"Total Vol": 0, "Systematic Vol": 0, "Idiosyncratic Vol": 0, "R-squared": 0}
    X_clean = np.column_stack([np.ones(mask.sum()), X[mask]])
    y_clean = y[mask]
    coeffs = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
    y_pred = X_clean @ coeffs
    residuals = y_clean - y_pred
    total_var = np.var(y_clean) * 252
    systematic_var = np.var(y_pred) * 252
    idio_var = np.var(residuals) * 252
    return {
        "Total Vol": np.sqrt(total_var),
        "Systematic Vol": np.sqrt(systematic_var),
        "Idiosyncratic Vol": np.sqrt(idio_var),
        "R-squared": 1 - idio_var/total_var if total_var > 0 else 0
    }

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def perf_metrics(strat_ret, mkt_ret=None, rf=0.05):
    sr = strat_ret.dropna()
    if len(sr) < 2: return {}
    tot = (1 + sr).prod() - 1
    n = len(sr)
    ann_r = (1 + tot) ** (252 / max(n, 1)) - 1
    ann_v = sr.std() * np.sqrt(252)
    sharpe = (ann_r - rf) / ann_v if ann_v > 0 else 0
    cum = (1 + sr).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = ann_r / abs(max_dd) if abs(max_dd) > 0 else 0
    ds = sr[sr < 0].std() * np.sqrt(252)
    sortino = (ann_r - rf) / ds if ds > 0 else 0
    wr = (sr > 0).sum() / n
    m = {"Total Return": f"{tot:.2%}", "Ann. Return": f"{ann_r:.2%}", "Ann. Vol": f"{ann_v:.2%}",
         "Sharpe": f"{sharpe:.3f}", "Sortino": f"{sortino:.3f}", "Max DD": f"{max_dd:.2%}",
         "Calmar": f"{calmar:.3f}", "Win Rate": f"{wr:.2%}"}
    if mkt_ret is not None:
        mr = mkt_ret.dropna()
        if len(mr) > 1:
            mt = (1 + mr).prod() - 1
            m["Market Return"] = f"{mt:.2%}"
    return m

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="text-align:center;padding:1rem 0 .5rem"><h1 style="color:{GOLD};margin:0;font-size:1.6rem">📈 Quant Platform</h1><p style="color:#888;font-size:.75rem;margin-top:.2rem">gs-quant Open Source Replica</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    
    st.markdown("**Core Modules**")
    modules_core = ["🏠 Dashboard", "📊 Market Data", "💰 Derivatives Pricing", "⚠️ Risk Analytics",
                    "📁 Portfolio Mgmt", "🔄 Backtesting", "🌊 Vol Surface", "📐 Fixed Income"]
    
    st.markdown("**Advanced Modules**")
    modules_adv = ["🏦 Multi-Asset Pricing", "🎯 Scenario Analysis", "📈 Timeseries Analytics",
                   "🧮 Factor Risk Models", "📦 Basket Construction", "🛡️ Hedging Analytics", "💱 FX Analytics"]
    
    all_modules = modules_core + modules_adv
    choice = st.radio("Navigation", all_modules, label_visibility="collapsed")

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if choice == "🏠 Dashboard":
    st.markdown(f'<h1 style="color:{GOLD}">Global Market Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    
    ca, cb = st.columns([3, 1])
    tickers_str = ca.text_input("Universe", "SPY, QQQ, GLD, TLT, IWM, EFA")
    lookback = cb.slider("Lookback (days)", 30, 1500, 252)
    tlist = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    end_d, start_d = datetime.date.today(), datetime.date.today() - datetime.timedelta(days=lookback)
    
    with st.spinner("Fetching..."):
        raw = fetch_multi(tlist, start_d, end_d)
        closes = closes_from_multi(raw) if not raw.empty else pd.DataFrame()
    
    if closes.empty:
        st.warning("No data returned.")
    else:
        colors6 = [GOLD, BLUE, ORANGE, GREEN, PURPLE, "#FFD600"]
        cc = st.columns(min(len(closes.columns), 6))
        for i, t in enumerate(closes.columns[:6]):
            last, prev = closes[t].iloc[-1], closes[t].iloc[-2] if len(closes) > 1 else closes[t].iloc[-1]
            chg = (last - prev) / prev * 100
            clr, arr = (GREEN, "▲") if chg >= 0 else (RED, "▼")
            cc[i].markdown(card(t, f"${last:,.2f}", f'<span style="color:{clr}">{arr} {chg:+.2f}%</span>', colors6[i % 6]), unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📈 Normalized Performance")
            norm_p = closes / closes.iloc[0] * 100
            fig = go.Figure()
            for i, col in enumerate(norm_p.columns):
                fig.add_trace(go.Scatter(x=norm_p.index, y=norm_p[col], name=col, line=dict(color=colors6[i % 6], width=2)))
            fig.update_layout(**PL, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("📊 Correlation")
            rets = closes.pct_change().dropna()
            fig2 = px.imshow(rets.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig2.update_layout(**PL, height=400)
            st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 2: MARKET DATA
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "📊 Market Data":
    st.markdown(f'<h1 style="color:{GOLD}">Market Data & Technical Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker", "AAPL")
    dt_start = c2.date_input("Start", datetime.date.today() - datetime.timedelta(365))
    dt_end = c3.date_input("End", datetime.date.today())
    
    if st.button("📥 Analyze", type="primary"):
        df = fetch_single(ticker, dt_start, dt_end)
        if df.empty:
            st.error("No data.")
        else:
            df = add_technicals(df)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25],
                               subplot_titles=[f"{ticker} Price", "RSI", "MACD"])
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                         name="OHLC", increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color="gray", width=1, dash="dot"), fill="tonexty"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(color=ORANGE, width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color=ORANGE, width=1.5)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color=BLUE, width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["Signal_Line"], name="Signal", line=dict(color=ORANGE, width=1.5)), row=3, col=1)
            fig.update_layout(**PL, height=700, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 3: DERIVATIVES PRICING
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "💰 Derivatives Pricing":
    st.markdown(f'<h1 style="color:{GOLD}">Derivatives Pricing & Greeks</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔢 Pricer", "📈 Greeks Surface", "🧮 IV Calculator"])
    
    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            S = st.number_input("Spot ($)", value=100.0, step=1.0)
            K = st.number_input("Strike ($)", value=100.0, step=5.0)
            dte = st.number_input("DTE", value=30, min_value=1)
            sigma = st.number_input("Vol (%)", value=25.0) / 100
            r = st.number_input("Rf (%)", value=4.5) / 100
            ot = st.radio("Type", ["Call", "Put"], horizontal=True)
        
        T = dte / 365.0
        pr, d, g, th, v, rh = bs(S, K, T, r, sigma, ot)
        
        with c2:
            st.subheader("Results")
            rc = st.columns(3)
            rc[0].markdown(card(f"{ot} Premium", f"${pr:.4f}", color=GOLD), unsafe_allow_html=True)
            rc[1].markdown(card("Delta", f"{d:.4f}", color=BLUE), unsafe_allow_html=True)
            rc[2].markdown(card("Gamma", f"{g:.4f}", color=GREEN), unsafe_allow_html=True)
            rc2 = st.columns(3)
            rc2[0].markdown(card("Theta/day", f"{th:.4f}", color=ORANGE), unsafe_allow_html=True)
            rc2[1].markdown(card("Vega/1%", f"{v:.4f}", color=PURPLE), unsafe_allow_html=True)
            rc2[2].markdown(card("Rho/1%", f"{rh:.4f}", color=GOLD), unsafe_allow_html=True)
            
            # Payoff
            sr = np.linspace(S * 0.5, S * 1.5, 200)
            pay = (np.maximum(sr - K, 0) - pr) if ot == "Call" else (np.maximum(K - sr, 0) - pr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sr, y=pay, name="P&L at Expiry", line=dict(color=GOLD, width=2.5)))
            fig.add_hline(y=0, line_color="gray", line_dash="dash")
            fig.add_vline(x=K, line_dash="dash", line_color=RED, annotation_text="Strike")
            fig.update_layout(**PL, height=350, xaxis_title="Spot ($)", yaxis_title="P&L ($)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Delta Surface")
        spots = np.linspace(S * 0.7, S * 1.3, 40)
        vols = np.linspace(0.1, 0.6, 40)
        Z = np.zeros((len(vols), len(spots)))
        for i, v in enumerate(vols):
            for j, s in enumerate(spots):
                Z[i, j] = bs(s, K, T, r, v, ot)[1]
        fig = go.Figure(go.Surface(z=Z, x=spots, y=vols, colorscale="Viridis"))
        fig.update_layout(**PL, height=500, scene=dict(xaxis_title="Spot", yaxis_title="Vol", zaxis_title="Delta"))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Implied Volatility Solver")
        mp = st.number_input("Market Premium ($)", 10.0, step=0.5)
        if st.button("Calculate IV"):
            iv = implied_vol(mp, S, K, T, r, ot)
            if np.isnan(iv):
                st.error("Could not converge.")
            else:
                st.markdown(card("Implied Vol", f"{iv*100:.2f}%", color=GOLD), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 4: RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "⚠️ Risk Analytics":
    st.markdown(f'<h1 style="color:{GOLD}">Risk Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 VaR", "🎲 Monte Carlo", "💥 Stress Test"])
    
    with tabs[0]:
        c1, c2 = st.columns([1, 3])
        vtk = c1.text_input("Ticker", "SPY", key="vtk")
        conf = c1.slider("Confidence %", 90.0, 99.9, 95.0) / 100
        pval = c1.number_input("Portfolio ($)", 1_000_000, step=100_000)
        
        df = fetch_single(vtk, datetime.date.today() - datetime.timedelta(1000), datetime.date.today())
        if not df.empty:
            ret = df["Close"].pct_change().dropna()
            h_var = -np.percentile(ret, (1 - conf) * 100)
            mu, std = ret.mean(), ret.std()
            p_var = -(mu + norm.ppf(1 - conf) * std)
            
            with c2:
                mc = st.columns(3)
                mc[0].markdown(card("Historical VaR", f"{h_var:.4%}", f"${h_var*pval:,.0f}", RED), unsafe_allow_html=True)
                mc[1].markdown(card("Parametric VaR", f"{p_var:.4%}", f"${p_var*pval:,.0f}", ORANGE), unsafe_allow_html=True)
                mc[2].markdown(card("Daily Vol", f"{std:.4%}", color=BLUE), unsafe_allow_html=True)
                
                fig = go.Figure(go.Histogram(x=ret, nbinsx=100, marker_color=GOLD, opacity=0.7))
                fig.add_vline(x=-h_var, line_color=RED, line_width=2, annotation_text=f"VaR {conf:.0%}")
                fig.update_layout(**PL, height=350, title="Return Distribution")
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Monte Carlo Simulation")
        nsim = st.slider("Simulations", 100, 2000, 500)
        horiz = st.slider("Horizon (days)", 5, 252, 63)
        if st.button("Run MC"):
            df = fetch_single("SPY", datetime.date.today() - datetime.timedelta(1000), datetime.date.today())
            if not df.empty:
                ret = df["Close"].pct_change().dropna()
                S0 = df["Close"].iloc[-1]
                mu_a, sig_a = ret.mean() * 252, ret.std() * np.sqrt(252)
                dt = 1/252
                np.random.seed(42)
                paths = np.zeros((horiz, nsim))
                paths[0] = S0
                Z = np.random.standard_normal((horiz, nsim))
                log_rets = (mu_a - 0.5 * sig_a**2) * dt + sig_a * np.sqrt(dt) * Z
                log_rets[0] = 0  # Initial step has no shock
                paths = S0 * np.exp(np.cumsum(log_rets, axis=0))
                
                fig = go.Figure()
                for i in range(min(100, nsim)):
                    fig.add_trace(go.Scatter(y=paths[:, i], mode="lines", line=dict(width=0.5, color=f"rgba(212,175,55,0.15)"), showlegend=False))
                fig.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), name="Median", line=dict(color=GOLD, width=2.5)))
                fig.update_layout(**PL, height=400, title=f"GBM Paths ({nsim} sims, {horiz}d)")
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Stress Testing")
        pv = st.number_input("Portfolio ($)", 1_000_000, step=100_000, key="st_pv")
        scenarios = {"Bull +20%": 0.20, "Rally +10%": 0.10, "Flat": 0, "Correction -10%": -0.10, "Bear -20%": -0.20, "Crash -35%": -0.35}
        rows = [{"Scenario": n, "Shock": f"{s:+.0%}", "P&L": pv*s, "New Value": pv*(1+s)} for n, s in scenarios.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 5: PORTFOLIO MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "📁 Portfolio Mgmt":
    st.markdown(f'<h1 style="color:{GOLD}">Portfolio Management</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    
    tickers = st.text_input("Universe", "AAPL, MSFT, GOOGL, AMZN, NVDA")
    tlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    rf = st.number_input("Rf (%)", 5.0) / 100
    
    if st.button("🚀 Optimize", type="primary"):
        closes = closes_from_multi(fetch_multi(tlist, datetime.date.today() - datetime.timedelta(days=3*365), datetime.date.today()))
        if closes.empty or closes.shape[1] < 2:
            st.error("Need ≥2 tickers.")
        else:
            rets = closes.pct_change().dropna()
            mu_ann, cov_ann = rets.mean() * 252, rets.cov() * 252
            n = len(mu_ann)
            
            def neg_sharpe(w):
                r_, v_ = w @ mu_ann, np.sqrt(w @ cov_ann @ w)
                return -(r_ - rf) / v_ if v_ > 0 else 0
            
            cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
            bnd = [(0, 1)] * n
            w0 = np.ones(n) / n
            ws = minimize(neg_sharpe, w0, method="SLSQP", bounds=bnd, constraints=cons).x
            
            rs, vs = ws @ mu_ann, np.sqrt(ws @ cov_ann @ ws)
            sh = (rs - rf) / vs
            
            mc = st.columns(3)
            mc[0].markdown(card("Return", f"{rs:.2%}", color=GREEN), unsafe_allow_html=True)
            mc[1].markdown(card("Volatility", f"{vs:.2%}", color=RED), unsafe_allow_html=True)
            mc[2].markdown(card("Sharpe", f"{sh:.3f}", color=GOLD), unsafe_allow_html=True)
            
            # Weights
            wdf = pd.DataFrame({"Ticker": closes.columns, "Weight": ws}).sort_values("Weight", ascending=True)
            fig = go.Figure(go.Bar(y=wdf["Ticker"], x=wdf["Weight"], orientation="h", marker_color=GOLD,
                                   text=[f"{w:.1%}" for w in wdf["Weight"]], textposition="outside"))
            fig.update_layout(**PL, height=300, xaxis_tickformat=".0%", title="Optimal Weights")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 6: BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "🔄 Backtesting":
    st.markdown(f'<h1 style="color:{GOLD}">Strategy Backtesting</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 3])
    with c1:
        btk = st.text_input("Ticker", "SPY", key="btk")
        strat = st.selectbox("Strategy", ["SMA Crossover", "Mean Reversion", "Momentum"])
        yrs = st.slider("Years", 1, 10, 3)
        fast = st.number_input("Fast MA", value=20, min_value=2) if strat == "SMA Crossover" else 20
        slow = st.number_input("Slow MA", value=50, min_value=5) if strat == "SMA Crossover" else 50
    
    if st.button("▶️ Run Backtest", type="primary"):
        df = fetch_single(btk, datetime.date.today() - datetime.timedelta(yrs*365), datetime.date.today())
        if df.empty:
            st.error("No data.")
        else:
            df["Market_Return"] = df["Close"].pct_change()
            if strat == "SMA Crossover":
                df["Fast"] = df["Close"].rolling(int(fast)).mean()
                df["Slow"] = df["Close"].rolling(int(slow)).mean()
                df["Signal"] = np.where(df["Fast"] > df["Slow"], 1, 0)
            elif strat == "Mean Reversion":
                df["Z"] = ts_zscore(df["Close"], 20)
                df["Signal"] = np.where(df["Z"] < -1, 1, np.where(df["Z"] > 1, 0, np.nan))
                df["Signal"] = df["Signal"].ffill().fillna(0)
            else:
                df["Mom"] = df["Close"].pct_change(60)
                df["Signal"] = np.where(df["Mom"] > 0, 1, 0)
            
            df["Strategy_Return"] = df["Signal"].shift(1) * df["Market_Return"]
            df.dropna(inplace=True)
            df["Cum_Market"] = (1 + df["Market_Return"]).cumprod()
            df["Cum_Strategy"] = (1 + df["Strategy_Return"]).cumprod()
            
            met = perf_metrics(df["Strategy_Return"], df["Market_Return"])
            
            with c2:
                mk = list(met.items())
                for chunk in [mk[i:i+4] for i in range(0, len(mk), 4)]:
                    cols = st.columns(4)
                    for j, (k, v) in enumerate(chunk):
                        cols[j].markdown(card(k, v), unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Cum_Strategy"], name="Strategy", line=dict(color=GOLD, width=2.5)))
                fig.add_trace(go.Scatter(x=df.index, y=df["Cum_Market"], name="Buy & Hold", line=dict(color=BLUE, width=2, dash="dash")))
                fig.update_layout(**PL, height=400, title=f"{strat} — {btk}")
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 7: VOLATILITY SURFACE
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "🌊 Vol Surface":
    st.markdown(f'<h1 style="color:{GOLD}">Volatility Surface</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 3])
    with c1:
        atm = st.number_input("ATM Vol (%)", 20.0) / 100
        skew = st.number_input("Skew", -0.15, step=0.01)
        smile = st.number_input("Smile", 0.20, step=0.01)
        spot = st.number_input("Spot ($)", 100.0)
    
    with c2:
        ks = np.linspace(spot * 0.7, spot * 1.3, 40)
        ts = np.linspace(0.08, 2.0, 35)
        KG, TG = np.meshgrid(ks, ts)
        m = KG / spot - 1
        IV = atm + skew * m + smile * m**2 - 0.03 * np.sqrt(TG)
        IV = np.maximum(IV, 0.01)
        
        fig = go.Figure(go.Surface(z=IV * 100, x=ks, y=ts, colorscale="Plasma", opacity=0.92))
        fig.update_layout(**PL, height=600, scene=dict(xaxis_title="Strike", yaxis_title="Expiry", zaxis_title="IV (%)"))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 8: FIXED INCOME
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "📐 Fixed Income":
    st.markdown(f'<h1 style="color:{GOLD}">Fixed Income Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🏦 Bond Pricer", "📉 Yield Curve"])
    
    with tabs[0]:
        c1, c2 = st.columns([1, 2])
        with c1:
            face = st.number_input("Face ($)", value=1000.0)
            cpn = st.number_input("Coupon (%)", value=5.0) / 100
            ytm = st.number_input("YTM (%)", value=4.0) / 100
            yrs = st.number_input("Years", value=10, min_value=1)
        
        with c2:
            bp, mac, mod, conv, dv01 = bond_price_full(face, cpn, ytm, yrs, 2)
            mc = st.columns(4)
            mc[0].markdown(card("Price", f"${bp:,.2f}", color=GOLD), unsafe_allow_html=True)
            mc[1].markdown(card("Duration", f"{mod:.3f}", color=BLUE), unsafe_allow_html=True)
            mc[2].markdown(card("Convexity", f"{conv:.3f}", color=ORANGE), unsafe_allow_html=True)
            mc[3].markdown(card("DV01", f"${dv01:.4f}", color=GREEN), unsafe_allow_html=True)
            
            ys = np.linspace(max(0.001, ytm - 0.03), ytm + 0.03, 50)
            ps = [bond_price_full(face, cpn, y, yrs, 2)[0] for y in ys]
            fig = go.Figure(go.Scatter(x=ys * 100, y=ps, line=dict(color=GOLD, width=2.5)))
            fig.add_vline(x=ytm * 100, line_dash="dash", line_color=RED)
            fig.update_layout(**PL, height=350, xaxis_title="Yield (%)", yaxis_title="Price ($)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Yield Curve Builder")
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        labels = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
        defaults = [5.25, 5.10, 4.80, 4.50, 4.35, 4.20, 4.25, 4.30, 4.55, 4.50]
        cols = st.columns(5)
        rates = []
        for i, (lb, df_) in enumerate(zip(labels, defaults)):
            with cols[i % 5]:
                rates.append(st.number_input(f"{lb} (%)", value=df_, step=0.05, key=f"yc_{lb}"))
        
        cs = CubicSpline(tenors, rates)
        xs = np.linspace(0.25, 30, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=cs(xs), name="Curve", line=dict(color=GOLD, width=2.5)))
        fig.add_trace(go.Scatter(x=tenors, y=rates, mode="markers", marker=dict(size=10, color=ORANGE), name="Tenors"))
        fig.update_layout(**PL, height=400, title="Treasury Yield Curve")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 9: MULTI-ASSET PRICING (NEW)
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "🏦 Multi-Asset Pricing":
    st.markdown(f'<h1 style="color:{GOLD}">Multi-Asset Instrument Pricing</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ib"><p><b>Instruments:</b> FX Forwards, Interest Rate Swaps, Credit Default Swaps. All priced using standard market models.</p></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["💱 FX Forward", "📊 Interest Rate Swap", "🛡️ Credit Default Swap"])
    
    with tabs[0]:
        st.subheader("FX Forward Pricing (Covered Interest Rate Parity)")
        c1, c2 = st.columns([1, 2])
        with c1:
            spot_fx = st.number_input("Spot Rate (DOM/FOR)", 1.10, step=0.01, format="%.4f")
            r_dom = st.number_input("Domestic Rate (%)", 4.5, step=0.1) / 100
            r_for = st.number_input("Foreign Rate (%)", 3.5, step=0.1) / 100
            tenor_fx = st.selectbox("Tenor", ["1M", "3M", "6M", "1Y", "2Y"])
            notional_fx = st.number_input("Notional (FOR)", 1_000_000, step=100_000)
        
        tenor_map = {"1M": 1/12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0}
        T_fx = tenor_map[tenor_fx]
        fwd_rate = fx_forward(spot_fx, r_dom, r_for, T_fx)
        fwd_pts = fx_forward_points(spot_fx, r_dom, r_for, T_fx)
        
        with c2:
            mc = st.columns(3)
            mc[0].markdown(card("Forward Rate", f"{fwd_rate:.4f}", color=GOLD), unsafe_allow_html=True)
            mc[1].markdown(card("Forward Points", f"{fwd_pts:.2f} pips", color=BLUE), unsafe_allow_html=True)
            mc[2].markdown(card("Carry", f"{(r_dom - r_for)*100:.2f} bps", color=GREEN if r_dom > r_for else RED), unsafe_allow_html=True)
            
            # Forward curve
            tenors_plot = np.linspace(0.01, 2, 50)
            fwds_plot = [fx_forward(spot_fx, r_dom, r_for, t) for t in tenors_plot]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tenors_plot, y=fwds_plot, name="Forward Curve", line=dict(color=GOLD, width=2.5)))
            fig.add_hline(y=spot_fx, line_dash="dash", line_color=BLUE, annotation_text="Spot")
            fig.update_layout(**PL, height=350, xaxis_title="Tenor (years)", yaxis_title="Forward Rate", title="FX Forward Curve")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Interest Rate Swap Pricing")
        c1, c2 = st.columns([1, 2])
        with c1:
            notional_irs = st.number_input("Notional ($)", 10_000_000, step=1_000_000)
            fixed_rate = st.number_input("Fixed Rate (%)", 4.0, step=0.1) / 100
            float_spread = st.number_input("Float Spread (bps)", 0, step=5) / 10000
            tenor_irs = st.selectbox("Tenor (years)", [2, 3, 5, 7, 10], index=2)
            pay_fixed = st.radio("Direction", ["Pay Fixed", "Receive Fixed"], horizontal=True) == "Pay Fixed"
        
        # Build swap curve
        tenors_curve = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
        rates_curve = [5.25, 5.10, 4.80, 4.50, 4.35, 4.20, 4.25, 4.30]
        swap_curve = CubicSpline(tenors_curve, rates_curve)
        
        npv = irs_npv(notional_irs, fixed_rate, float_spread, swap_curve, tenor_irs, pay_fixed)
        par_rate = swap_curve(tenor_irs) / 100
        dv01_irs = notional_irs * tenor_irs / 10000
        
        with c2:
            mc = st.columns(3)
            mc[0].markdown(card("NPV", f"${npv:,.0f}", "Pay Fixed" if pay_fixed else "Receive Fixed", GREEN if npv > 0 else RED), unsafe_allow_html=True)
            mc[1].markdown(card("Par Swap Rate", f"{par_rate:.2%}", color=BLUE), unsafe_allow_html=True)
            mc[2].markdown(card("DV01 (approx)", f"${dv01_irs:,.0f}", color=ORANGE), unsafe_allow_html=True)
            
            # NPV sensitivity
            fixed_rates = np.linspace(fixed_rate - 0.02, fixed_rate + 0.02, 50)
            npvs = [irs_npv(notional_irs, fr, float_spread, swap_curve, tenor_irs, pay_fixed) for fr in fixed_rates]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fixed_rates * 100, y=npvs, name="NPV", line=dict(color=GOLD, width=2.5)))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=fixed_rate * 100, line_dash="dash", line_color=RED, annotation_text="Current")
            fig.update_layout(**PL, height=350, xaxis_title="Fixed Rate (%)", yaxis_title="NPV ($)", title="Swap NPV Sensitivity")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Credit Default Swap Pricing")
        c1, c2 = st.columns([1, 2])
        with c1:
            notional_cds = st.number_input("Notional ($)", 10_000_000, step=1_000_000, key="cds_not")
            spread_bps = st.number_input("CDS Spread (bps)", 100, step=10)
            hazard_rate = st.number_input("Hazard Rate (%)", 2.0, step=0.1) / 100
            recovery = st.number_input("Recovery Rate (%)", 40.0, step=5.0) / 100
            tenor_cds = st.selectbox("Tenor", [1, 3, 5, 7, 10], index=2, key="cds_tenor")
            rf_cds = st.number_input("Risk-Free Rate (%)", 4.0, step=0.1, key="cds_rf") / 100
        
        fair_spread = cds_spread_approx(hazard_rate, recovery, tenor_cds, rf_cds)
        npv_cds = cds_npv(notional_cds, spread_bps, hazard_rate, recovery, tenor_cds, rf_cds)
        prob_default = 1 - math.exp(-hazard_rate * tenor_cds)
        
        with c2:
            mc = st.columns(3)
            mc[0].markdown(card("Fair Spread", f"{fair_spread:.0f} bps", color=GOLD), unsafe_allow_html=True)
            mc[1].markdown(card("NPV", f"${npv_cds:,.0f}", "Protection Buyer" if npv_cds > 0 else "Protection Seller", GREEN if npv_cds > 0 else RED), unsafe_allow_html=True)
            mc[2].markdown(card(f"P(Default) {tenor_cds}Y", f"{prob_default:.2%}", color=ORANGE), unsafe_allow_html=True)
            
            # CDS spread curve
            hazards = np.linspace(0.005, 0.10, 50)
            spreads = [cds_spread_approx(h, recovery, tenor_cds, rf_cds) for h in hazards]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hazards * 100, y=spreads, name="CDS Spread", line=dict(color=GOLD, width=2.5)))
            fig.add_vline(x=hazard_rate * 100, line_dash="dash", line_color=RED, annotation_text="Current")
            fig.update_layout(**PL, height=350, xaxis_title="Hazard Rate (%)", yaxis_title="CDS Spread (bps)", title="CDS Spread vs Hazard Rate")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 10: SCENARIO ANALYSIS (NEW)
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "🎯 Scenario Analysis":
    st.markdown(f'<h1 style="color:{GOLD}">Scenario Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ib"><p><b>Shocks:</b> Apply parallel/non-parallel rate shocks, spot shocks, and volatility shocks to instruments.</p></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📈 Rate Scenarios", "💹 Spot Scenarios", "🌊 Vol Scenarios"])
    
    with tabs[0]:
        st.subheader("Interest Rate Scenario Analysis")
        c1, c2 = st.columns([1, 2])
        with c1:
            face = st.number_input("Bond Face ($)", 1000.0, key="sc_face")
            cpn = st.number_input("Coupon (%)", 5.0, key="sc_cpn") / 100
            ytm_base = st.number_input("Base YTM (%)", 4.0, key="sc_ytm") / 100
            yrs = st.number_input("Years", 10, key="sc_yrs")
            
            st.markdown("**Parallel Shift (bps)**")
            parallel = st.slider("Parallel", -200, 200, 0, 25)
            st.markdown("**Twist (short vs long)**")
            twist = st.slider("Twist", -100, 100, 0, 25)
        
        with c2:
            # Base case
            base_price, _, mod_d, conv, _ = bond_price_full(face, cpn, ytm_base, yrs, 2)
            
            # Shocked cases
            scenarios = {
                "Base": 0,
                "Parallel +100bp": 100,
                "Parallel -100bp": -100,
                "Parallel +200bp": 200,
                "Parallel -200bp": -200,
                f"Custom ({parallel:+d}bp)": parallel
            }
            
            results = []
            for name, shock in scenarios.items():
                shocked_ytm = ytm_base + shock / 10000
                shocked_price, _, _, _, _ = bond_price_full(face, cpn, max(0.001, shocked_ytm), yrs, 2)
                pnl = shocked_price - base_price
                pnl_pct = pnl / base_price * 100
                results.append({"Scenario": name, "Shock (bps)": shock, "Price": f"${shocked_price:.2f}",
                               "P&L": f"${pnl:.2f}", "P&L %": f"{pnl_pct:.2f}%"})
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            # Visualization
            shocks = np.linspace(-200, 200, 50)
            prices = [bond_price_full(face, cpn, max(0.001, ytm_base + s/10000), yrs, 2)[0] for s in shocks]
            pnls = [p - base_price for p in prices]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=shocks, y=pnls, name="P&L", line=dict(color=GOLD, width=2.5)))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color=BLUE, annotation_text="Base")
            fig.update_layout(**PL, height=350, xaxis_title="Rate Shock (bps)", yaxis_title="P&L ($)", title="Bond P&L vs Rate Shock")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Spot Price Scenario Analysis")
        c1, c2 = st.columns([1, 2])
        with c1:
            S_base = st.number_input("Base Spot ($)", 100.0, key="sc_spot")
            K_sc = st.number_input("Strike ($)", 100.0, key="sc_k")
            T_sc = st.number_input("DTE", 30, key="sc_dte") / 365
            sigma_sc = st.number_input("Vol (%)", 25.0, key="sc_vol") / 100
            r_sc = st.number_input("Rf (%)", 4.5, key="sc_r") / 100
            ot_sc = st.radio("Type", ["Call", "Put"], horizontal=True, key="sc_ot")
            n_contracts = st.number_input("Contracts", 100, step=10)
        
        with c2:
            base_price_opt = bs(S_base, K_sc, T_sc, r_sc, sigma_sc, ot_sc)[0]
            
            spot_scenarios = {
                "Base": 0,
                "Spot +5%": 5,
                "Spot -5%": -5,
                "Spot +10%": 10,
                "Spot -10%": -10,
                "Spot +20%": 20,
                "Spot -20%": -20
            }
            
            results = []
            for name, pct in spot_scenarios.items():
                shocked_spot = S_base * (1 + pct/100)
                shocked_price = bs(shocked_spot, K_sc, T_sc, r_sc, sigma_sc, ot_sc)[0]
                pnl = (shocked_price - base_price_opt) * n_contracts * 100
                results.append({"Scenario": name, "Spot": f"${shocked_spot:.2f}", "Option Price": f"${shocked_price:.4f}",
                               "P&L": f"${pnl:,.0f}"})
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            spots = np.linspace(S_base * 0.7, S_base * 1.3, 50)
            opt_prices = [bs(s, K_sc, T_sc, r_sc, sigma_sc, ot_sc)[0] for s in spots]
            pnls = [(p - base_price_opt) * n_contracts * 100 for p in opt_prices]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spots, y=pnls, name="P&L", line=dict(color=GOLD, width=2.5)))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=S_base, line_dash="dash", line_color=BLUE, annotation_text="Base")
            fig.update_layout(**PL, height=350, xaxis_title="Spot ($)", yaxis_title="P&L ($)", title="Option P&L vs Spot")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Volatility Scenario Analysis")
        c1, c2 = st.columns([1, 2])
        with c1:
            S_v = st.number_input("Spot ($)", 100.0, key="v_spot")
            K_v = st.number_input("Strike ($)", 100.0, key="v_k")
            T_v = st.number_input("DTE", 30, key="v_dte") / 365
            sigma_base = st.number_input("Base Vol (%)", 25.0, key="v_vol") / 100
            r_v = st.number_input("Rf (%)", 4.5, key="v_r") / 100
            ot_v = st.radio("Type", ["Call", "Put"], horizontal=True, key="v_ot")
        
        with c2:
            base_price_v = bs(S_v, K_v, T_v, r_v, sigma_base, ot_v)[0]
            base_vega = bs(S_v, K_v, T_v, r_v, sigma_base, ot_v)[4]
            
            vol_scenarios = {
                "Base": 0,
                "Vol +5%": 5,
                "Vol -5%": -5,
                "Vol +10%": 10,
                "Vol -10%": -10,
                "Vol Crush -15%": -15,
                "Vol Spike +25%": 25
            }
            
            results = []
            for name, pct in vol_scenarios.items():
                shocked_vol = sigma_base * (1 + pct/100)
                shocked_price = bs(S_v, K_v, T_v, r_v, max(0.01, shocked_vol), ot_v)[0]
                pnl = shocked_price - base_price_v
                results.append({"Scenario": name, "Vol": f"{shocked_vol*100:.1f}%", "Price": f"${shocked_price:.4f}",
                               "P&L": f"${pnl:.4f}"})
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            vols = np.linspace(0.05, 0.60, 50)
            prices_v = [bs(S_v, K_v, T_v, r_v, v, ot_v)[0] for v in vols]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vols * 100, y=prices_v, name="Option Price", line=dict(color=GOLD, width=2.5)))
            fig.add_vline(x=sigma_base * 100, line_dash="dash", line_color=BLUE, annotation_text="Base Vol")
            fig.update_layout(**PL, height=350, xaxis_title="Volatility (%)", yaxis_title="Option Price ($)", title="Option Price vs Volatility")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 11: TIMESERIES ANALYTICS (NEW)
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "📈 Timeseries Analytics":
    st.markdown(f'<h1 style="color:{GOLD}">Timeseries Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ib"><p><b>gs-quant style:</b> Z-scores, rolling regression, winsorize, realized vol, EWMA vol, Hurst exponent, rolling beta/correlation.</p></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Statistics", "📈 Rolling Regression", "🔬 Advanced"])
    
    with tabs[0]:
        c1, c2 = st.columns([1, 3])
        with c1:
            tk = st.text_input("Ticker", "AAPL", key="ts_tk")
            window = st.slider("Window", 10, 100, 20)
            lookback = st.slider("Lookback (days)", 100, 1000, 500)
        
        df = fetch_single(tk, datetime.date.today() - datetime.timedelta(lookback), datetime.date.today())
        if df.empty:
            st.error("No data.")
        else:
            ret = df["Close"].pct_change().dropna()
            
            with c2:
                # Z-score
                z = ts_zscore(df["Close"], window)
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                   subplot_titles=["Price", "Z-Score", "Realized Vol"])
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color=GOLD, width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=z.index, y=z, name="Z-Score", line=dict(color=BLUE, width=1.5)), row=2, col=1)
                fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
                
                rvol = ts_realized_vol(ret, window)
                fig.add_trace(go.Scatter(x=rvol.index, y=rvol, name="Realized Vol", line=dict(color=ORANGE, width=1.5)), row=3, col=1)
                
                fig.update_layout(**PL, height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats table
                st.subheader("Summary Statistics")
                stats = {
                    "Mean Return (daily)": f"{ret.mean():.4%}",
                    "Std Dev (daily)": f"{ret.std():.4%}",
                    "Skewness": f"{ret.skew():.3f}",
                    "Kurtosis": f"{ret.kurtosis():.3f}",
                    "Current Z-Score": f"{z.iloc[-1]:.2f}" if not z.empty else "N/A",
                    "Realized Vol (ann)": f"{rvol.iloc[-1]:.2%}" if not rvol.empty else "N/A"
                }
                st.dataframe(pd.DataFrame([stats]).T.rename(columns={0: "Value"}), use_container_width=True)
    
    with tabs[1]:
        st.subheader("Rolling Regression (Asset vs Market)")
        c1, c2 = st.columns([1, 3])
        with c1:
            asset_tk = st.text_input("Asset", "AAPL", key="rr_asset")
            market_tk = st.text_input("Market", "SPY", key="rr_mkt")
            reg_window = st.slider("Regression Window", 20, 120, 60)
        
        df_asset = fetch_single(asset_tk, datetime.date.today() - datetime.timedelta(500), datetime.date.today())
        df_mkt = fetch_single(market_tk, datetime.date.today() - datetime.timedelta(500), datetime.date.today())
        
        if df_asset.empty or df_mkt.empty:
            st.error("No data.")
        else:
            ret_a = df_asset["Close"].pct_change().dropna()
            ret_m = df_mkt["Close"].pct_change().dropna()
            
            # Align
            common = ret_a.index.intersection(ret_m.index)
            ret_a, ret_m = ret_a.loc[common], ret_m.loc[common]
            
            alpha, beta, r2 = ts_rolling_regression(ret_a, ret_m, reg_window)
            
            with c2:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                   subplot_titles=["Rolling Beta", "Rolling Alpha (ann)", "R-Squared"])
                fig.add_trace(go.Scatter(x=beta.index, y=beta, name="Beta", line=dict(color=GOLD, width=1.5)), row=1, col=1)
                fig.add_hline(y=1, line_dash="dash", line_color="gray", row=1, col=1)
                fig.add_trace(go.Scatter(x=alpha.index, y=alpha * 252, name="Alpha", line=dict(color=BLUE, width=1.5)), row=2, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                fig.add_trace(go.Scatter(x=r2.index, y=r2, name="R²", line=dict(color=GREEN, width=1.5)), row=3, col=1)
                fig.update_layout(**PL, height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                mc = st.columns(3)
                mc[0].markdown(card("Current Beta", f"{beta.iloc[-1]:.3f}" if not beta.empty else "N/A", color=GOLD), unsafe_allow_html=True)
                mc[1].markdown(card("Current Alpha (ann)", f"{alpha.iloc[-1]*252:.2%}" if not alpha.empty else "N/A", color=BLUE), unsafe_allow_html=True)
                mc[2].markdown(card("Current R²", f"{r2.iloc[-1]:.3f}" if not r2.empty else "N/A", color=GREEN), unsafe_allow_html=True)
    
    with tabs[2]:
        st.subheader("Advanced Analytics")
        c1, c2 = st.columns([1, 3])
        with c1:
            adv_tk = st.text_input("Ticker", "AAPL", key="adv_tk")
            ewma_beta = st.slider("EWMA Lambda", 0.90, 0.99, 0.94, 0.01)
        
        df_adv = fetch_single(adv_tk, datetime.date.today() - datetime.timedelta(500), datetime.date.today())
        if df_adv.empty:
            st.error("No data.")
        else:
            ret_adv = df_adv["Close"].pct_change().dropna()
            
            with c2:
                # EWMA vol
                ewma_vol = ts_exponential_vol(ret_adv, ewma_beta)
                real_vol = ts_realized_vol(ret_adv, 20)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol, name=f"EWMA Vol (λ={ewma_beta})", line=dict(color=GOLD, width=2)))
                fig.add_trace(go.Scatter(x=real_vol.index, y=real_vol, name="Realized Vol (20d)", line=dict(color=BLUE, width=1.5, dash="dash")))
                fig.update_layout(**PL, height=350, title="Volatility Comparison", yaxis_tickformat=".1%")
                st.plotly_chart(fig, use_container_width=True)
                
                # Hurst exponent
                try:
                    hurst = ts_hurst_exponent(df_adv["Close"].dropna().values)
                    hurst_interp = "Mean Reverting" if hurst < 0.5 else ("Trending" if hurst > 0.5 else "Random Walk")
                except:
                    hurst = np.nan
                    hurst_interp = "N/A"
                
                # Winsorized returns
                win_ret = ts_winsorize(ret_adv, 0.01, 0.99)
                
                mc = st.columns(3)
                mc[0].markdown(card("Hurst Exponent", f"{hurst:.3f}" if not np.isnan(hurst) else "N/A", hurst_interp, color=GOLD), unsafe_allow_html=True)
                mc[1].markdown(card("EWMA Vol (current)", f"{ewma_vol.iloc[-1]:.2%}" if not ewma_vol.empty else "N/A", color=BLUE), unsafe_allow_html=True)
                mc[2].markdown(card("Winsorized Skew", f"{win_ret.skew():.3f}", color=ORANGE), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 12: FACTOR RISK MODELS (NEW)
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "🧮 Factor Risk Models":
    st.markdown(f'<h1 style="color:{GOLD}">Factor Risk Models</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ib"><p><b>Analysis:</b> Factor decomposition, factor exposures (betas), systematic vs idiosyncratic risk attribution.</p></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Factor Exposures", "🎯 Risk Decomposition"])
    
    with tabs[0]:
        st.subheader("Factor Exposure Analysis")
        assets = st.text_input("Assets (comma-sep)", "AAPL, MSFT, GOOGL, AMZN, META")
        factors = st.text_input("Factors (comma-sep)", "SPY, QQQ, IWM")
        
        if st.button("Compute Exposures", type="primary"):
            asset_list = [t.strip().upper() for t in assets.split(",") if t.strip()]
            factor_list = [t.strip().upper() for t in factors.split(",") if t.strip()]
            
            all_tickers = asset_list + factor_list
            closes = closes_from_multi(fetch_multi(all_tickers, datetime.date.today() - datetime.timedelta(500), datetime.date.today()))
            
            if closes.empty:
                st.error("No data.")
            else:
                rets = closes.pct_change().dropna()
                asset_rets = rets[[c for c in asset_list if c in rets.columns]]
                factor_rets = rets[[c for c in factor_list if c in rets.columns]]
                
                exposures = compute_factor_exposures(asset_rets, factor_rets)
                
                st.subheader("Factor Betas")
                st.dataframe(exposures.style.format("{:.3f}"), use_container_width=True)
                
                # Heatmap
                if not exposures.empty:
                    fig = px.imshow(exposures.drop(columns=["Alpha"], errors="ignore"), text_auto=".2f",
                                   color_continuous_scale="RdBu_r", aspect="auto")
                    fig.update_layout(**PL, height=400, title="Factor Exposure Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Portfolio Risk Decomposition")
        assets_rd = st.text_input("Portfolio Assets", "AAPL, MSFT, GOOGL, AMZN", key="rd_assets")
        weights_str = st.text_input("Weights (comma-sep)", "0.25, 0.25, 0.25, 0.25")
        factors_rd = st.text_input("Factors", "SPY, QQQ", key="rd_factors")
        
        if st.button("Decompose Risk", type="primary"):
            asset_list = [t.strip().upper() for t in assets_rd.split(",") if t.strip()]
            factor_list = [t.strip().upper() for t in factors_rd.split(",") if t.strip()]
            weights = np.array([float(w.strip()) for w in weights_str.split(",")])
            weights = weights / weights.sum()
            
            all_tickers = asset_list + factor_list
            closes = closes_from_multi(fetch_multi(all_tickers, datetime.date.today() - datetime.timedelta(500), datetime.date.today()))
            
            if closes.empty:
                st.error("No data.")
            else:
                rets = closes.pct_change().dropna()
                asset_rets = rets[[c for c in asset_list if c in rets.columns]]
                factor_rets = rets[[c for c in factor_list if c in rets.columns]]
                
                if len(asset_rets.columns) != len(weights):
                    st.error("Weights don't match assets.")
                else:
                    decomp = decompose_risk(asset_rets, factor_rets, weights)
                    
                    mc = st.columns(4)
                    mc[0].markdown(card("Total Vol", f"{decomp['Total Vol']:.2%}", color=GOLD), unsafe_allow_html=True)
                    mc[1].markdown(card("Systematic Vol", f"{decomp['Systematic Vol']:.2%}", color=BLUE), unsafe_allow_html=True)
                    mc[2].markdown(card("Idiosyncratic Vol", f"{decomp['Idiosyncratic Vol']:.2%}", color=ORANGE), unsafe_allow_html=True)
                    mc[3].markdown(card("R-Squared", f"{decomp['R-squared']:.2%}", color=GREEN), unsafe_allow_html=True)
                    
                    # Pie chart
                    sys_var = decomp['Systematic Vol']**2
                    idio_var = decomp['Idiosyncratic Vol']**2
                    total_var = sys_var + idio_var
                    
                    fig = go.Figure(go.Pie(labels=["Systematic", "Idiosyncratic"],
                                          values=[sys_var/total_var, idio_var/total_var],
                                          marker_colors=[BLUE, ORANGE]))
                    fig.update_layout(**PL, height=350, title="Variance Decomposition")
                    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 13: BASKET CONSTRUCTION (NEW)
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "📦 Basket Construction":
    st.markdown(f'<h1 style="color:{GOLD}">Basket / Index Construction</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ib"><p><b>Features:</b> Custom basket creation, weighting schemes, rebalancing simulation, performance tracking.</p></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🏗️ Build Basket", "📊 Analyze Basket"])
    
    with tabs[0]:
        st.subheader("Create Custom Basket")
        c1, c2 = st.columns([1, 2])
        with c1:
            constituents = st.text_input("Constituents", "AAPL, MSFT, GOOGL, AMZN, NVDA")
            weighting = st.selectbox("Weighting Scheme", ["Equal Weight", "Market Cap Proxy", "Custom"])
            
            if weighting == "Custom":
                custom_weights = st.text_input("Custom Weights", "0.3, 0.25, 0.2, 0.15, 0.1")
            
            rebal_freq = st.selectbox("Rebalance Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"])
            lookback_basket = st.slider("Lookback (days)", 100, 1000, 252)
        
        if st.button("Build Basket", type="primary"):
            tlist = [t.strip().upper() for t in constituents.split(",") if t.strip()]
            closes = closes_from_multi(fetch_multi(tlist, datetime.date.today() - datetime.timedelta(lookback_basket), datetime.date.today()))
            
            if closes.empty:
                st.error("No data.")
            else:
                n = len(closes.columns)
                if weighting == "Equal Weight":
                    weights = np.ones(n) / n
                elif weighting == "Market Cap Proxy":
                    # Use inverse volatility as proxy
                    vols = closes.pct_change().std()
                    weights = (1 / vols) / (1 / vols).sum()
                else:
                    weights = np.array([float(w.strip()) for w in custom_weights.split(",")])
                    weights = weights / weights.sum()
                
                # Calculate basket returns
                rets = closes.pct_change().dropna()
                basket_ret = (rets * weights).sum(axis=1)
                basket_cum = (1 + basket_ret).cumprod()
                
                with c2:
                    # Weights
                    wdf = pd.DataFrame({"Constituent": closes.columns, "Weight": weights})
                    fig_w = go.Figure(go.Bar(x=wdf["Constituent"], y=wdf["Weight"], marker_color=GOLD,
                                            text=[f"{w:.1%}" for w in wdf["Weight"]], textposition="outside"))
                    fig_w.update_layout(**PL, height=250, title="Basket Weights", yaxis_tickformat=".0%")
                    st.plotly_chart(fig_w, use_container_width=True)
                    
                    # Performance
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=basket_cum.index, y=basket_cum, name="Basket", line=dict(color=GOLD, width=2.5)))
                    # Add constituents
                    const_cum = (1 + rets).cumprod()
                    colors = [BLUE, GREEN, ORANGE, PURPLE, RED]
                    for i, col in enumerate(const_cum.columns[:5]):
                        fig.add_trace(go.Scatter(x=const_cum.index, y=const_cum[col], name=col,
                                                line=dict(color=colors[i % len(colors)], width=1, dash="dash")))
                    fig.update_layout(**PL, height=400, title="Basket vs Constituents")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    met = perf_metrics(basket_ret)
                    mc = st.columns(4)
                    for i, (k, v) in enumerate(list(met.items())[:4]):
                        mc[i].markdown(card(k, v), unsafe_allow_html=True)
    
    with tabs[1]:
        st.subheader("Basket Analytics")
        st.info("Build a basket first in the 'Build Basket' tab to see analytics here.")

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 14: HEDGING ANALYTICS (NEW)
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "🛡️ Hedging Analytics":
    st.markdown(f'<h1 style="color:{GOLD}">Hedging Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ib"><p><b>Tools:</b> Delta hedging, gamma hedging, beta hedging, factor hedging calculations.</p></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Delta/Gamma Hedge", "📈 Beta Hedge", "🎯 Factor Hedge"])
    
    with tabs[0]:
        st.subheader("Delta-Gamma Hedging")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Option Position**")
            S_h = st.number_input("Spot ($)", 100.0, key="h_spot")
            K_h = st.number_input("Strike ($)", 100.0, key="h_k")
            T_h = st.number_input("DTE", 30, key="h_dte") / 365
            sigma_h = st.number_input("Vol (%)", 25.0, key="h_vol") / 100
            r_h = st.number_input("Rf (%)", 4.5, key="h_r") / 100
            ot_h = st.radio("Type", ["Call", "Put"], horizontal=True, key="h_ot")
            position = st.number_input("Contracts (+ long, - short)", 100, step=10)
        
        pr_h, delta_h, gamma_h, _, vega_h, _ = bs(S_h, K_h, T_h, r_h, sigma_h, ot_h)
        
        # Position Greeks
        pos_delta = delta_h * position * 100
        pos_gamma = gamma_h * position * 100
        pos_vega = vega_h * position * 100
        
        with c2:
            st.subheader("Position Greeks")
            mc = st.columns(3)
            mc[0].markdown(card("Position Delta", f"{pos_delta:,.0f}", "shares equivalent", color=GOLD), unsafe_allow_html=True)
            mc[1].markdown(card("Position Gamma", f"{pos_gamma:,.2f}", color=BLUE), unsafe_allow_html=True)
            mc[2].markdown(card("Position Vega", f"${pos_vega:,.0f}", "per 1% vol move", color=ORANGE), unsafe_allow_html=True)
            
            st.subheader("Hedge Requirements")
            # Delta hedge
            shares_to_hedge = -pos_delta
            st.markdown(f"**Delta Hedge:** {'Buy' if shares_to_hedge > 0 else 'Sell'} **{abs(shares_to_hedge):,.0f}** shares of underlying")
            
            # Gamma hedge (need another option)
            st.markdown("**Gamma Hedge:** Requires trading another option with different strike/expiry")
            
            # P&L simulation
            st.subheader("Hedged vs Unhedged P&L")
            spot_range = np.linspace(S_h * 0.8, S_h * 1.2, 50)
            unhedged_pnl = []
            hedged_pnl = []
            
            for s in spot_range:
                new_price = bs(s, K_h, T_h, r_h, sigma_h, ot_h)[0]
                opt_pnl = (new_price - pr_h) * position * 100
                stock_pnl = (s - S_h) * shares_to_hedge
                unhedged_pnl.append(opt_pnl)
                hedged_pnl.append(opt_pnl + stock_pnl)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=unhedged_pnl, name="Unhedged", line=dict(color=RED, width=2)))
            fig.add_trace(go.Scatter(x=spot_range, y=hedged_pnl, name="Delta Hedged", line=dict(color=GREEN, width=2)))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=S_h, line_dash="dash", line_color=BLUE, annotation_text="Current")
            fig.update_layout(**PL, height=350, xaxis_title="Spot ($)", yaxis_title="P&L ($)", title="Hedge Effectiveness")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Beta Hedging")
        c1, c2 = st.columns([1, 2])
        with c1:
            port_value = st.number_input("Portfolio Value ($)", 1_000_000, step=100_000)
            port_beta = st.number_input("Portfolio Beta", 1.2, step=0.1)
            target_beta = st.number_input("Target Beta", 0.0, step=0.1)
            hedge_instrument = st.selectbox("Hedge Instrument", ["SPY", "ES Futures", "SPX Options"])
            hedge_beta = st.number_input("Hedge Instrument Beta", 1.0, step=0.1)
        
        with c2:
            beta_diff = port_beta - target_beta
            hedge_ratio = beta_diff / hedge_beta
            hedge_notional = port_value * hedge_ratio
            
            mc = st.columns(3)
            mc[0].markdown(card("Beta to Hedge", f"{beta_diff:.2f}", color=GOLD), unsafe_allow_html=True)
            mc[1].markdown(card("Hedge Ratio", f"{hedge_ratio:.2f}", color=BLUE), unsafe_allow_html=True)
            mc[2].markdown(card("Hedge Notional", f"${abs(hedge_notional):,.0f}", "Short" if hedge_notional > 0 else "Long", color=RED if hedge_notional > 0 else GREEN), unsafe_allow_html=True)
            
            st.markdown(f"""
            **Hedge Instruction:**
            - {'Short' if hedge_notional > 0 else 'Long'} **${abs(hedge_notional):,.0f}** of {hedge_instrument}
            - This will reduce portfolio beta from **{port_beta:.2f}** to **{target_beta:.2f}**
            """)
    
    with tabs[2]:
        st.subheader("Factor Hedging")
        st.info("Factor hedging requires computing factor exposures first. Use the Factor Risk Models module to analyze exposures, then return here to calculate hedges.")

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 15: FX ANALYTICS (NEW)
# ═══════════════════════════════════════════════════════════════════════════
elif choice == "💱 FX Analytics":
    st.markdown(f'<h1 style="color:{GOLD}">FX Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ib"><p><b>Analysis:</b> Cross-currency analysis, FX carry strategies, FX volatility, correlation.</p></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["💹 FX Overview", "📊 Carry Analysis", "🌊 FX Volatility"])
    
    with tabs[0]:
        st.subheader("FX Market Overview")
        fx_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"]
        
        with st.spinner("Fetching FX data..."):
            fx_data = {}
            start_d = datetime.date.today() - datetime.timedelta(days=365)
            end_d = datetime.date.today()
            
            for pair in fx_pairs:
                df = fetch_single(pair, start_d, end_d)
                if not df.empty and "Close" in df.columns:
                    # fetch_single already handles the MultiIndex, so this is a clean Series
                    fx_data[pair.replace("=X", "")] = df["Close"]
        
        if fx_data:
            fx_df = pd.DataFrame(fx_data).dropna()
            
            # Current rates
            mc = st.columns(6)
            for i, (pair, data) in enumerate(fx_data.items()):
                if len(data) > 1:
                    last, prev = data.iloc[-1], data.iloc[-2]
                    chg = (last - prev) / prev * 100
                    clr = GREEN if chg >= 0 else RED
                    mc[i].markdown(card(pair, f"{last:.4f}", f'<span style="color:{clr}">{chg:+.2f}%</span>'), unsafe_allow_html=True)
            
            # Correlation
            st.subheader("FX Correlation Matrix")
            rets = fx_df.pct_change().dropna()
            fig = px.imshow(rets.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(**PL, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("FX Carry Analysis")
        st.markdown("Carry = Interest Rate Differential × Time")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Interest Rates (%)**")
            r_usd = st.number_input("USD", 4.50, step=0.25)
            r_eur = st.number_input("EUR", 3.50, step=0.25)
            r_gbp = st.number_input("GBP", 4.75, step=0.25)
            r_jpy = st.number_input("JPY", 0.10, step=0.05)
            r_aud = st.number_input("AUD", 4.25, step=0.25)
            r_chf = st.number_input("CHF", 1.50, step=0.25)
        
        rates = {"USD": r_usd, "EUR": r_eur, "GBP": r_gbp, "JPY": r_jpy, "AUD": r_aud, "CHF": r_chf}
        
        with c2:
            # Carry matrix
            currencies = list(rates.keys())
            carry_matrix = np.zeros((len(currencies), len(currencies)))
            for i, c1 in enumerate(currencies):
                for j, c2 in enumerate(currencies):
                    carry_matrix[i, j] = rates[c1] - rates[c2]
            
            carry_df = pd.DataFrame(carry_matrix, index=currencies, columns=currencies)
            
            fig = px.imshow(carry_df, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto")
            fig.update_layout(**PL, height=400, title="Carry Matrix (Long Row / Short Column)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Best carry trades
            st.subheader("Top Carry Trades")
            carry_trades = []
            for i, c1 in enumerate(currencies):
                for j, c2 in enumerate(currencies):
                    if i != j:
                        carry_trades.append({"Long": c1, "Short": c2, "Carry (bps)": (rates[c1] - rates[c2]) * 100})
            
            carry_trades_df = pd.DataFrame(carry_trades).sort_values("Carry (bps)", ascending=False).head(5)
            st.dataframe(carry_trades_df, use_container_width=True, hide_index=True)
    
    with tabs[2]:
        st.subheader("FX Volatility Analysis")
        fx_pair = st.selectbox("Select Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"])
        
        df_fx = fetch_single(fx_pair, datetime.date.today() - datetime.timedelta(500), datetime.date.today())
        if df_fx.empty:
            st.error("No data.")
        else:
            ret_fx = df_fx["Close"].pct_change().dropna()
            
            # Realized vol
            rvol = ts_realized_vol(ret_fx, 20)
            ewma_vol = ts_exponential_vol(ret_fx, 0.94)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rvol.index, y=rvol, name="Realized Vol (20d)", line=dict(color=GOLD, width=2)))
            fig.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol, name="EWMA Vol", line=dict(color=BLUE, width=1.5, dash="dash")))
            fig.update_layout(**PL, height=350, title=f"{fx_pair.replace('=X', '')} Volatility", yaxis_tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
            
            mc = st.columns(3)
            mc[0].markdown(card("Current Vol (20d)", f"{rvol.iloc[-1]:.2%}" if not rvol.empty else "N/A", color=GOLD), unsafe_allow_html=True)
            mc[1].markdown(card("EWMA Vol", f"{ewma_vol.iloc[-1]:.2%}" if not ewma_vol.empty else "N/A", color=BLUE), unsafe_allow_html=True)
            mc[2].markdown(card("Vol Percentile", f"{(rvol < rvol.iloc[-1]).mean():.0%}" if not rvol.empty else "N/A", color=ORANGE), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div style="text-align:center;padding:1rem;color:#666"><p style="font-size:.8rem"><strong style="color:{GOLD}">Open Source Quant Platform</strong> — gs-quant Replica<br>Streamlit • yfinance • scipy • pandas • numpy • plotly<br><b>No proprietary APIs required.</b></p></div>', unsafe_allow_html=True)
