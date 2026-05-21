<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</p>

<h1 align="center">📈 Open Source Quant Platform</h1>

<p align="center">
  <strong>A professional-grade quantitative finance toolkit built entirely with open-source tools.</strong><br>
  <em>Inspired by Goldman Sachs' gs-quant — No proprietary APIs required.</em>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-modules">Modules</a> •
  <a href="#-screenshots">Screenshots</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🎯 Overview

This platform provides institutional-grade quantitative finance capabilities accessible to everyone. Whether you're a quant researcher, portfolio manager, risk analyst, or finance student, this toolkit offers the analytical power typically reserved for expensive proprietary systems.

### Why This Project?

- **No Vendor Lock-in**: Built entirely on open-source libraries
- **No API Keys Required**: Uses free market data from Yahoo Finance
- **Production-Ready**: Professional UI with real-time calculations
- **Educational**: Well-documented code for learning quantitative finance
- **Extensible**: Modular architecture for easy customization

---

## ✨ Features

### Core Capabilities

| Category | Features |
|----------|----------|
| **Market Data** | Real-time quotes, OHLCV data, technical indicators (RSI, MACD, Bollinger Bands, SMA/EMA) |
| **Derivatives** | Black-Scholes pricing, 5 Greeks (Δ, Γ, Θ, ν, ρ), IV solver, 3D surfaces, payoff diagrams |
| **Risk Analytics** | Historical/Parametric/Monte Carlo VaR, Expected Shortfall, stress testing |
| **Portfolio** | Mean-Variance Optimization, Efficient Frontier, Max Sharpe/Min Vol portfolios |
| **Fixed Income** | Bond pricing, yield curves (cubic spline), duration, convexity, DV01 |
| **Multi-Asset** | FX Forwards, Interest Rate Swaps, Credit Default Swaps |
| **Factor Models** | Factor decomposition, systematic vs idiosyncratic risk attribution |

### Technical Highlights

- 🚀 **Sub-second calculations** for real-time analysis
- 📊 **Interactive 3D visualizations** with Plotly
- 🔄 **Automatic data caching** for performance
- 📱 **Responsive design** works on desktop and tablet
- 🎨 **Professional dark theme** with gold accents

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/quant-platform.git
cd quant-platform
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install streamlit numpy pandas scipy plotly yfinance statsmodels
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

### Requirements File

Create a `requirements.txt` with:

```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
plotly>=5.18.0
yfinance>=0.2.31
statsmodels>=0.14.0
```

---

## 🚀 Quick Start

### Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### First Steps

1. **Dashboard**: Start with the Dashboard to see a market overview
2. **Market Data**: Analyze individual securities with technical indicators
3. **Derivatives**: Price options and visualize Greeks
4. **Portfolio**: Build and optimize a portfolio

---

## 📚 Modules

### 1. 🏠 Dashboard — Global Market Overview

Real-time market dashboard with:
- Multi-asset price tracking
- Normalized performance comparison
- Correlation matrix heatmap
- Customizable universe selection

**Use Case**: Morning market review, cross-asset analysis

---

### 2. 📊 Market Data — Technical Analysis

Comprehensive technical analysis suite:
- OHLCV candlestick charts
- Bollinger Bands (20-day, 2σ)
- RSI (14-period) with overbought/oversold zones
- MACD with signal line
- SMA/EMA overlays (20, 50, 200-day)

**Use Case**: Technical trading signals, trend identification

---

### 3. 💰 Derivatives Pricing — Options Analytics

Full Black-Scholes implementation:

```
Price = S·N(d₁) - K·e^(-rT)·N(d₂)  [Call]
Price = K·e^(-rT)·N(-d₂) - S·N(-d₁)  [Put]
```

Features:
- Real-time Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Interactive 3D Greeks surfaces
- Implied Volatility solver (Brent's method)
- Payoff diagrams at expiry

**Use Case**: Options trading, hedging analysis, IV analysis

---

### 4. ⚠️ Risk Analytics — VaR & Stress Testing

Three VaR methodologies:

| Method | Description |
|--------|-------------|
| **Historical** | Non-parametric, uses actual return distribution |
| **Parametric** | Assumes normal distribution, uses μ and σ |
| **Monte Carlo** | GBM simulation with configurable paths |

Additional features:
- Expected Shortfall (CVaR)
- Stress test scenarios (Bull/Bear/Crash)
- Return distribution visualization

**Use Case**: Risk management, regulatory reporting

---

### 5. 📁 Portfolio Management — Optimization

Mean-Variance Optimization (Markowitz):

```
max  (w'μ - rf) / √(w'Σw)  [Sharpe Ratio]
s.t. Σwᵢ = 1, wᵢ ≥ 0
```

Features:
- Maximum Sharpe Ratio portfolio
- Minimum Volatility portfolio
- Efficient Frontier visualization
- Weight allocation charts

**Use Case**: Asset allocation, portfolio construction

---

### 6. 🔄 Backtesting — Strategy Testing

Pre-built strategies:

| Strategy | Logic |
|----------|-------|
| **SMA Crossover** | Long when Fast MA > Slow MA |
| **Mean Reversion** | Long when Z-score < -1, exit when > 1 |
| **Momentum** | Long when 60-day return > 0 |

Performance metrics:
- Total/Annualized Return
- Sharpe/Sortino/Calmar Ratios
- Maximum Drawdown
- Win Rate

**Use Case**: Strategy development, historical analysis

---

### 7. 🌊 Volatility Surface — IV Modeling

Parametric IV surface with controls:
- ATM volatility level
- Skew (put/call asymmetry)
- Smile (wing curvature)
- Term structure

**Use Case**: Options market making, vol trading

---

### 8. 📐 Fixed Income — Bond Analytics

Bond pricing with full analytics:

```
Price = Σ(C/(1+y)^t) + F/(1+y)^n
```

Features:
- Clean/Dirty price calculation
- Macaulay & Modified Duration
- Convexity
- DV01 (dollar duration)
- Yield curve builder (cubic spline interpolation)

**Use Case**: Fixed income trading, ALM

---

### 9. 🏦 Multi-Asset Pricing — Derivatives

**FX Forwards** (Covered Interest Rate Parity):
```
F = S × e^((r_dom - r_for) × T)
```

**Interest Rate Swaps**:
- NPV calculation
- Par rate derivation
- DV01 sensitivity

**Credit Default Swaps**:
- Fair spread approximation
- NPV for protection buyer/seller
- Default probability

**Use Case**: Derivatives trading, hedging

---

### 10. 🎯 Scenario Analysis — What-If

Shock types:
- **Rate Scenarios**: Parallel shifts, curve twists
- **Spot Scenarios**: Equity/FX price moves
- **Vol Scenarios**: IV crush/spike analysis

**Use Case**: Risk scenarios, P&L attribution

---

### 11. 📈 Timeseries Analytics — Statistical Tools

gs-quant style functions:
- `ts_zscore()`: Rolling z-score
- `ts_winsorize()`: Outlier treatment
- `ts_rolling_beta()`: Rolling market beta
- `ts_rolling_regression()`: Rolling OLS (α, β, R²)
- `ts_exponential_vol()`: EWMA volatility (RiskMetrics)
- `ts_realized_vol()`: Historical volatility
- `ts_hurst_exponent()`: Mean reversion detection

**Use Case**: Quantitative research, signal generation

---

### 12. 🧮 Factor Risk Models — Attribution

Factor analysis:
- Multi-factor regression
- Factor exposure (beta) calculation
- Risk decomposition:
  - Systematic risk (factor-driven)
  - Idiosyncratic risk (stock-specific)
- R-squared attribution

**Use Case**: Risk attribution, factor investing

---

### 13. 📦 Basket Construction — Index Building

Custom index creation:
- Equal weight
- Inverse volatility weight
- Custom weights
- Rebalancing simulation
- Performance tracking vs constituents

**Use Case**: Custom benchmarks, thematic investing

---

### 14. 🛡️ Hedging Analytics — Risk Management

**Delta-Gamma Hedging**:
- Position Greeks aggregation
- Share equivalent calculation
- Hedge effectiveness visualization

**Beta Hedging**:
- Portfolio beta calculation
- Hedge ratio determination
- Notional sizing

**Use Case**: Options hedging, portfolio protection

---

### 15. 💱 FX Analytics — Currency Analysis

Features:
- Real-time FX rates
- Cross-currency correlation
- Carry trade analysis (rate differentials)
- FX volatility (realized & EWMA)
- Volatility percentile ranking

**Use Case**: FX trading, currency hedging

---

## 🏗️ Architecture

### Project Structure

```
quant-platform/
├── app.py              # Main application (single file)
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── LICENSE            # MIT License
└── assets/            # Screenshots and images
    └── screenshots/
```

### Technology Stack

```
┌─────────────────────────────────────────────────────┐
│                    Frontend                          │
│  Streamlit (UI) + Plotly (Visualization)            │
├─────────────────────────────────────────────────────┤
│                 Computation Layer                    │
│  NumPy (Arrays) + Pandas (DataFrames) + SciPy       │
├─────────────────────────────────────────────────────┤
│                    Data Layer                        │
│  yfinance (Market Data) + Caching (st.cache_data)   │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single-file architecture | Easy deployment, no import complexity |
| Streamlit caching | Performance optimization for repeated queries |
| Plotly for charts | Interactive, professional-quality visualizations |
| No database | Stateless design, data fetched on-demand |
| Dark theme | Reduced eye strain, professional appearance |

---

## 🔧 Configuration

### Customization Options

**Color Scheme** (in `app.py`):
```python
GOLD = "#D4AF37"    # Primary accent
BLUE = "#4FC3F7"    # Secondary accent
RED = "#FF1744"     # Negative values
GREEN = "#00C853"   # Positive values
```

**Cache Duration**:
```python
@st.cache_data(ttl=3600)  # 1 hour cache
```

**Default Tickers**:
Modify default values in `st.text_input()` calls.

---

## 📊 API Reference

### Core Functions

#### Black-Scholes Pricing
```python
def bs(S, K, T, r, sigma, otype="Call"):
    """
    Black-Scholes option pricing with Greeks.
    
    Parameters:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)
        otype: "Call" or "Put"
    
    Returns:
        tuple: (price, delta, gamma, theta, vega, rho)
    """
```

#### Implied Volatility
```python
def implied_vol(market_price, S, K, T, r, otype="Call"):
    """
    Solve for implied volatility using Brent's method.
    
    Returns:
        float: Implied volatility (decimal) or np.nan if no solution
    """
```

#### Bond Pricing
```python
def bond_price_full(face, cpn_rate, ytm, years, freq=2):
    """
    Full bond analytics.
    
    Returns:
        tuple: (price, macaulay_duration, modified_duration, convexity, dv01)
    """
```

#### Timeseries Functions
```python
def ts_zscore(series, window=20)           # Rolling z-score
def ts_winsorize(series, lower, upper)     # Clip outliers
def ts_rolling_beta(asset, market, window) # Rolling beta
def ts_realized_vol(returns, window)       # Historical vol
def ts_exponential_vol(returns, beta)      # EWMA vol
def ts_hurst_exponent(series, max_lag)     # Hurst exponent
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/quant-platform.git
cd quant-platform

# Create branch
git checkout -b feature/your-feature-name

# Make changes and test
streamlit run app.py

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### Contribution Ideas

- [ ] Add more technical indicators (ATR, OBV, etc.)
- [ ] Implement exotic option pricing (barriers, Asians)
- [ ] Add cryptocurrency support
- [ ] Create unit tests
- [ ] Add data export functionality
- [ ] Implement user authentication
- [ ] Add real-time streaming data

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions
- Keep functions focused and small

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Not suitable for live trading without proper validation
- Market data may be delayed
- Models use simplifying assumptions
- Past performance does not guarantee future results

Always consult with qualified financial professionals before making investment decisions.

---

## 🙏 Acknowledgments

- **Goldman Sachs** for the gs-quant inspiration
- **Streamlit** team for the amazing framework
- **yfinance** for free market data access
- **Plotly** for beautiful visualizations
- The open-source quantitative finance community

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/quant-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/quant-platform/discussions)

---

<p align="center">
  <strong>Built with ❤️ for the Quant Community</strong><br>
  <em>Star ⭐ this repo if you find it useful!</em>
</p>