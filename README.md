# DecodeLabs: Algorithmic Trading & Quantitative Analysis Suite 📈🐍

Welcome to my capstone Quantitative Finance project! This repository contains a fully automated, 4-part algorithmic trading and financial modeling suite built entirely in Python. 

As a Data Science student, my goal was to bridge the gap between academic data science and noisy, real-world financial markets. This suite handles everything from technical charting and strict backtesting to institutional-grade fundamental valuation and portfolio risk management.

## 🗂️ Project Structure

The suite is broken down into four independent modules:

### 1. `dashboard.py` — Technical Analysis & Execution Engine
A fully automated technical dashboard. It pulls real-time market data to calculate moving averages, RSI momentum, and candlestick rejection signals. It features a custom "IPO Decision Engine" that looks for specific alignments (like Golden Crosses) to trigger actionable BUY signals with precise Take-Profit and Stop-Loss limits.

### 2. `valuation.py` — Fundamental Valuation & DCF Model
This script acts like an institutional financial analyst. It bypasses basic screeners and scrapes raw EDGAR-equivalent accounting data (Income Statement, Balance Sheet, Cash Flow) directly via `yfinance`. It projects future cash flows, discounts them back to Present Value (DCF), audits earnings quality, and gates investments using a strict 30% Margin of Safety.

### 3. `backtest.py` — Quantitative Backtesting Simulator
Having a strategy is useless if you can't prove it works. This is a rigorous, vectorized backtesting environment. It runs strategies against 5 years of historical data, calculating institutional KPIs like the Sharpe Ratio, Calmar Ratio, and Maximum Drawdown. Crucially, it uses Walk-Forward Optimization (WFO) to split data into Train/Test cycles, strictly preventing look-ahead bias and overfitting.

### 4. `portfolio.py` — Modern Portfolio Theory & Risk Management
Predicting the market doesn't matter if your position sizing blows up your account. This script uses Monte Carlo simulations across 10,000 randomized portfolios to map the Efficient Frontier and find the optimal asset weighting. It calculates correlation matrices, portfolio beta, and uses the "1% Rule" to dynamically size positions based on strict absolute dollar-risk limits.

## 🛠️ Tech Stack
- **Language:** Python
- **Data Ingestion:** `yfinance`
- **Data Wrangling:** `pandas`, `numpy`
- **Visualization:** `plotly` (interactive web-based charts)

## 🚀 How to Run
Ensure you have the required dependencies installed:
```bash
pip install yfinance pandas numpy plotly
```
Then, simply run any of the modules directly from your terminal:
```bash
python dashboard.py
```
