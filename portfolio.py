"""
Portfolio Optimization & Risk Management System
Capital Base: $100,000
Assets: XLK, XLP, XLU, GLD (Benchmark: SPY)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CAPITAL = 100000.0
ASSETS = ['XLK', 'XLP', 'XLU', 'GLD']
BENCHMARK = 'SPY'
RISK_FREE_RATE = 0.05

# Allocation Table Rules
ALLOCATION_WEIGHTS = {'XLK': 0.20, 'XLP': 0.30, 'XLU': 0.25, 'GLD': 0.25}
STOP_LOSSES = {'XLK': 0.10, 'XLP': 0.06, 'XLU': 0.07, 'GLD': 0.08}

def fetch_data():
    print("="*50)
    print("1. ASSET UNIVERSE (CORRELATION SHIELD)")
    print("="*50)
    
    tickers = ASSETS + [BENCHMARK]
    print(f"[*] Fetching 2 years of daily data for: {', '.join(tickers)}")
    
    # Download 2 years of data
    df = yf.download(tickers, period="2y", progress=False)['Close']
    
    # Calculate daily returns
    returns = df.pct_change().dropna()
    
    return df, returns

def correlation_analysis(returns):
    print("\n" + "="*50)
    print("2. CORRELATION ANALYSIS")
    print("="*50)
    
    # Isolate portfolio assets
    asset_returns = returns[ASSETS]
    corr_matrix = asset_returns.corr()
    
    print("Daily Returns Correlation Matrix:")
    print(corr_matrix.round(3))
    
    print("\nAnalyst Note:")
    print("Notice the correlation between XLK (Tech) and GLD (Gold). True diversification requires assets that behave differently under market stress. Pseudo-diversification (e.g., holding 10 different tech stocks) fails because their correlation approaches 1.0; they will all crash together. XLP, XLU, and GLD provide non-correlated defensive ballast against XLK's cyclical beta.")
    
    # Heatmap Plot (handled later in main render function to combine plots)
    return corr_matrix

def run_mpt_monte_carlo(returns):
    print("\n" + "="*50)
    print("3. MODERN PORTFOLIO THEORY (MPT)")
    print("="*50)
    
    asset_returns = returns[ASSETS]
    mean_daily_returns = asset_returns.mean()
    cov_matrix = asset_returns.cov()
    
    num_portfolios = 10000
    print(f"[*] Running Monte Carlo simulation with {num_portfolios} random portfolios...")
    
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        # Generate random weights that sum to 1
        weights = np.random.random(len(ASSETS))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Expected Return
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        
        # Portfolio Volatility
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_std_dev
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i] = sharpe_ratio
        
    # Find portfolio with max Sharpe Ratio
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_weights = weights_record[max_sharpe_idx]
    
    print("-> Max Sharpe Ratio Found:")
    print(f"   Expected Return: {results[0, max_sharpe_idx]*100:.2f}%")
    print(f"   Volatility:      {results[1, max_sharpe_idx]*100:.2f}%")
    print(f"   Sharpe Ratio:    {results[2, max_sharpe_idx]:.2f}")
    print(f"   Optimal Weights: {dict(zip(ASSETS, np.round(max_sharpe_weights, 3)))}")
    
    return results, weights_record, max_sharpe_idx

def calculate_beta(returns):
    print("\n" + "="*50)
    print("4. SYSTEMIC RISK MEASUREMENT (BETA CALCULATION)")
    print("="*50)
    
    var_spy = returns[BENCHMARK].var()
    
    betas = {}
    print("Individual Betas (vs SPY):")
    for asset in ASSETS:
        cov_matrix = returns[[asset, BENCHMARK]].cov()
        cov_asset_spy = cov_matrix.iloc[0,1]
        beta = cov_asset_spy / var_spy
        betas[asset] = beta
        print(f"  {asset}: {beta:.2f}")
        
    # Weighted Portfolio Beta using fixed allocation
    weighted_beta = sum(ALLOCATION_WEIGHTS[asset] * betas[asset] for asset in ASSETS)
    print(f"\nWeighted Portfolio Beta: {weighted_beta:.2f}")
    
    print("GATEKEEPER CHECK: Beta_p < 1.0")
    if weighted_beta < 1.0:
        print("  -> PASSED. Portfolio is less volatile than the broader market.")
    else:
        print("  -> FAILED. Portfolio carries excess systemic risk.")
        
    return weighted_beta

def master_allocation():
    print("\n" + "="*50)
    print("5. MASTER ALLOCATION TABLE ($100k)")
    print("="*50)
    
    print("The Fortified Structure:")
    print(f"{'Sector':<6} | {'Weight':<6} | {'Allocated $':<11} | {'Stop-Loss %':<11} | {'Absolute Risk $':<15}")
    print("-" * 65)
    
    total_absolute_risk = 0
    
    for asset in ASSETS:
        weight = ALLOCATION_WEIGHTS[asset]
        allocated = CAPITAL * weight
        sl_pct = STOP_LOSSES[asset]
        abs_risk = allocated * sl_pct
        total_absolute_risk += abs_risk
        
        print(f"{asset:<6} | {weight*100:>.0f}%    | ${allocated:>9,.0f} | -{sl_pct*100:>.0f}%        | ${abs_risk:>10,.0f}")
        
    print("-" * 65)
    print(f"Total Capital:      ${CAPITAL:,.0f}")
    print(f"Total Absolute Risk: ${total_absolute_risk:,.0f}")
    
    return total_absolute_risk

def position_sizing_funnel(df_current_prices):
    print("\n" + "="*50)
    print("6. POSITION SIZING FUNNEL (1% RULE)")
    print("="*50)
    
    # 1% Rule dictates absolute max risk per trade is $1,000
    dollar_risk = CAPITAL * 0.01 
    print(f"Strict Dollar Risk Per Trade: ${dollar_risk:,.0f} (1% of Total Capital)")
    
    print(f"\n{'Sector':<6} | {'Current Price':<14} | {'Stop Dist ($)':<14} | {'Units to Buy':<12} | {'Actual Capital Required':<25}")
    print("-" * 80)
    
    for asset in ASSETS:
        price = df_current_prices[asset]
        sl_pct = STOP_LOSSES[asset]
        
        stop_distance = price * sl_pct
        units = int(dollar_risk / stop_distance)
        cap_req = units * price
        
        print(f"{asset:<6} | ${price:>12.2f} | ${stop_distance:>12.2f} | {units:>12} | ${cap_req:>20,.2f}")

def portfolio_heat_check(total_absolute_risk):
    print("\n" + "="*50)
    print("7. PORTFOLIO HEAT CHECK (GATEKEEPER CONSTRAINT 2)")
    print("="*50)
    
    heat_pct = (total_absolute_risk / CAPITAL) * 100
    max_heat_pct = 8.0
    
    print(f"Total Portfolio Heat: {heat_pct:.2f}% (${total_absolute_risk:,.0f})")
    print(f"Max Allowable Heat:   {max_heat_pct:.2f}% (${CAPITAL * (max_heat_pct/100):,.0f})")
    
    print("\nVERDICT:")
    if heat_pct <= max_heat_pct:
        print("-> PASSED. Total heat is within structural limits.")
    else:
        print("-> FAILED. Total absolute risk is too high. Rebalancing or tightening stop losses required.")

def survival_math_demonstration():
    print("\n" + "="*50)
    print("8. SURVIVAL MATH DEMONSTRATION")
    print("="*50)
    
    print("Multiplicative Decay: Remaining Capital after Consecutive Losses")
    print(f"{'Consecutive Losses':<20} | {'At 1% Risk':<15} | {'At 5% Risk':<15}")
    print("-" * 55)
    
    scenarios = [5, 10, 20]
    
    for n in scenarios:
        rem_1 = CAPITAL * ((1 - 0.01) ** n)
        rem_5 = CAPITAL * ((1 - 0.05) ** n)
        
        print(f"{n:<20} | ${rem_1:>10,.0f}    | ${rem_5:>10,.0f}")
        
    print("\nAnalyst Note:")
    print("At 5% risk, just 20 losses destroys ~64% of your account, requiring a +178% return just to break even. At 1% risk, you survive to fight another day.")

def final_dashboard(weighted_beta, total_absolute_risk, max_sharpe_weights):
    print("\n" + "="*50)
    print("9. FINAL DASHBOARD OUTPUT")
    print("="*50)
    
    # Calculate max single risk from Master Allocation
    max_single_risk_usd = max([CAPITAL * ALLOCATION_WEIGHTS[a] * STOP_LOSSES[a] for a in ASSETS])
    max_single_risk_pct = (max_single_risk_usd / CAPITAL) * 100
    
    heat_pct = (total_absolute_risk / CAPITAL) * 100
    
    # Gatekeeper logic
    beta_pass = weighted_beta < 1.0
    msr_pass = max_single_risk_pct <= 2.0
    heat_pass = heat_pct <= 8.0
    
    ready = beta_pass and msr_pass and heat_pass
    
    print(f"Portfolio Beta:       {weighted_beta:.2f} — {'PASSED' if beta_pass else 'FAILED'} (< 1.0)")
    print(f"Max Single Risk:      {max_single_risk_pct:.2f}% — {'PASSED' if msr_pass else 'FAILED'} (<= 2%)")
    print(f"Total Portfolio Heat: {heat_pct:.2f}% — {'PASSED' if heat_pass else 'FAILED'} (<= 8%)")
    
    print("\nOptimal MPT Weights (for reference vs fixed allocation):")
    optimal_dict = dict(zip(ASSETS, np.round(max_sharpe_weights, 3)))
    for a, w in optimal_dict.items():
        print(f"  {a}: {w*100:.1f}%")
        
    print("\nSYSTEM VERDICT:")
    if ready:
        print(">>> PORTFOLIO DEPLOYMENT READY <<<")
        print("All structural risk metrics are within strict institutional boundaries.")
    else:
        print(">>> REVISION REQUIRED <<<")
        print("One or more gatekeeper constraints failed. Adjust allocations or stop losses.")
    print("==================================================\n")

def render_charts(corr_matrix, mpt_results, max_sharpe_idx):
    print("[*] Rendering Correlation Heatmap and Efficient Frontier in browser...")
    
    # Correlation Heatmap
    fig1 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1, zmax=1))
    fig1.update_layout(title="Asset Correlation Heatmap (Daily Returns)", template='plotly_dark')
    fig1.show()
    
    # Efficient Frontier
    returns_arr = mpt_results[0,:] * 100
    vol_arr = mpt_results[1,:] * 100
    sharpe_arr = mpt_results[2,:]
    
    fig2 = go.Figure()
    
    # Plot all 10k portfolios
    fig2.add_trace(go.Scatter(
        x=vol_arr,
        y=returns_arr,
        mode='markers',
        marker=dict(
            color=sharpe_arr,
            colorscale='Viridis',
            showscale=True,
            size=5,
            colorbar=dict(title='Sharpe Ratio')
        ),
        name='Random Portfolios'
    ))
    
    # Mark Max Sharpe
    fig2.add_trace(go.Scatter(
        x=[vol_arr[max_sharpe_idx]],
        y=[returns_arr[max_sharpe_idx]],
        mode='markers',
        marker=dict(color='red', symbol='star', size=15, line=dict(color='white', width=1)),
        name='Max Sharpe Ratio'
    ))
    
    fig2.update_layout(title="Modern Portfolio Theory: Efficient Frontier (10,000 Portfolios)",
                       xaxis_title="Volatility (Risk) %",
                       yaxis_title="Expected Annual Return %",
                       template='plotly_dark')
    fig2.show()

def main():
    print("Initializing Portfolio Optimization & Risk Management System...\n")
    
    df, returns = fetch_data()
    
    # 2
    corr_matrix = correlation_analysis(returns)
    
    # 3
    mpt_results, mpt_weights, max_idx = run_mpt_monte_carlo(returns)
    
    # 4
    weighted_beta = calculate_beta(returns)
    
    # 5
    total_abs_risk = master_allocation()
    
    # 6
    # Get current (last available) prices
    current_prices = df[ASSETS].iloc[-1]
    position_sizing_funnel(current_prices)
    
    # 7
    portfolio_heat_check(total_abs_risk)
    
    # 8
    survival_math_demonstration()
    
    # 9
    final_dashboard(weighted_beta, total_abs_risk, mpt_weights[max_idx])
    
    # Render Interactive Charts
    render_charts(corr_matrix, mpt_results, max_idx)

if __name__ == "__main__":
    main()
