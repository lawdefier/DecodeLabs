import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
CAPITAL = 100000.0
ASSETS = ['XLK', 'XLP', 'XLU', 'GLD']
BENCHMARK = 'SPY'
WEIGHTS = {'XLK': 0.20, 'XLP': 0.30, 'XLU': 0.25, 'GLD': 0.25}
STOPS = {'XLK': 0.10, 'XLP': 0.06, 'XLU': 0.07, 'GLD': 0.08}
def grab_data():
    print(f"\n[*] Fetching history for our basket: {ASSETS}...")
    df = yf.download(ASSETS + [BENCHMARK], period="2y", progress=False)['Close']
    return df.pct_change().dropna(), df.iloc[-1]
def run_mpt(returns):
    print("\n--- FINDING THE EFFICIENT FRONTIER ---")
    ret = returns[ASSETS]
    means, cov = ret.mean(), ret.cov()
    print("[*] Running 10,000 random portfolios through the Monte Carlo simulator...")
    results = np.zeros((3, 10000))
    all_weights = []
    for i in range(10000):
        w = np.random.random(len(ASSETS))
        w /= np.sum(w)                     
        all_weights.append(w)
        p_ret = np.sum(means * w) * 252
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)
        results[0,i] = p_ret
        results[1,i] = p_vol
        results[2,i] = (p_ret - 0.05) / p_vol          
    best_idx = np.argmax(results[2])
    best_w = dict(zip(ASSETS, np.round(all_weights[best_idx], 3)))
    print(f"Found the optimal setup! Max Sharpe: {results[2, best_idx]:.2f}")
    print(f"Optimal Weights: {best_w}")
    return results, best_idx
def analyze_risk(returns):
    print("\n--- SYSTEMIC RISK (BETA) ---")
    var_spy = returns[BENCHMARK].var()
    port_beta = 0
    for a in ASSETS:
        b = returns[[a, BENCHMARK]].cov().iloc[0,1] / var_spy
        port_beta += WEIGHTS[a] * b
    print(f"Total Portfolio Beta: {port_beta:.2f}")
    if port_beta < 1.0: print(" -> Good! We are less volatile than the broader market.")
    else: print(" -> Careful, this portfolio swings harder than the S&P 500.")
def size_positions(current_prices):
    print("\n--- POSITION SIZING (THE 1% RULE) ---")
    max_loss = CAPITAL * 0.01
    print(f"Rule: We refuse to lose more than ${max_loss:,.0f} on any single trade.\n")
    total_heat = 0
    for a in ASSETS:
        px = current_prices[a]
        stop_dist = px * STOPS[a]
        units = int(max_loss / stop_dist)
        allocated = units * px
        abs_risk = allocated * STOPS[a]
        total_heat += abs_risk
        print(f"{a}: Buy {units} shares at ${px:.2f}. Total cost: ${allocated:,.0f}. Max risk: ${abs_risk:,.0f}")
    print(f"\nTotal Portfolio Heat: {total_heat/CAPITAL * 100:.2f}%")
    if total_heat <= CAPITAL * 0.08:
        print("Verdict: ALL CLEAR. Deploy capital.")
    else:
        print("Verdict: TOO HOT. Scale down your sizes.")
def plot_mpt(results, best_idx):
    print("\n[*] Sending the Efficient Frontier chart to your browser...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results[1]*100, y=results[0]*100, mode='markers',
                             marker=dict(color=results[2], colorscale='Viridis', size=5, colorbar=dict(title='Sharpe')),
                             name='Random Portfolios'))
    fig.add_trace(go.Scatter(x=[results[1, best_idx]*100], y=[results[0, best_idx]*100], mode='markers',
                             marker=dict(color='red', symbol='star', size=15), name='Optimal Portfolio'))
    fig.update_layout(title="Modern Portfolio Theory: Efficient Frontier", xaxis_title="Volatility %", yaxis_title="Return %", template='plotly_dark')
    fig.show()
def main():
    print("Waking up the Risk Manager...")
    returns, prices = grab_data()
    mpt_res, best_idx = run_mpt(returns)
    analyze_risk(returns)
    size_positions(prices)
    plot_mpt(mpt_res, best_idx)
if __name__ == "__main__":
    main()