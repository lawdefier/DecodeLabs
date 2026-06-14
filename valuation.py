import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
TARGET = "AAPL"
PEERS = ["AAPL", "MSFT", "GOOGL"]
WACC = 0.099                       
TAX_RATE = 0.15                           
SCENARIOS = {
    "Baseline": {"cagr": 0.0639, "op_margin": 0.320, "exit_pe": 23.2},
    "Bull": {"cagr": 0.100, "op_margin": 0.330, "exit_pe": 30.0}
}
def get_value(df, keys, default=0):
    if df is None or df.empty: return default
    for k in keys:
        if k in df.index:
            val = df.loc[k].iloc[0]
            if pd.notna(val): return float(val)
    return default
def pull_financials(ticker_symbol):
    print(f"\n[*] Digging into the SEC filings for {ticker_symbol}...")
    stock = yf.Ticker(ticker_symbol)
    inc, bs, cf, info = stock.financials, stock.balance_sheet, stock.cashflow, stock.info
    data = {}
    data['Rev'] = get_value(inc, ['Total Revenue', 'Operating Revenue'])
    data['Prev_Rev'] = inc.loc['Total Revenue'].iloc[1] if ('Total Revenue' in inc.index and len(inc.columns) > 1) else data['Rev']
    data['Gross'] = get_value(inc, ['Gross Profit'])
    data['OpInc'] = get_value(inc, ['Operating Income'])
    data['NetInc'] = get_value(inc, ['Net Income'])
    data['EPS'] = info.get('trailingEps', 0)
    data['Assets'] = get_value(bs, ['Total Assets'])
    data['Liabilities'] = get_value(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
    data['Equity'] = get_value(bs, ['Stockholders Equity', 'Total Stockholder Equity'])
    data['Debt'] = get_value(bs, ['Total Debt'])
    data['Cash'] = get_value(bs, ['Cash, Cash Equivalents & Short Term Investments', 'Cash And Cash Equivalents'])
    data['OCF'] = get_value(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
    data['CapEx'] = abs(get_value(cf, ['Capital Expenditure']))
    data['FCF'] = data['OCF'] - data['CapEx']
    data['Price'] = info.get('currentPrice', info.get('previousClose', 0))
    data['Shares'] = info.get('sharesOutstanding', data['NetInc'] / data['EPS'] if data['EPS'] != 0 else 1)
    return data, inc, bs, cf, info
def show_snapshot(d):
    print("\n--- 1. FINANCIAL SNAPSHOT ---")
    print(f"Revenue:       ${d['Rev']:,.0f}")
    print(f"Net Income:    ${d['NetInc']:,.0f}")
    print(f"Free Cash Flow:${d['FCF']:,.0f}")
    print(f"Total Debt:    ${d['Debt']:,.0f}")
    print(f"Cash on Hand:  ${d['Cash']:,.0f}")
def check_margins(d):
    print("\n--- 2. PROFITABILITY CHECK ---")
    growth = ((d['Rev'] / d['Prev_Rev']) - 1) * 100 if d['Prev_Rev'] else 0
    print(f"YoY Growth:       {growth:.2f}%")
    print(f"Operating Margin: {(d['OpInc']/d['Rev'])*100:.2f}%")
    eq = d['OCF'] / d['NetInc'] if d['NetInc'] != 0 else 0
    print(f"Earnings Quality: {eq:.2f}x (OCF/Net Income)")
    if eq > 1.0: print(" -> Cash is king! Earnings are backed by real cash flow.")
    else: print(" -> Warning: Net income looks a bit artificially inflated.")
def balance_sheet_audit(d):
    print("\n--- 3. BALANCE SHEET AUDIT ---")
    de_ratio = d['Debt'] / d['Equity'] if d['Equity'] > 0 else 0
    net_cash = d['Cash'] - d['Debt']
    print(f"D/E Ratio: {de_ratio:.2f}")
    print(f"Net Cash:  ${net_cash:,.0f}")
    if de_ratio > 1.5:
        print(" -> Note: High leverage. For mature tech, this is often just aggressive buybacks, but keep an eye on it.")
    else:
        print(" -> Note: Balance sheet looks pretty clean and healthy.")
def run_dcf(d):
    print("\n--- 4. DISCOUNTED CASH FLOW (DCF) MODEL ---")
    rev, shares = d['Rev'], d['Shares']
    results = {}
    for name, p in SCENARIOS.items():
        pv_fcf = 0
        proj_rev = rev
        for year in range(1, 6):
            proj_rev *= (1 + p['cagr'])
            nopat = (proj_rev * p['op_margin']) * (1 - TAX_RATE)
            pv_fcf += nopat / ((1 + WACC) ** year)
            if year == 5: year5_nopat = nopat
        tv = year5_nopat * p['exit_pe']
        pv_tv = tv / ((1 + WACC) ** 5)
        eq_val = pv_fcf + pv_tv + d['Cash'] - d['Debt']
        value_per_share = eq_val / shares
        results[name] = value_per_share
        print(f"{name} Scenario -> Intrinsic Value: ${value_per_share:.2f}/share")
    return results
def gatekeeper(base_val, price):
    print("\n--- 5. MARGIN OF SAFETY ---")
    safe_entry = base_val * 0.70
    premium = ((price / base_val) - 1) * 100
    print(f"Current Price:  ${price:.2f}")
    print(f"Max Buy Price:  ${safe_entry:.2f} (Applying a 30% safety buffer)")
    if price <= safe_entry:
        print("Verdict: BUY 🚀 - It's trading at a massive discount.")
    elif price <= base_val:
        print("Verdict: HOLD ✋ - It's fairly valued, but no margin of safety.")
    else:
        print(f"Verdict: SELL 📉 - Trading at a {premium:.1f}% premium. Too expensive.")
    return safe_entry
def peer_comp():
    print("\n--- 6. PEER COMPARISON ---")
    matrix = []
    for p in PEERS:
        t = yf.Ticker(p)
        matrix.append({
            "Ticker": p,
            "Price": f"${t.info.get('currentPrice', 0):.2f}",
            "P/E": f"{t.info.get('trailingPE', 0):.1f}x"
        })
    print(pd.DataFrame(matrix).to_string(index=False))
    print("\nInsight: AAPL usually trades at a premium due to buybacks, while GOOGL looks cheaper structurally.")
def plot_pe_history(ticker, inc):
    print("\n--- 7. HISTORICAL P/E CHECK ---")
    try:
        dates = inc.columns[:5]
        data = []
        tkr = yf.Ticker(ticker)
        for d in dates:
            start, end = d - pd.Timedelta(days=5), d + pd.Timedelta(days=5)
            hist = tkr.history(start=start, end=end)
            if not hist.empty:
                data.append({"Date": d, "NI": inc.loc['Net Income', d], "Price": hist['Close'].iloc[-1]})
        df = pd.DataFrame(data).sort_values("Date")
        df['NI_Idx'] = (df['NI'] / df['NI'].iloc[0]) * 100
        df['Px_Idx'] = (df['Price'] / df['Price'].iloc[0]) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Px_Idx'], name='Price Growth', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['NI_Idx'], name='Net Income Growth', line=dict(color='orange', dash='dot')))
        fig.update_layout(title=f"{ticker} Multiple Expansion Risk", template='plotly_dark')
        fig.show()
    except Exception as e:
        print("Not enough history for the chart.")
def main():
    print("Firing up the Valuation Engine...\n")
    d, inc, bs, cf, info = pull_financials(TARGET)
    show_snapshot(d)
    check_margins(d)
    balance_sheet_audit(d)
    dcf_vals = run_dcf(d)
    gatekeeper(dcf_vals['Baseline'], d['Price'])
    peer_comp()
    plot_pe_history(TARGET, inc)
    print("\nDone! Check your browser for the chart.")
if __name__ == "__main__":
    main()