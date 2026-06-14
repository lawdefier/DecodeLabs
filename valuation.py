"""
Fundamental Valuation & Financial Modeling Tool
Live Asset: Apple Inc. (AAPL)
Provides DCF Modeling, Margin of Safety, Peer Comparison, and Historical P/E Backtesting.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- CONFIGURATION ---
TARGET_TICKER = "AAPL"
PEERS = ["AAPL", "MSFT", "GOOGL"]
WACC = 0.099
TAX_RATE = 0.15 # Approx effective tax rate for AAPL
SHARES_OUTSTANDING = None # Will fetch dynamically
CURRENT_PRICE = None # Will fetch dynamically

# DCF Scenarios
SCENARIOS = {
    "Baseline": {
        "cagr": 0.0639,
        "op_margin": 0.320,
        "exit_pe": 23.2
    },
    "Bull": {
        "cagr": 0.100,
        "op_margin": 0.330,
        "exit_pe": 30.0
    }
}

def safe_get(df, keys, default=0):
    """Safely extracts the first available key from a yfinance dataframe index."""
    if df is None or df.empty: return default
    for key in keys:
        if key in df.index:
            val = df.loc[key].iloc[0]
            if pd.notna(val):
                return float(val)
    return default

def fetch_data(ticker_symbol):
    print(f"\n[*] Fetching EDGAR-equivalent financial data for {ticker_symbol} from yfinance...")
    ticker = yf.Ticker(ticker_symbol)
    
    inc = ticker.financials
    bs = ticker.balance_sheet
    cf = ticker.cashflow
    info = ticker.info
    
    # 1. DATA SOURCE & EXTRACTION
    data = {}
    
    # Income Statement
    data['Revenue'] = safe_get(inc, ['Total Revenue', 'Operating Revenue'])
    data['Prev_Revenue'] = inc.loc['Total Revenue'].iloc[1] if ('Total Revenue' in inc.index and len(inc.columns) > 1) else data['Revenue']
    data['Gross Profit'] = safe_get(inc, ['Gross Profit'])
    data['Operating Income'] = safe_get(inc, ['Operating Income'])
    data['Net Income'] = safe_get(inc, ['Net Income'])
    data['Diluted EPS'] = info.get('trailingEps', 0)
    
    # Balance Sheet
    data['Total Assets'] = safe_get(bs, ['Total Assets'])
    data['Total Liabilities'] = safe_get(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
    data['Shareholders Equity'] = safe_get(bs, ['Stockholders Equity', 'Total Stockholder Equity'])
    data['Total Debt'] = safe_get(bs, ['Total Debt'])
    data['Cash'] = safe_get(bs, ['Cash, Cash Equivalents & Short Term Investments', 'Cash And Cash Equivalents', 'Cash Financial'])
    
    # Cash Flow
    data['Operating Cash Flow'] = safe_get(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
    capex = safe_get(cf, ['Capital Expenditure'])
    data['CapEx'] = abs(capex) if capex != 0 else 0
    
    # FCF Calculation
    data['Free Cash Flow'] = data['Operating Cash Flow'] - data['CapEx']
    
    # Market Data
    data['Price'] = info.get('currentPrice', info.get('previousClose', 0))
    data['Shares Outstanding'] = info.get('sharesOutstanding', data['Net Income'] / data['Diluted EPS'] if data['Diluted EPS'] != 0 else 1)
    
    return data, inc, bs, cf, info

def print_snapshot(data):
    print("\n" + "="*50)
    print("1. FINANCIAL SNAPSHOT (TTM / LATEST ANNUAL)")
    print("="*50)
    print(f"Revenue:              ${data['Revenue']:,.0f}")
    print(f"Gross Profit:         ${data['Gross Profit']:,.0f}")
    print(f"Operating Income:     ${data['Operating Income']:,.0f}")
    print(f"Net Income:           ${data['Net Income']:,.0f}")
    print(f"Diluted EPS:          ${data['Diluted EPS']:.2f}")
    print("-" * 30)
    print(f"Total Assets:         ${data['Total Assets']:,.0f}")
    print(f"Total Liabilities:    ${data['Total Liabilities']:,.0f}")
    print(f"Shareholders Equity:  ${data['Shareholders Equity']:,.0f}")
    print(f"Total Debt:           ${data['Total Debt']:,.0f}")
    print(f"Cash & Equivalents:   ${data['Cash']:,.0f}")
    print("-" * 30)
    print(f"Operating Cash Flow:  ${data['Operating Cash Flow']:,.0f}")
    print(f"CapEx:                ${data['CapEx']:,.0f}")
    print(f"Free Cash Flow:       ${data['Free Cash Flow']:,.0f}")

def analyze_income_statement(data):
    print("\n" + "="*50)
    print("2. INCOME STATEMENT ANALYSIS")
    print("="*50)
    
    rev_growth = ((data['Revenue'] / data['Prev_Revenue']) - 1) * 100 if data['Prev_Revenue'] else 0
    gross_margin = (data['Gross Profit'] / data['Revenue']) * 100
    op_margin = (data['Operating Income'] / data['Revenue']) * 100
    net_margin = (data['Net Income'] / data['Revenue']) * 100
    
    print(f"Revenue YoY Growth:   {rev_growth:.2f}%")
    print(f"Gross Margin:         {gross_margin:.2f}%")
    print(f"Operating Margin:     {op_margin:.2f}%")
    print(f"Net Margin:           {net_margin:.2f}%")
    
    # Earnings Quality Signal
    eq_ratio = data['Operating Cash Flow'] / data['Net Income'] if data['Net Income'] != 0 else 0
    print(f"Earnings Quality:     {eq_ratio:.2f}x (OCF / Net Income)")
    if eq_ratio > 1.0:
        print("  -> SIGNAL: High Quality. Cash generation exceeds accounting profits.")
    else:
        print("  -> SIGNAL: Warning. Net income is not translating fully into cash flow.")

def audit_balance_sheet(data):
    print("\n" + "="*50)
    print("3. BALANCE SHEET AUDIT")
    print("="*50)
    
    de_ratio = data['Total Debt'] / data['Shareholders Equity'] if data['Shareholders Equity'] > 0 else 0
    net_cash = data['Cash'] - data['Total Debt']
    
    print(f"D/E Ratio:            {de_ratio:.2f}")
    print(f"Net Cash:             ${net_cash:,.0f}")
    
    print("\nAnalyst Translation:")
    if de_ratio > 1.5:
        print(f"  -> High D/E Ratio ({de_ratio:.2f}). For mature tech hardware, this often reflects aggressive stock buybacks shrinking equity, rather than severe credit risk, assuming cash flow covers interest.")
    else:
        print(f"  -> Healthy D/E Ratio ({de_ratio:.2f}). Suggests a conservatively leveraged capital structure.")
        
    if net_cash > 0:
        print("  -> FLAG: POSITIVE Net Cash. Fortress balance sheet.")
    else:
        print("  -> FLAG: NEGATIVE Net Cash. Company uses leverage to fund operations or shareholder returns.")

def run_dcf(data):
    print("\n" + "="*50)
    print("4. DCF MODEL (INTRINSIC VALUE ENGINE)")
    print("="*50)
    
    current_rev = data['Revenue']
    shares = data['Shares Outstanding']
    
    results = {}
    
    for name, params in SCENARIOS.items():
        cagr = params['cagr']
        op_margin = params['op_margin']
        exit_pe = params['exit_pe']
        
        pv_fcf = 0
        proj_rev = current_rev
        
        # 5 Year Projection
        for year in range(1, 6):
            proj_rev *= (1 + cagr)
            proj_op_inc = proj_rev * op_margin
            # FCF proxy = NOPAT (Net Operating Profit After Tax)
            proj_fcf = proj_op_inc * (1 - TAX_RATE)
            
            # Discount to PV
            discount_factor = (1 + WACC) ** year
            pv_fcf += proj_fcf / discount_factor
            
            # Keep year 5 net income for Terminal Value calculation via Exit P/E
            if year == 5:
                year_5_net_income = proj_fcf # Proxying Net income with NOPAT for simplicity
        
        # Terminal Value
        terminal_value = year_5_net_income * exit_pe
        pv_tv = terminal_value / ((1 + WACC) ** 5)
        
        # Enterprise / Equity Value
        enterprise_value = pv_fcf + pv_tv
        # Equity value = EV + Cash - Debt
        equity_value = enterprise_value + data['Cash'] - data['Total Debt']
        
        intrinsic_value_per_share = equity_value / shares
        results[name] = intrinsic_value_per_share
        
        print(f"\n{name.upper()} SCENARIO:")
        print(f"  Revenue CAGR:     {cagr*100:.2f}%")
        print(f"  Operating Margin: {op_margin*100:.1f}%")
        print(f"  Exit P/E:         {exit_pe}x")
        print(f"  -> Intrinsic Value: ${intrinsic_value_per_share:.2f} per share")

    return results

def margin_of_safety(baseline_value, current_price):
    print("\n" + "="*50)
    print("5. MARGIN OF SAFETY GATEKEEPER")
    print("="*50)
    
    max_entry = baseline_value * 0.70
    discount_premium = ((current_price / baseline_value) - 1) * 100
    
    print(f"Current Market Price: ${current_price:.2f}")
    print(f"Baseline Intrinsic:   ${baseline_value:.2f}")
    print(f"Max Entry Price (30% Margin of Safety): ${max_entry:.2f}")
    
    if discount_premium > 0:
        print(f"Valuation Premium:    +{discount_premium:.1f}%")
    else:
        print(f"Valuation Discount:   {discount_premium:.1f}%")
        
    print("\nVERDICT:")
    if current_price <= max_entry:
        print(">>> BUY <<<")
        print("Rationale: The stock is trading below our strict 30% margin of safety threshold. Exceptional risk/reward entry.")
    elif current_price <= baseline_value:
        print(">>> HOLD <<<")
        print("Rationale: The stock is fairly valued. While below intrinsic value, it does not offer our required 30% margin of safety for a new purchase.")
    else:
        print(">>> SELL <<<")
        print(f"Rationale: The stock is trading at a {discount_premium:.1f}% premium to our baseline DCF. Fundamental upside is constrained.")
        
    return max_entry

def peer_comparison():
    print("\n" + "="*50)
    print("6. PEER COMPARISON MATRIX")
    print("="*50)
    
    matrix = []
    for peer in PEERS:
        t = yf.Ticker(peer)
        inf = t.info
        bs = t.balance_sheet
        
        price = inf.get('currentPrice', inf.get('previousClose', 0))
        eps = inf.get('trailingEps', 0)
        pe = inf.get('trailingPE', 0)
        
        equity = safe_get(bs, ['Stockholders Equity', 'Total Stockholder Equity'])
        debt = safe_get(bs, ['Total Debt'])
        de = debt / equity if equity > 0 else 0
        
        matrix.append({
            "Ticker": peer,
            "Price": f"${price:.2f}",
            "EPS (TTM)": f"${eps:.2f}",
            "P/E": f"{pe:.1f}x",
            "D/E Ratio": f"{de:.2f}"
        })
        
    df_peers = pd.DataFrame(matrix)
    print(df_peers.to_string(index=False))
    
    print("\nAnalyst Insight:")
    print("In this peer group, GOOGL generally trades at a structural P/E discount to AAPL and MSFT due to AI search concerns, whereas MSFT commands a premium for cloud and AI growth. AAPL relies heavily on multiple expansion and buybacks to drive EPS, masking slower top-line growth.")

def historical_pe_backtest(ticker_symbol, inc_df):
    print("\n" + "="*50)
    print("7. HISTORICAL P/E BACKTEST")
    print("="*50)
    
    ticker = yf.Ticker(ticker_symbol)
    
    # We want to approximate historical P/E by grabbing historical prices around the financial reporting dates.
    dates = inc_df.columns[:5] # Last up to 5 years
    
    history_data = []
    for dt in dates:
        try:
            # Net Income
            ni = inc_df.loc['Net Income', dt]
            
            # To get historical EPS, we need shares. If not easily accessible, we approximate:
            # Let's fetch historical prices.
            start_date = dt - pd.Timedelta(days=5)
            end_date = dt + pd.Timedelta(days=5)
            hist = ticker.history(start=start_date, end=end_date)
            if hist.empty:
                continue
                
            price = hist['Close'].iloc[-1]
            
            history_data.append({
                "Date": dt,
                "Net Income": ni,
                "Price": price
            })
        except:
            continue
            
    if not history_data:
        print("Insufficient historical data for backtest.")
        return None
        
    df_hist = pd.DataFrame(history_data)
    
    # Sort chronologically
    df_hist = df_hist.sort_values(by="Date")
    
    # Calculate relative growth to plot against multiple
    # Note: Accurately calculating historical P/E from scratch needs exact share counts. 
    # We will use Net Income and Market Cap proxies if needed, but for plotting, 
    # we'll plot Price & Net Income normalized to show multiple expansion.
    
    # Normalize to base 100 for comparison
    base_ni = df_hist['Net Income'].iloc[0]
    base_px = df_hist['Price'].iloc[0]
    
    df_hist['NI_Index'] = (df_hist['Net Income'] / base_ni) * 100
    df_hist['Price_Index'] = (df_hist['Price'] / base_px) * 100
    
    current_pe = ticker.info.get('trailingPE', 0)
    
    print(f"Current P/E is {current_pe:.1f}x.")
    print("Vulnerability Assessment:")
    if current_pe > 25:
        print("  -> YES. The stock is exposed to severe multiple contraction risk. Historical mean reversion could drag the price down even if Net Income remains stable.")
    else:
        print("  -> MODERATE. Valuation multiple is relatively anchored, reducing risk of pure P/E contraction.")
        
    return df_hist

def plot_backtest(df_hist, ticker_symbol):
    """Plots the historical P/E backtest in browser."""
    if df_hist is None or df_hist.empty: return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['Price_Index'],
                             mode='lines+markers', name='Stock Price (Indexed)',
                             line=dict(color='cyan', width=3)))
                             
    fig.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['NI_Index'],
                             mode='lines+markers', name='Net Income (Indexed)',
                             line=dict(color='orange', width=3, dash='dot')))
                             
    fig.update_layout(title=f"{ticker_symbol} Multiple Expansion Backtest (Price vs. Net Income Growth)",
                      template='plotly_dark',
                      yaxis_title="Indexed Growth (Base=100)",
                      xaxis_title="Fiscal Year")
                      
    # Will open in browser
    fig.show()

def institutional_report(data, baseline_val, bull_val, max_entry):
    print("\n" + "="*50)
    print("8. FINAL INSTITUTIONAL REPORT")
    print("="*50)
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    current_price = data['Price']
    pe = data['Price'] / data['Diluted EPS'] if data['Diluted EPS'] > 0 else 0
    
    print(f"Reference Date:        {date_str}")
    print(f"Asset:                 {TARGET_TICKER}")
    print(f"Closing Price:         ${current_price:.2f}")
    print(f"P/E Ratio (TTM):       {pe:.1f}x")
    print(f"Baseline Intrinsic:    ${baseline_val:.2f}")
    print(f"Bull Case Intrinsic:   ${bull_val:.2f}")
    print(f"Max Purchase Price:    ${max_entry:.2f} (30% Margin of Safety)")
    
    print("\nFINAL VERDICT:")
    if current_price <= max_entry:
        print("STRONG BUY.")
        print("Rationale: Asset trades at a severe discount to conservative free cash flow projections, satisfying our required 30% margin of safety.")
    elif current_price <= baseline_val:
        print("HOLD.")
        print("Rationale: Asset is fairly valued under baseline DCF modeling. Existing holders should maintain positions, but lack of margin of safety precludes new capital deployment.")
    else:
        print("SELL.")
        print("Rationale: Current market pricing demands aggressive, perpetual growth assumptions (Bull Case) simply to break even on a present-value basis. The structural premium creates an asymmetric downside profile susceptible to multiple contraction.")

def main():
    print("Initializing Fundamental Valuation Engine...")
    
    # 1
    data, inc_df, bs_df, cf_df, info = fetch_data(TARGET_TICKER)
    print_snapshot(data)
    
    # 2
    analyze_income_statement(data)
    
    # 3
    audit_balance_sheet(data)
    
    # 4
    dcf_results = run_dcf(data)
    
    # 5
    baseline_val = dcf_results["Baseline"]
    bull_val = dcf_results["Bull"]
    max_entry = margin_of_safety(baseline_val, data['Price'])
    
    # 6
    peer_comparison()
    
    # 7
    df_hist = historical_pe_backtest(TARGET_TICKER, inc_df)
    
    # 8
    institutional_report(data, baseline_val, bull_val, max_entry)
    
    # Render chart
    plot_backtest(df_hist, TARGET_TICKER)

if __name__ == "__main__":
    main()
