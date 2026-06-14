"""
Algorithmic Trading Backtesting Engine
Asset: SPY (S&P 500)
Strategy: Moving Average Crossover with RSI Filter and strict TP/SL execution.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIG ---
TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"
RISK_FREE_RATE = 0.05

def get_data():
    print("="*50)
    print("1. DATA LAYER")
    print("="*50)
    print(f"[*] Fetching {TICKER} data from {START_DATE} to {END_DATE}...")
    
    # Download data
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    # Flatten multi-index if yfinance returns it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Forward-fill missing values (strictly no bfill to avoid look-ahead bias)
    df.ffill(inplace=True)
    
    print(f"[*] Data Shape: {df.shape}")
    missing = df.isna().sum().sum()
    print(f"[*] Missing Values Remaining: {missing}")
    
    # Strict validation
    assert missing == 0, "CRITICAL: NaN values detected after ffill!"
    
    return df

def feature_engineering(df):
    print("\n" + "="*50)
    print("2. FEATURE ENGINEERING")
    print("="*50)
    print("[*] Calculating Fast MA (10), Slow MA (50), and RSI (14)...")
    
    # Simple Moving Averages
    df['Fast_MA'] = df['Close'].rolling(window=10).mean()
    df['Slow_MA'] = df['Close'].rolling(window=50).mean()
    
    # RSI (Manual calculation as requested)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Using simple moving average of gains/losses over 14 periods
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # RS and RSI
    rs = avg_gain / avg_loss.replace(0, 1e-10) # Prevent div by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop the initial rows that have NaN values due to the 50-day lookback window
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"[*] Dropped {initial_len - len(df)} rows used for indicator warm-up. Clean rows: {len(df)}")
    
    return df

def run_logic_and_simulate(df):
    print("\n" + "="*50)
    print("3 & 4. ENTRY/EXIT LOGIC & BACKTEST SIMULATION")
    print("="*50)
    
    position = 0
    entry_price = 0.0
    entry_date = None
    
    stance = np.zeros(len(df))
    buy_signals = []
    sell_signals = []
    trade_log = []
    
    # --- LOGIC MATRIX ---
    for i in range(len(df)):
        row = df.iloc[i]
        close = row['Close']
        fast = row['Fast_MA']
        slow = row['Slow_MA']
        rsi = row['RSI']
        date = df.index[i]
        
        # 1. Check Exits first if we are currently Long
        if position == 1:
            unrealized_return = (close - entry_price) / entry_price
            exit_reason = None
            
            # Exit: Take Profit
            if unrealized_return >= 0.06:
                exit_reason = "Take Profit"
            # Exit: Stop Loss
            elif unrealized_return <= -0.02:
                exit_reason = "Stop Loss"
            # Exit: Signal Reversal
            elif fast < slow:
                exit_reason = "Signal Reversal"
                
            # If an exit condition was met, execute the SELL
            if exit_reason:
                position = 0
                sell_signals.append((date, close))
                
                # Format dates for clean printing
                e_date_str = entry_date.strftime('%Y-%m-%d')
                x_date_str = date.strftime('%Y-%m-%d')
                
                trade_log.append({
                    'Entry Date': e_date_str,
                    'Exit Date': x_date_str,
                    'Entry Price': round(entry_price, 2),
                    'Exit Price': round(close, 2),
                    'Return %': round(unrealized_return * 100, 2),
                    'Exit Reason': exit_reason
                })
                
                # Reset trade tracking variables
                entry_price = 0.0
                entry_date = None
                
        # 2. Check Entries if we are currently Flat
        if position == 0:
            # Entry Signal (Go Long)
            if fast > slow and rsi > 50:
                position = 1
                entry_price = close
                entry_date = date
                buy_signals.append((date, close))
                
        # Record the stance for the day
        stance[i] = position
        
    # --- SIMULATION ---
    # We copy the dataframe to safely append our new columns
    res_df = df.copy()
    res_df['Stance'] = stance
    
    # Calculate daily market return
    res_df['Market_Return'] = res_df['Close'].pct_change()
    
    # Strategy Return = Market Return * Stance of the PREVIOUS day (prevents look-ahead bias)
    res_df['Strategy_Return'] = res_df['Market_Return'] * res_df['Stance'].shift(1)
    res_df.fillna(0, inplace=True)
    
    # Equity Curves using cumulative log returns
    res_df['Log_Market'] = np.log1p(res_df['Market_Return'])
    res_df['Log_Strategy'] = np.log1p(res_df['Strategy_Return'])
    
    res_df['B&H_Equity'] = np.exp(res_df['Log_Market'].cumsum())
    res_df['Strategy_Equity'] = np.exp(res_df['Log_Strategy'].cumsum())
    
    return res_df, trade_log, buy_signals, sell_signals

def print_kpis(res_df, title="5. KPI DASHBOARD"):
    print("\n" + "="*50)
    print(title)
    print("="*50)
    
    # Time delta in years
    years = (res_df.index[-1] - res_df.index[0]).days / 365.25
    
    # Annualized Return & Volatility
    ann_ret = res_df['Strategy_Return'].mean() * 252
    ann_vol = res_df['Strategy_Return'].std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    
    # Maximum Drawdown (MDD)
    roll_max = res_df['Strategy_Equity'].cummax()
    drawdown = (res_df['Strategy_Equity'] / roll_max) - 1.0
    mdd = drawdown.min()
    
    # CAGR
    final_eq = res_df['Strategy_Equity'].iloc[-1]
    cagr = (final_eq ** (1 / years)) - 1 if years > 0 else 0
    
    # Calmar Ratio
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    
    print(f"Sharpe Ratio:       {sharpe:.2f}")
    print(f"Max Drawdown (MDD): {mdd*100:.2f}%")
    print(f"CAGR:               {cagr*100:.2f}%")
    print(f"Calmar Ratio:       {calmar:.2f}")
    
    return sharpe, cagr

def print_trade_log(trade_log):
    print("\n" + "="*50)
    print("8. TRADE LOG")
    print("="*50)
    
    if not trade_log:
        print("No trades were executed during this period.")
        return
        
    df_trades = pd.DataFrame(trade_log)
    print(df_trades.to_string(index=False))
    
    wins = df_trades[df_trades['Return %'] > 0]
    losses = df_trades[df_trades['Return %'] <= 0]
    
    win_rate = (len(wins) / len(df_trades)) * 100
    
    gross_profit = wins['Return %'].sum()
    gross_loss = abs(losses['Return %'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    print("\nOVERALL METRICS:")
    print(f"Total Trades Execute: {len(df_trades)}")
    print(f"Win Rate:             {win_rate:.1f}%")
    print(f"Profit Factor:        {profit_factor:.2f}")

def run_wfo(df):
    print("\n" + "="*50)
    print("7. OVERFITTING PROTECTION: Walk-Forward Optimization")
    print("="*50)
    
    cycles = [
        {"train": ("2020", "2022"), "test": ("2023", "2023")},
        {"train": ("2021", "2023"), "test": ("2024", "2024")},
        {"train": ("2022", "2024"), "test": ("2025", "2025")}
    ]
    
    for i, c in enumerate(cycles):
        train_start, train_end = c["train"]
        test_start, test_end = c["test"]
        
        # Extract out-of-sample data for the testing window
        # Note: We pass the pre-calculated features DataFrame so the logic works seamlessly
        test_df = df.loc[test_start:test_end]
        
        if test_df.empty:
            print(f"Cycle {i+1} [{test_start}]: Insufficient data for test.")
            continue
            
        res, _, _, _ = run_logic_and_simulate(test_df)
        
        # Calculate OOS KPIs silently
        years = max((res.index[-1] - res.index[0]).days / 365.25, 1/365.25)
        ann_ret = res['Strategy_Return'].mean() * 252
        ann_vol = res['Strategy_Return'].std() * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
        
        final_eq = res['Strategy_Equity'].iloc[-1]
        cagr = (final_eq ** (1 / years)) - 1
        
        print(f"Cycle {i+1} | Train: {train_start}-{train_end} | Test: {test_start}")
        print(f"  -> Out-of-Sample Sharpe: {sharpe:.2f}")
        print(f"  -> Out-of-Sample CAGR:   {cagr*100:.2f}%\n")
        
    print("WFO Conclusion: Check if the strategy returns consistently collapse in the out-of-sample windows. If they drop near zero or negative across multiple cycles, the full backtest is likely overfitted.")

def plot_equity_curve(res_df, buy_signals, sell_signals):
    print("\n[*] Rendering 5-Year Equity Curve Chart in browser...")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.3], vertical_spacing=0.05,
                        subplot_titles=("Equity Curve (Strategy vs Buy & Hold)", "SPY Price & Execution Signals"))
                        
    # 1. Equity Curves
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Strategy_Equity'], name='Strategy Equity', line=dict(color='cyan', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['B&H_Equity'], name='Buy & Hold', line=dict(color='orange', width=2, dash='dot')), row=1, col=1)
    
    # 2. Price and Signals
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Close'], name='SPY Price', line=dict(color='gray', width=1.5)), row=2, col=1)
    
    # Add buy markers
    if buy_signals:
        buy_dates, buy_prices = zip(*buy_signals)
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', 
                                 marker=dict(color='green', symbol='triangle-up', size=12), 
                                 name='BUY Signal'), row=2, col=1)
    
    # Add sell markers
    if sell_signals:
        sell_dates, sell_prices = zip(*sell_signals)
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', 
                                 marker=dict(color='red', symbol='triangle-down', size=12), 
                                 name='SELL Signal'), row=2, col=1)
        
    fig.update_layout(title="5-Year Backtest: MA Crossover + RSI Filter Strategy", 
                      template='plotly_dark',
                      height=800)
    fig.show()

def main():
    print("Initializing Algorithmic Trading Backtesting Engine...\n")
    
    # 1 & 2
    raw_df = get_data()
    feature_df = feature_engineering(raw_df)
    
    # 3 & 4
    simulated_df, trade_log, buys, sells = run_logic_and_simulate(feature_df)
    
    # 5
    print_kpis(simulated_df)
    
    # 8
    print_trade_log(trade_log)
    
    # 7
    run_wfo(feature_df)
    
    # 6
    plot_equity_curve(simulated_df, buys, sells)

if __name__ == "__main__":
    main()
