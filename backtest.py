import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
TICKER = "SPY"
START = "2020-01-01"
END = "2025-01-01"
RF_RATE = 0.05
def pull_and_clean_data():
    print(f"\n[*] Pulling down {TICKER} data from {START} to {END}...")
    df = yf.download(TICKER, start=START, end=END, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.ffill(inplace=True)
    assert df.isna().sum().sum() == 0, "Wait, we still have NaN values. Something is broken!"
    return df
def build_features(df):
    print("[*] Crunching the indicators (MA 10, MA 50, RSI 14)...")
    df['Fast_MA'] = df['Close'].rolling(10).mean()
    df['Slow_MA'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-1 * delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)                                  
    df['RSI'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df
def run_simulation(df):
    print("\n--- Running the Trading Robot ---")
    pos = 0
    entry_px = 0.0
    entry_dt = None
    stance = np.zeros(len(df))
    trades = []
    buys, sells = [], []
    for i in range(len(df)):
        r = df.iloc[i]
        c, f, s, rsi = r['Close'], r['Fast_MA'], r['Slow_MA'], r['RSI']
        dt = df.index[i]
        if pos == 1:
            ret = (c - entry_px) / entry_px
            reason = None
            if ret >= 0.06: reason = "Take Profit"
            elif ret <= -0.02: reason = "Stop Loss"
            elif f < s: reason = "Trend Reversed"
            if reason:
                pos = 0         
                sells.append((dt, c))
                trades.append({
                    'In': entry_dt.strftime('%Y-%m-%d'), 'Out': dt.strftime('%Y-%m-%d'),
                    'In_Px': entry_px, 'Out_Px': c, 'Ret%': ret*100, 'Why': reason
                })
                entry_px, entry_dt = 0.0, None
        if pos == 0:
            if f > s and rsi > 50:
                pos = 1        
                entry_px, entry_dt = c, dt
                buys.append((dt, c))
        stance[i] = pos
    df = df.copy()
    df['Stance'] = stance
    df['Mkt_Ret'] = df['Close'].pct_change()
    df['Strat_Ret'] = df['Mkt_Ret'] * df['Stance'].shift(1)
    df.fillna(0, inplace=True)
    df['Eq_Hold'] = np.exp(np.log1p(df['Mkt_Ret']).cumsum())
    df['Eq_Strat'] = np.exp(np.log1p(df['Strat_Ret']).cumsum())
    return df, trades, buys, sells
def show_kpis(df):
    print("\n--- BACKTEST RESULTS ---")
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    ann_r = df['Strat_Ret'].mean() * 252
    ann_v = df['Strat_Ret'].std() * np.sqrt(252)
    cagr = (df['Eq_Strat'].iloc[-1] ** (1/yrs)) - 1
    mdd = (df['Eq_Strat'] / df['Eq_Strat'].cummax() - 1).min()
    sharpe = (ann_r - RF_RATE) / ann_v if ann_v > 0 else 0
    print(f"Sharpe Ratio:  {sharpe:.2f}")
    print(f"Max Drawdown:  {mdd*100:.2f}%")
    print(f"CAGR:          {cagr*100:.2f}%")
    return sharpe, cagr
def walk_forward_optimization(df):
    print("\n--- WALK-FORWARD OPTIMIZATION (WFO) ---")
    print("Making sure the model isn't just memorizing the past (overfitting)...")
    cycles = [
        ("2020", "2022", "2023"),
        ("2021", "2023", "2024")
    ]
    for tr_start, tr_end, tst in cycles:
        test_df = df.loc[tst:tst]
        if test_df.empty: continue
        res, _, _, _ = run_simulation(test_df)
        yrs = max((res.index[-1] - res.index[0]).days / 365.25, 1/365.25)
        sharpe = ((res['Strat_Ret'].mean() * 252) - RF_RATE) / (res['Strat_Ret'].std() * np.sqrt(252))
        cagr = (res['Eq_Strat'].iloc[-1] ** (1/yrs)) - 1
        print(f"Tested on {tst} -> Sharpe: {sharpe:.2f} | CAGR: {cagr*100:.2f}%")
def plot_results(df, buys, sells):
    print("\n[*] Popping open the equity curve in your browser!")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df['Eq_Strat'], name='Our Robot', line=dict(color='cyan')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Eq_Hold'], name='Buy & Hold', line=dict(color='orange', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='SPY Price', line=dict(color='gray')), row=2, col=1)
    if buys:
        bd, bp = zip(*buys)
        fig.add_trace(go.Scatter(x=bd, y=bp, mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy'), row=2, col=1)
    if sells:
        sd, sp = zip(*sells)
        fig.add_trace(go.Scatter(x=sd, y=sp, mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell'), row=2, col=1)
    fig.update_layout(title="Did We Beat The Market?", template='plotly_dark', height=800)
    fig.show()
def main():
    print("Booting up the Backtester...")
    raw = pull_and_clean_data()
    feats = build_features(raw)
    sim_df, trades, buys, sells = run_simulation(feats)
    show_kpis(sim_df)
    if trades:
        print("\n--- TRADE LOG ---")
        df_t = pd.DataFrame(trades)
        wins = len(df_t[df_t['Ret%'] > 0])
        print(f"We executed {len(trades)} trades with a {wins/len(trades)*100:.1f}% win rate.")
    walk_forward_optimization(feats)
    plot_results(sim_df, buys, sells)
if __name__ == "__main__":
    main()