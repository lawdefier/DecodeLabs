"""
Stock Market Technical Analysis Dashboard
Fetches data, maps S/R zones, plots trendlines, and runs our custom IPO Decision Engine.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random # Just for demoing the risk circuits

# --- CONFIG ---
ASSETS = ['BTC-USD', 'SPY', 'EURUSD=X']
TIMEFRAME = '1y'

def get_data(ticker):
    print(f"\n[*] Grabbing {TIMEFRAME} data for {ticker}...")
    df = yf.download(ticker, period=TIMEFRAME, interval='1d', progress=False)
    
    # Sometimes yfinance gives us multi-index columns, let's flatten that out
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.dropna(inplace=True)
    return df

def crunch_candles(df):
    """
    Calculate wicks and body, and find those juicy rejection wicks.
    """
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Body'] = abs(df['Open'] - df['Close'])
    
    # Don't blow up on zero-body candles (dojis)
    safe_body = df['Body'].replace(0, 1e-5)
    
    # R_wb = (Upper + Lower) / Body
    df['R_wb'] = (df['Upper_Wick'] + df['Lower_Wick']) / safe_body
    
    # Flag candles where the wicks are at least twice the size of the body
    df['Rejection_Signal'] = df['R_wb'] > 2
    return df

def find_zones(df, window=21, tolerance=0.01):
    """
    Auto-detects support and resistance zones by looking for local mins/maxes
    that get touched multiple times.
    """
    # Find local bottoms and tops
    mins = df[df['Low'] == df['Low'].rolling(window=window, center=True).min()]['Low'].dropna()
    maxs = df[df['High'] == df['High'].rolling(window=window, center=True).max()]['High'].dropna()
    
    supports, resistances = [], []
    
    # Map supports
    for price in mins:
        # Check if we already have a zone around here
        if not any(abs(price - s)/s < tolerance for s in supports):
            # Did we touch this level at least twice?
            touches = sum(abs(df['Low'] - price)/price < tolerance)
            if touches >= 2:
                supports.append(price)
                
    # Map resistances
    for price in maxs:
        if not any(abs(price - r)/r < tolerance for r in resistances):
            touches = sum(abs(df['High'] - price)/price < tolerance)
            if touches >= 2:
                resistances.append(price)
                
    return supports, resistances

def draw_trendlines(df, window=14):
    """
    Connects recent higher lows (ascending) or lower highs (descending).
    """
    mins = df[df['Low'] == df['Low'].rolling(window=window, center=True).min()]['Low'].dropna()
    maxs = df[df['High'] == df['High'].rolling(window=window, center=True).max()]['High'].dropna()
    
    asc_line, desc_line = None, None
    
    # Need at least two points to draw a line
    if len(mins) >= 2:
        x1, x2 = mins.index[-2], mins.index[-1]
        y1, y2 = mins.values[-2], mins.values[-1]
        if y2 > y1: # higher low
            asc_line = ((x1, y1), (x2, y2))
            
    if len(maxs) >= 2:
        x1, x2 = maxs.index[-2], maxs.index[-1]
        y1, y2 = maxs.values[-2], maxs.values[-1]
        if y2 < y1: # lower high
            desc_line = ((x1, y1), (x2, y2))
            
    return asc_line, desc_line

def add_indicators(df):
    # EMAs
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # 14-period RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Crossovers
    df['Golden_Cross'] = (df['EMA_50'] > df['EMA_200']) & (df['EMA_50'].shift(1) <= df['EMA_200'].shift(1))
    df['Death_Cross'] = (df['EMA_50'] < df['EMA_200']) & (df['EMA_50'].shift(1) >= df['EMA_200'].shift(1))
    
    return df

def run_ipo_engine(ticker, df, supports, resistances):
    """
    Our core trading logic. Evaluates each day to see if it passes all 3 gates.
    """
    print(f"--- Running IPO Engine: {ticker} ---")
    
    consecutive_losses = 0
    
    for date, row in df.iterrows():
        close = row['Close']
        
        # Gate A: At a S/R boundary? (within 1%)
        near_sup = any(abs(close - s)/s <= 0.01 for s in supports)
        near_res = any(abs(close - r)/r <= 0.01 for r in resistances)
        gate_a = near_sup or near_res
        
        # Gate B: Bullish alignment?
        gate_b = row['EMA_50'] > row['EMA_200']
        
        # Gate C: Momentum confirmed?
        gate_c = row['RSI'] > 50
        
        # All systems go?
        if gate_a and gate_b and gate_c:
            dt_str = date.strftime('%Y-%m-%d')
            print(f"🚀 BUY SIGNAL on {dt_str}")
            
            entry = close
            take_profit = entry * 1.03
            stop_loss = entry * 0.985
            risk = entry - stop_loss
            reward = take_profit - entry
            rr = reward / risk if risk > 0 else 0
            
            print(f"    Entry:       ${entry:.2f}")
            print(f"    Take-Profit: ${take_profit:.2f} (+3%)")
            print(f"    Stop-Loss:   ${stop_loss:.2f} (-1.5%)")
            print(f"    R:R Ratio:   {rr:.2f}")
            
            # --- CIRCUIT BREAKERS (Demo logic) ---
            # Simulate trade outcomes randomly so we can occasionally see the breakers hit in the logs
            trade_lost = random.choice([True, False])
            if trade_lost:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
                
            if consecutive_losses >= 2:
                print("    🚨 RISK ALERT: Half-Size Rule! 2 losses in a row. Cutting size by 50%.")
                consecutive_losses = 0 # reset
                
            # Randomly trigger the 15-min hard stop for demo purposes
            if random.random() < 0.1:
                print("    🛑 RISK ALERT: 15-Minute Hard Stop! Drawdown hit 50% of daily limit. Step away.")

def render_chart(ticker, df, supports, resistances, asc_line, desc_line):
    """
    Builds the interactive Plotly chart and opens it in the browser.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price & Indicators', 'RSI (14)'),
                        row_width=[0.2, 0.7])

    # 1. Main Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                
    # Rejections (Wick-to-Body > 2)
    rejections = df[df['Rejection_Signal']]
    fig.add_trace(go.Scatter(x=rejections.index, y=rejections['High'] * 1.02,
                             mode='markers', marker=dict(color='yellow', size=8, symbol='star'),
                             name='Strong Rejection'), row=1, col=1)

    # 2. EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='blue', width=1.5), name='50 EMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='orange', width=1.5), name='200 EMA'), row=1, col=1)
    
    # Golden / Death Crosses
    golden = df[df['Golden_Cross']]
    death = df[df['Death_Cross']]
    fig.add_trace(go.Scatter(x=golden.index, y=golden['EMA_50'], mode='markers', marker=dict(color='green', size=12, symbol='triangle-up'), name='Golden Cross'), row=1, col=1)
    fig.add_trace(go.Scatter(x=death.index, y=death['EMA_50'], mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'), name='Death Cross'), row=1, col=1)

    # 3. S/R Zones
    for s in supports:
        fig.add_hline(y=s, line_dash="dash", line_color="green", opacity=0.4, row=1, col=1)
    for r in resistances:
        fig.add_hline(y=r, line_dash="dash", line_color="red", opacity=0.4, row=1, col=1)

    # 4. Trendlines
    if asc_line:
        fig.add_trace(go.Scatter(x=[asc_line[0][0], asc_line[1][0]], y=[asc_line[0][1], asc_line[1][1]],
                                 mode='lines', line=dict(color='cyan', width=2, dash='dot'), name='Ascending TL'), row=1, col=1)
    if desc_line:
        fig.add_trace(go.Scatter(x=[desc_line[0][0], desc_line[1][0]], y=[desc_line[0][1], desc_line[1][1]],
                                 mode='lines', line=dict(color='magenta', width=2, dash='dot'), name='Descending TL'), row=1, col=1)

    # 5. RSI Subplot
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="white", opacity=0.5, row=2, col=1)

    # Polish it up
    fig.update_layout(title=f"Technical Dashboard: {ticker}", template='plotly_dark', xaxis_rangeslider_visible=False, height=800)
    fig.show()

def main():
    print("Starting up the engine...")
    for ticker in ASSETS:
        df = get_data(ticker)
        if df.empty:
            print(f"Oops, couldn't fetch data for {ticker}.")
            continue
            
        df = crunch_candles(df)
        df = add_indicators(df)
        
        supports, resistances = find_zones(df)
        asc_line, desc_line = draw_trendlines(df)
        
        run_ipo_engine(ticker, df, supports, resistances)
        
        # This will spin up the local server and pop open your browser
        render_chart(ticker, df, supports, resistances, asc_line, desc_line)

if __name__ == "__main__":
    main()
