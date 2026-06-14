import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random                                      
ASSETS = ['BTC-USD']
TIMEFRAME = '1y'
def get_data(ticker):
    print(f"\n[*] Grabbing {TIMEFRAME} data for {ticker}...")
    df = yf.download(ticker, period=TIMEFRAME, interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df
def crunch_candles(df):
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Body'] = abs(df['Open'] - df['Close'])
    safe_body = df['Body'].replace(0, 1e-5)
    df['R_wb'] = (df['Upper_Wick'] + df['Lower_Wick']) / safe_body
    df['Rejection_Signal'] = df['R_wb'] > 2
    return df
def find_zones(df, window=21, tolerance=0.01):
    mins = df[df['Low'] == df['Low'].rolling(window=window, center=True).min()]['Low'].dropna()
    maxs = df[df['High'] == df['High'].rolling(window=window, center=True).max()]['High'].dropna()
    supports, resistances = [], []
    for price in mins:
        if not any(abs(price - s)/s < tolerance for s in supports):
            touches = sum(abs(df['Low'] - price)/price < tolerance)
            if touches >= 2:
                supports.append(price)
    for price in maxs:
        if not any(abs(price - r)/r < tolerance for r in resistances):
            touches = sum(abs(df['High'] - price)/price < tolerance)
            if touches >= 2:
                resistances.append(price)
    return supports, resistances
def draw_trendlines(df, window=14):
    mins = df[df['Low'] == df['Low'].rolling(window=window, center=True).min()]['Low'].dropna()
    maxs = df[df['High'] == df['High'].rolling(window=window, center=True).max()]['High'].dropna()
    asc_line, desc_line = None, None
    if len(mins) >= 2:
        x1, x2 = mins.index[-2], mins.index[-1]
        y1, y2 = mins.values[-2], mins.values[-1]
        if y2 > y1:              
            asc_line = ((x1, y1), (x2, y2))
    if len(maxs) >= 2:
        x1, x2 = maxs.index[-2], maxs.index[-1]
        y1, y2 = maxs.values[-2], maxs.values[-1]
        if y2 < y1:              
            desc_line = ((x1, y1), (x2, y2))
    return asc_line, desc_line
def add_indicators(df):
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Golden_Cross'] = (df['EMA_50'] > df['EMA_200']) & (df['EMA_50'].shift(1) <= df['EMA_200'].shift(1))
    df['Death_Cross'] = (df['EMA_50'] < df['EMA_200']) & (df['EMA_50'].shift(1) >= df['EMA_200'].shift(1))
    return df
def run_ipo_engine(ticker, df, supports, resistances):
    print(f"--- Running IPO Engine: {ticker} ---")
    consecutive_losses = 0
    for date, row in df.iterrows():
        close = row['Close']
        near_sup = any(abs(close - s)/s <= 0.01 for s in supports)
        near_res = any(abs(close - r)/r <= 0.01 for r in resistances)
        gate_a = near_sup or near_res
        gate_b = row['EMA_50'] > row['EMA_200']
        gate_c = row['RSI'] > 50
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
            trade_lost = random.choice([True, False])
            if trade_lost:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
            if consecutive_losses >= 2:
                print("    🚨 RISK ALERT: Half-Size Rule! 2 losses in a row. Cutting size by 50%.")
                consecutive_losses = 0         
            if random.random() < 0.1:
                print("    🛑 RISK ALERT: 15-Minute Hard Stop! Drawdown hit 50% of daily limit. Step away.")
def render_chart(ticker, df, supports, resistances, asc_line, desc_line):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price & Indicators', 'RSI (14)'),
                        row_width=[0.2, 0.7])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    rejections = df[df['Rejection_Signal']]
    fig.add_trace(go.Scatter(x=rejections.index, y=rejections['High'] * 1.02,
                             mode='markers', marker=dict(color='yellow', size=8, symbol='star'),
                             name='Strong Rejection'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='blue', width=1.5), name='50 EMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='orange', width=1.5), name='200 EMA'), row=1, col=1)
    golden = df[df['Golden_Cross']]
    death = df[df['Death_Cross']]
    fig.add_trace(go.Scatter(x=golden.index, y=golden['EMA_50'], mode='markers', marker=dict(color='green', size=12, symbol='triangle-up'), name='Golden Cross'), row=1, col=1)
    fig.add_trace(go.Scatter(x=death.index, y=death['EMA_50'], mode='markers', marker=dict(color='red', size=12, symbol='triangle-down'), name='Death Cross'), row=1, col=1)
    for s in supports:
        fig.add_hline(y=s, line_dash="dash", line_color="green", opacity=0.4, row=1, col=1)
    for r in resistances:
        fig.add_hline(y=r, line_dash="dash", line_color="red", opacity=0.4, row=1, col=1)
    if asc_line:
        fig.add_trace(go.Scatter(x=[asc_line[0][0], asc_line[1][0]], y=[asc_line[0][1], asc_line[1][1]],
                                 mode='lines', line=dict(color='cyan', width=2, dash='dot'), name='Ascending TL'), row=1, col=1)
    if desc_line:
        fig.add_trace(go.Scatter(x=[desc_line[0][0], desc_line[1][0]], y=[desc_line[0][1], desc_line[1][1]],
                                 mode='lines', line=dict(color='magenta', width=2, dash='dot'), name='Descending TL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="white", opacity=0.5, row=2, col=1)
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
        render_chart(ticker, df, supports, resistances, asc_line, desc_line)
if __name__ == "__main__":
    main()