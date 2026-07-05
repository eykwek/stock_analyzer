#!/usr/bin/env python3
"""
stock_analyzer.py - Render Web Version
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
import io

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency. Install with:  pip install yfinance pandas numpy")
    sys.exit(1)

from flask import Flask

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global Configurations
# ---------------------------------------------------------------------------
LONG_TIMEFRAMES = [
    ("Weekly",    "1wk", "5y",   None),
    ("Daily",     "1d",  "2y",   None),
]
SHORT_TIMEFRAMES = [
    ("Hourly",    "60m", "730d", None),
    ("30-Min",    "30m", "60d",  None),
    ("15-Min",    "15m", "60d",  None),
    ("10-Min",    "5m",  "60d",  "10min"),   
    ("5-Min",     "5m",  "60d",  None),
]
ALL_TIMEFRAMES = LONG_TIMEFRAMES + SHORT_TIMEFRAMES
LONG_LABELS = {t[0] for t in LONG_TIMEFRAMES}
SHORT_LABELS = {t[0] for t in SHORT_TIMEFRAMES}

EMA_PERIOD = 20
SMA_PERIOD = 200
ATR_PERIOD = 14
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_MULTIPLIER = 2.0

@dataclass
class TimeframeResult:
    label: str
    last_price: float
    atr: float
    stop_long: float
    stop_short: float
    macd: float
    signal: float
    hist: float
    rsi: float
    ema20: float
    sma200: float

# ---------------------------------------------------------------------------
# Calculation Logic
# ---------------------------------------------------------------------------
def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period).mean()

def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_macd(close: pd.Series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def fetch_bars(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for interval={interval}, period={period}")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df

def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])

def analyze_timeframe(label, interval, period, resample_rule, ticker) -> TimeframeResult:
    df = fetch_bars(ticker, interval, period)
    if resample_rule:
        df = resample_bars(df, resample_rule)

    atr = compute_atr(df)
    macd_line, signal_line, hist = compute_macd(df["Close"])
    rsi = compute_rsi(df["Close"])
    ema20 = compute_ema(df["Close"], EMA_PERIOD)
    sma200 = compute_sma(df["Close"], SMA_PERIOD)

    last_price = float(df["Close"].iloc[-1])
    last_atr = float(atr.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_sma200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else float("nan")

    return TimeframeResult(
        label=label, last_price=last_price, atr=last_atr,
        stop_long=last_price - ATR_MULTIPLIER * last_atr,
        stop_short=last_price + ATR_MULTIPLIER * last_atr,
        macd=float(macd_line.iloc[-1]), signal=float(signal_line.iloc[-1]),
        hist=float(hist.iloc[-1]), rsi=float(rsi.iloc[-1]),
        ema20=last_ema20, sma200=last_sma200
    )

def score_timeframe(r: TimeframeResult) -> int:
    score = 0
    if r.macd > r.signal: score += 1
    elif r.macd < r.signal: score -= 1

    if r.rsi < 30: score += 1
    elif r.rsi > 70: score -= 1

    if r.last_price > r.ema20: score += 1
    elif r.last_price < r.ema20: score -= 1

    if not np.isnan(r.sma200):
        if r.last_price > r.sma200: score += 1
        elif r.last_price < r.sma200: score -= 1
    return score

def split_by_group(results: list) -> tuple:
    long_results = [r for r in results if r.label in LONG_LABELS]
    short_results = [r for r in results if r.label in SHORT_LABELS]
    return long_results, short_results

def overall_recommendation(results: list[TimeframeResult], buy_threshold=3, sell_threshold=-3, require_trend_filter=True):
    scores = [score_timeframe(r) for r in results]
    total = sum(scores)
    bullish_tf = sum(1 for s in scores if s > 0)
    bearish_tf = sum(1 for s in scores if s < 0)

    anchor = results[0]
    anchor_bullish = anchor.macd > anchor.signal
    anchor_bearish = anchor.macd < anchor.signal

    if total >= buy_threshold and (anchor_bullish or not require_trend_filter):
        rec = "BUY"
    elif total <= sell_threshold and (anchor_bearish or not require_trend_filter):
        rec = "SELL"
    else:
        rec = "HOLD"

    reasoning = (
        f"{bullish_tf}/{len(results)} timeframes bullish, "
        f"{bearish_tf}/{len(results)} bearish (combined score {total:+d}). "
        f"{anchor.label} MACD is {'above' if anchor_bullish else 'below'} its signal line, "
        f"{anchor.label} RSI is {anchor.rsi:.1f}."
    )
    return rec, reasoning

def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:,.2f}"

def print_report(ticker: str, results: list[TimeframeResult], short_buy_threshold=3, short_sell_threshold=-3, long_buy_threshold=2, long_sell_threshold=-2, require_trend_filter=True):
    finest_price = results[-1].last_price
    slowest_price = results[0].last_price

    print("=" * 78)
    print(f"  {ticker.upper()} — Technical Snapshot")
    print("=" * 78)
    print(f"Current Price (latest {results[-1].label} bar): ${fmt(finest_price)}")
    print(f"Current Price (latest {results[0].label} bar):   ${fmt(slowest_price)}")
    print()

    print(f"--- ATR & {ATR_MULTIPLIER:.0f}x-ATR Trailing Stop ---")
    header = f"{'Timeframe':<10} {'ATR':>10} {'Stop (Long)':>14} {'Stop (Short)':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.label:<10} {fmt(r.atr):>10} {fmt(r.stop_long):>14} {fmt(r.stop_short):>14}")
    print()

    print("--- MACD / Signal / RSI by Timeframe ---")
    header2 = f"{'Timeframe':<10} {'MACD':>10} {'Signal':>10} {'Hist':>10} {'RSI(14)':>10} {'Score':>7}"
    print(header2)
    print("-" * len(header2))
    for r in results:
        print(f"{r.label:<10} {fmt(r.macd):>10} {fmt(r.signal):>10} {fmt(r.hist):>10} "
              f"{fmt(r.rsi):>10} {score_timeframe(r):>+7d}")
    print()

    print(f"--- Trend: {EMA_PERIOD} EMA / {SMA_PERIOD} SMA by Timeframe ---")
    header3 = f"{'Timeframe':<10} {'Price':>10} {'EMA'+str(EMA_PERIOD):>10} {'SMA'+str(SMA_PERIOD):>10} {'vs EMA':>8} {'vs SMA':>8}"
    print(header3)
    print("-" * len(header3))
    for r in results:
        vs_ema = "Above" if r.last_price > r.ema20 else "Below"
        vs_sma = "n/a" if np.isnan(r.sma200) else ("Above" if r.last_price > r.sma200 else "Below")
        print(f"{r.label:<10} {fmt(r.last_price):>10} {fmt(r.ema20):>10} {fmt(r.sma200):>10} "
              f"{vs_ema:>8} {vs_sma:>8}")
    print()

    long_results, short_results = split_by_group(results)

    print("--- Recommendations ---")
    if short_results:
        rec_s, reasoning_s = overall_recommendation(short_results, short_buy_threshold, short_sell_threshold, require_trend_filter)
        labels_s = "/".join(r.label for r in short_results)
        print(f"Short-Term (hours-to-days; {labels_s}):")
        print(f"  Signal: {rec_s}")
        print(f"  Reasoning: {reasoning_s}")
    if long_results:
        rec_l, reasoning_l = overall_recommendation(long_results, long_buy_threshold, long_sell_threshold, require_trend_filter)
        labels_l = "/".join(r.label for r in long_results)
        print(f"Long-Term (days-to-months; {labels_l}):")
        print(f"  Signal: {rec_l}")
        print(f"  Reasoning: {reasoning_l}")
    print("=" * 78)

# ---------------------------------------------------------------------------
# Flask Web Server Route
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    # Capture the standard console prints into a text stream for the web browser
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    # --- DEFINE TICKERS TO ANALYZE HERE ---
    watchlist = ["AAPL", "MSFT", "TSLA"] 
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Snapshot as of: {timestamp}\n")

    for ticker in watchlist:
        results = []
        for label, interval, period, resample_rule in ALL_TIMEFRAMES:
            try:
                results.append(analyze_timeframe(label, interval, period, resample_rule, ticker))
            except Exception as e:
                pass
        if results:
            print_report(ticker, results)
            print("\n\n")

    # Restore default print behavior and capture string
    sys.stdout = old_stdout
    report_output = buffer.getvalue()

    # Return it wrapped inside an HTML preformatted tag to look nice in browser
    return f"<html><body style='font-family: monospace; background-color: #1e1e1e; color: #d4d4d4; padding: 20px;'><pre>{report_output}</pre></body></html>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
