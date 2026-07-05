#!/usr/bin/env python3
"""
stock_analyzer.py

Fetches the latest trade data for a ticker and reports, split by horizon:
  - Long-Term  (Weekly, Daily)              -> days-to-months read
  - Short-Term (Hourly, 30/15/10/5-Min)     -> hours-to-days read
Each horizon gets its own BUY / SELL / HOLD call, based on MACD + RSI +
20 EMA + 200 SMA confluence across that horizon's timeframes. Also reports:
  - Current price
  - ATR and a 2x-ATR trailing stop, per timeframe
  - MACD line, Signal line, Histogram, and RSI(14), per timeframe
  - 20 EMA / 200 SMA trend position, per timeframe

Requirements:
    pip install yfinance pandas numpy

Usage:
    python stock_analyzer.py                               # no ticker given -> prompts you
    python stock_analyzer.py AAPL
    python stock_analyzer.py AAPL MSFT TSLA                 # multiple tickers, one full report each
    python stock_analyzer.py --watchlist mylist.txt         # table mode: BUY/SELL/HOLD for the
                                                             # whole list, auto-refreshing every 5 min
    python stock_analyzer.py --watchlist mylist.txt --refresh 0   # same table, one-shot (no refresh)
    python stock_analyzer.py AAPL MSFT --refresh 60         # single-ticker mode, re-run every 60s
    python stock_analyzer.py TSLA --rsi-period 14 --atr-period 14

Notes on --refresh:
    Yahoo Finance's free data feed (used by yfinance) is not tick-by-tick
    real-time — it's typically delayed and only updates on the order of
    every ~15-60 seconds at best, and hammering it every few seconds risks
    getting temporarily rate-limited/blocked. In --watchlist mode the
    default refresh is 300 seconds (5 minutes), which is a reasonable
    balance of freshness vs. not getting rate-limited when scanning many
    tickers each cycle. For single-ticker mode there's no default
    auto-refresh unless you pass --refresh explicitly.

IMPORTANT: This tool applies a fixed, mechanical set of technical-analysis
rules (MACD crossover + RSI level). It is NOT financial advice, does not
account for fundamentals, news, market regime, or risk tolerance, and
technical indicators can and do give false signals. Use it as one input
among many, and consider consulting a licensed financial advisor before
trading.
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency. Install with:  pip install yfinance pandas numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Timeframe configuration
# yfinance does not natively support a 10-minute bar, so we build it by
# resampling 5-minute bars.
#
# Split into two horizon groups so BUY/SELL/HOLD can be read separately:
#   LONG_TIMEFRAMES  - Weekly, Daily      (days-to-months horizon)
#   SHORT_TIMEFRAMES - Hourly...5-Min     (hours-to-days horizon)
# Order within each group matters: index 0 is treated as that group's slowest/
# anchor timeframe for trend confirmation.
# ---------------------------------------------------------------------------
LONG_TIMEFRAMES = [
    # label,      yfinance interval, period, needs_resample_from
    ("Weekly",    "1wk", "5y",   None),
    ("Daily",     "1d",  "2y",   None),
]
SHORT_TIMEFRAMES = [
    ("Hourly",    "60m", "730d", None),
    ("30-Min",    "30m", "60d",  None),
    ("15-Min",    "15m", "60d",  None),
    ("10-Min",    "5m",  "60d",  "10min"),   # resampled from 5m
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


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period).mean()


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder's smoothing (standard ATR)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral if undefined (e.g. no losses yet)


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
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    return out


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
    # SMA200 is NaN until there are 200 bars of history; that's expected on
    # shorter-history intraday timeframes and is handled gracefully downstream.
    last_sma200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else float("nan")

    return TimeframeResult(
        label=label,
        last_price=last_price,
        atr=last_atr,
        stop_long=last_price - ATR_MULTIPLIER * last_atr,
        stop_short=last_price + ATR_MULTIPLIER * last_atr,
        macd=float(macd_line.iloc[-1]),
        signal=float(signal_line.iloc[-1]),
        hist=float(hist.iloc[-1]),
        rsi=float(rsi.iloc[-1]),
        ema20=last_ema20,
        sma200=last_sma200,
    )


def score_timeframe(r: TimeframeResult) -> int:
    """+1 bullish, -1 bearish, 0 neutral, per factor:
    MACD cross, RSI extreme, price vs 20 EMA, price vs 200 SMA.
    Max range widened from +/-2 to +/-4 per timeframe now that trend
    filters are included -- re-tune --buy-threshold/--sell-threshold
    if you were relying on the old +/-3 defaults meaning what they used to.
    """
    score = 0
    if r.macd > r.signal:
        score += 1
    elif r.macd < r.signal:
        score -= 1

    if r.rsi < 30:
        score += 1        # oversold -> bullish tilt
    elif r.rsi > 70:
        score -= 1        # overbought -> bearish tilt

    if r.last_price > r.ema20:
        score += 1        # above short-term trend
    elif r.last_price < r.ema20:
        score -= 1

    if not np.isnan(r.sma200):
        if r.last_price > r.sma200:
            score += 1     # above long-term trend
        elif r.last_price < r.sma200:
            score -= 1
    # if sma200 is NaN (not enough history on this timeframe), it simply
    # contributes 0 rather than skewing the score.

    return score


def split_by_group(results: list) -> tuple:
    """Split a list of TimeframeResult into (long_term, short_term) sublists,
    preserving the coarse-to-fine order each group needs for its anchor logic."""
    long_results = [r for r in results if r.label in LONG_LABELS]
    short_results = [r for r in results if r.label in SHORT_LABELS]
    return long_results, short_results


def overall_recommendation(results: list[TimeframeResult], buy_threshold=3, sell_threshold=-3,
                            require_trend_filter=True):
    """Compute BUY/SELL/HOLD for a list of TimeframeResults. results[0] is treated
    as the group's slowest/anchor timeframe for trend confirmation (e.g. Weekly for
    a long-term group, Hourly for a short-term group)."""
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


def print_report(ticker: str, results: list[TimeframeResult],
                  short_buy_threshold=3, short_sell_threshold=-3,
                  long_buy_threshold=2, long_sell_threshold=-2,
                  require_trend_filter=True):
    # results is expected in ALL_TIMEFRAMES order: Weekly, Daily, Hourly...5-Min
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
        rec_s, reasoning_s = overall_recommendation(
            short_results, short_buy_threshold, short_sell_threshold, require_trend_filter)
        labels_s = "/".join(r.label for r in short_results)
        print(f"Short-Term (hours-to-days; {labels_s}):")
        print(f"  Signal: {rec_s}")
        print(f"  Reasoning: {reasoning_s}")
    else:
        print("Short-Term: no data available.")
    print()
    if long_results:
        rec_l, reasoning_l = overall_recommendation(
            long_results, long_buy_threshold, long_sell_threshold, require_trend_filter)
        labels_l = "/".join(r.label for r in long_results)
        print(f"Long-Term (days-to-months; {labels_l}):")
        print(f"  Signal: {rec_l}")
        print(f"  Reasoning: {reasoning_l}")
    else:
        print("Long-Term: no data available.")
    print()
    print("Note: mechanical, rules-based read (MACD crossover + RSI level) only.")
    print("Not financial advice — verify against fundamentals, news, and your own risk tolerance.")
    print("=" * 78)


def load_watchlist(path: str) -> list[str]:
    tickers = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line.upper())
    return tickers


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def run_once(tickers: list[str], short_buy=3, short_sell=-3, long_buy=2, long_sell=-2,
             require_trend_filter=True, only=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Snapshot as of: {timestamp}")
    print()
    matches = 0
    for ticker in tickers:
        results = []
        for label, interval, period, resample_rule in ALL_TIMEFRAMES:
            try:
                results.append(analyze_timeframe(label, interval, period, resample_rule, ticker))
            except Exception as e:
                print(f"Warning: could not compute {label} timeframe for {ticker} ({e}). Skipping.",
                      file=sys.stderr)
        if not results:
            print(f"No timeframes could be analyzed for {ticker}. Check the symbol / connection.")
            print()
            continue

        long_results, short_results = split_by_group(results)
        rec_s, _ = overall_recommendation(short_results, short_buy, short_sell, require_trend_filter) \
            if short_results else ("HOLD", "")
        rec_l, _ = overall_recommendation(long_results, long_buy, long_sell, require_trend_filter) \
            if long_results else ("HOLD", "")

        if only in ("buy", "sell", "signals"):
            target = {"buy": ["BUY"], "sell": ["SELL"], "signals": ["BUY", "SELL"]}[only]
            if rec_s not in target and rec_l not in target:
                continue

        matches += 1
        print_report(ticker, results, short_buy, short_sell, long_buy, long_sell, require_trend_filter)
        print()

    if only:
        print(f"({matches}/{len(tickers)} tickers matched filter: {only})")
        print()


def build_ticker_summary(ticker: str, short_buy=3, short_sell=-3, long_buy=2, long_sell=-2,
                          require_trend_filter=True):
    """Run the full multi-timeframe analysis for one ticker and condense it into
    a single summary row (with separate short-term and long-term reads), for use
    in the watchlist table view."""
    results = []
    for label, interval, period, resample_rule in ALL_TIMEFRAMES:
        try:
            results.append(analyze_timeframe(label, interval, period, resample_rule, ticker))
        except Exception as e:
            print(f"Warning: could not compute {label} timeframe for {ticker} ({e}). Skipping.",
                  file=sys.stderr)

    if not results:
        return {
            "ticker": ticker, "price": float("nan"),
            "rec_short": "ERROR", "score_short": 0, "atr_short": float("nan"),
            "rec_long": "ERROR", "score_long": 0, "atr_long": float("nan"),
        }

    long_results, short_results = split_by_group(results)

    if short_results:
        rec_s, _ = overall_recommendation(short_results, short_buy, short_sell, require_trend_filter)
        score_s = sum(score_timeframe(r) for r in short_results)
        atr_s = short_results[0].atr   # anchor timeframe (Hourly)
    else:
        rec_s, score_s, atr_s = "n/a", 0, float("nan")

    if long_results:
        rec_l, _ = overall_recommendation(long_results, long_buy, long_sell, require_trend_filter)
        score_l = sum(score_timeframe(r) for r in long_results)
        atr_l = long_results[0].atr    # anchor timeframe (Weekly)
    else:
        rec_l, score_l, atr_l = "n/a", 0, float("nan")

    # Prefer the most recent price available (finest short-term timeframe), falling
    # back to the long-term group if short-term data couldn't be fetched.
    price = short_results[-1].last_price if short_results else (
        long_results[-1].last_price if long_results else float("nan"))

    return {
        "ticker": ticker,
        "price": price,
        "rec_short": rec_s,
        "score_short": score_s,
        "atr_short": atr_s,
        "rec_long": rec_l,
        "score_long": score_l,
        "atr_long": atr_l,
    }


def print_watchlist_table(tickers: list[str], short_buy=3, short_sell=-3, long_buy=2, long_sell=-2,
                           require_trend_filter=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 74)
    print(f"  Watchlist Scan — {timestamp}")
    print("=" * 74)
    print()

    rows = [build_ticker_summary(t, short_buy, short_sell, long_buy, long_sell, require_trend_filter)
            for t in tickers]

    header = (f"{'Ticker':<8} {'Price':>10} {'Short-Term':>11} {'S.Score':>8} {'S.ATR':>8} "
              f"{'Long-Term':>10} {'L.Score':>8} {'L.ATR':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f'{r["ticker"]:<8} {fmt(r["price"]):>10} {r["rec_short"]:>11} {r["score_short"]:>+8d} '
              f'{fmt(r["atr_short"]):>8} '
              f'{r["rec_long"]:>10} {r["score_long"]:>+8d} {fmt(r["atr_long"]):>8}')
    print()
    print("(S.ATR = Hourly ATR, the short-term anchor timeframe; "
          "L.ATR = Weekly ATR, the long-term anchor timeframe.)")
    print()

    def counts(key):
        return (sum(1 for r in rows if r[key] == "BUY"),
                sum(1 for r in rows if r[key] == "SELL"),
                sum(1 for r in rows if r[key] == "HOLD"))

    sb, ss, sh = counts("rec_short")
    lb, ls, lh = counts("rec_long")
    print(f"Short-Term: {sb} BUY / {ss} SELL / {sh} HOLD")
    print(f"Long-Term:  {lb} BUY / {ls} SELL / {lh} HOLD")
    print(f"({len(rows)} tickers total)")
    print()
    print("Mechanical, rules-based read only — not financial advice.")


def main():
    global ATR_PERIOD, RSI_PERIOD

    parser = argparse.ArgumentParser(description="ATR / MACD / RSI stock snapshot tool")
    parser.add_argument("tickers", nargs="*",
                         help="One or more ticker symbols, e.g. AAPL MSFT TSLA. "
                              "If omitted and --watchlist isn't used either, you'll be "
                              "prompted to enter one interactively.")
    parser.add_argument("--watchlist",
                         help="Path to a text file with one ticker per line. Switches to a "
                              "condensed BUY/SELL/HOLD table for the whole list, auto-refreshing "
                              "every 5 minutes by default (see --refresh).")
    parser.add_argument("--atr-period", type=int, default=ATR_PERIOD)
    parser.add_argument("--rsi-period", type=int, default=RSI_PERIOD)
    parser.add_argument(
        "--refresh",
        type=int,
        default=None,
        help="Re-run every N seconds until interrupted (Ctrl+C). Default: 300s (5 min) when "
             "using --watchlist; off otherwise. Pass --refresh 0 to disable auto-refresh "
             "even in watchlist mode. Minimum recommended: 60 for single tickers, "
             "and definitely not below 15.",
    )
    parser.add_argument(
        "--short-buy-threshold",
        type=int,
        default=3,
        help="Combined score across the 5 short-term timeframes (Hourly..5-Min) needed "
             "to call BUY on the short-term read. Default: 3.",
    )
    parser.add_argument(
        "--short-sell-threshold",
        type=int,
        default=-3,
        help="Combined score needed to call SELL on the short-term read. Default: -3.",
    )
    parser.add_argument(
        "--long-buy-threshold",
        type=int,
        default=2,
        help="Combined score across the 2 long-term timeframes (Weekly, Daily) needed "
             "to call BUY on the long-term read. Default: 2 (lower than the short-term "
             "default since there are only 2 timeframes contributing instead of 5).",
    )
    parser.add_argument(
        "--long-sell-threshold",
        type=int,
        default=-2,
        help="Combined score needed to call SELL on the long-term read. Default: -2.",
    )
    parser.add_argument(
        "--no-trend-filter",
        action="store_true",
        help="By default, each horizon's BUY/SELL also requires that group's own slowest "
             "timeframe (Weekly for long-term, Hourly for short-term) to have its MACD "
             "agree with the direction. This is often why the tool sits on HOLD. Pass "
             "this flag to score purely off the combined multi-timeframe total instead.",
    )
    parser.add_argument(
        "--only",
        choices=["buy", "sell", "signals"],
        default=None,
        help="Single-ticker/positional mode only: only print tickers where EITHER the "
             "short-term or long-term read matches this outcome. 'signals' shows both "
             "BUY and SELL, hiding HOLD.",
    )
    args = parser.parse_args()

    ATR_PERIOD = args.atr_period
    RSI_PERIOD = args.rsi_period
    require_trend_filter = not args.no_trend_filter

    watchlist_mode = bool(args.watchlist)

    tickers = list(args.tickers)
    if args.watchlist:
        tickers.extend(load_watchlist(args.watchlist))

    # If nothing was given at all (no positional tickers, no --watchlist), prompt interactively.
    if not tickers and not watchlist_mode:
        entered = input("Enter ticker symbol(s) (comma or space separated): ").strip()
        tickers = [t for t in re.split(r"[,\s]+", entered) if t]

    # de-dupe while preserving order; also strip stray "$" prefixes (e.g. "$CAT" -> "CAT")
    tickers = [t.lstrip("$").strip().upper() for t in tickers]
    seen = set()
    tickers = [t for t in tickers if t and not (t in seen or seen.add(t))]

    if not tickers:
        parser.error("No tickers provided (positionally, via --watchlist, or interactively).")

    if watchlist_mode:
        refresh = args.refresh if args.refresh is not None else 300  # default: 5 minutes
    else:
        refresh = args.refresh if args.refresh is not None else 0    # default: run once

    if refresh and refresh < 15:
        print("Note: --refresh below 15s is not recommended (data won't be that fresh, "
              "and you risk rate-limiting). Continuing anyway.", file=sys.stderr)

    def do_run():
        if watchlist_mode:
            print_watchlist_table(tickers, args.short_buy_threshold, args.short_sell_threshold,
                                   args.long_buy_threshold, args.long_sell_threshold,
                                   require_trend_filter)
        else:
            run_once(tickers, args.short_buy_threshold, args.short_sell_threshold,
                     args.long_buy_threshold, args.long_sell_threshold,
                     require_trend_filter, args.only)

    if not refresh:
        do_run()
        return

    try:
        while True:
            clear_screen()
            do_run()
            print(f"Refreshing every {refresh}s — press Ctrl+C to stop.")
            time.sleep(refresh)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
