#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build RSI(14) divergence SIGNALS from datewise daily OHLCV store.

Input:
  data/ohlcv_daily/ohlcv_YYYY-MM-DD.csv.gz
  Each file has rows: date,symbol,open,high,low,close,volume (one date, many symbols)

Output (compact, one row per divergence event):
  signals/divergences_YYYY-MM-DD.csv
  signals/divergences_latest.csv

Leak-fixed:
- Pivots are only "known" at confirm_idx (reversal threshold).
- SignalDate = Date[confirm_idx]
- EntryDate  = next bar date if available else next business day (BDay(1))
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


# =========================
# CONFIG (edit via env)
# =========================
OHLCV_DAILY_DIR = os.environ.get("OHLCV_DAILY_DIR", "./data/ohlcv_daily")
OUT_DIR = os.environ.get("DIVERGENCE_OUT_DIR", "./signals")

MODE = os.environ.get("DIV_MODE", "latest")  # "latest" only (recommended), "all" optional
LOOKBACK_BARS = int(os.environ.get("LOOKBACK_BARS", "252"))
SECOND_POINT_LOOKBACK = int(os.environ.get("SECOND_POINT_LOOKBACK", "5"))
ATR_PERIOD = int(os.environ.get("ATR_PERIOD", "14"))
ATR_MULT = float(os.environ.get("ATR_MULT", "1.1"))

# how many daily files to load (trading days). Must be >= LOOKBACK_BARS + warmup
LOAD_LAST_FILES = int(os.environ.get("LOAD_LAST_FILES", str(max(LOOKBACK_BARS + 120, 340))))

# Filters
EXCLUDE_ZERO_VOLUME_ON_TARGET_DAY = os.environ.get("EXCLUDE_ZERO_VOLUME_ON_TARGET_DAY", "1").strip().lower() not in {"0","false","no"}
EXCLUDE_INDEX_LIKE = os.environ.get("EXCLUDE_INDEX_LIKE", "1").strip().lower() not in {"0","false","no"}

INDEX_LIKE_RE = re.compile(r"(?:^|[^A-Z])NIFTY|BANKNIFTY|SENSEX|NIFTY\s*50|NIFTY50", re.IGNORECASE)


# =========================
# INDICATORS
# =========================
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


# =========================
# Leak-fixed ZigZag
# =========================
def zigzag_atr_confirmed(df: pd.DataFrame,
                         atr_period: int = 14,
                         atr_mult: float = 2.0) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    n = len(closes)
    if n < 2:
        return [], []

    prev_close = np.empty_like(closes)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]

    tr = np.maximum.reduce([
        (highs - lows),
        np.abs(highs - prev_close),
        np.abs(lows - prev_close),
    ])

    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().to_numpy()

    swing_highs: List[Tuple[int,int]] = []
    swing_lows: List[Tuple[int,int]] = []

    direction: Optional[str] = None
    pivot_price = closes[0]

    candidate_high_idx = 0
    candidate_low_idx = 0

    for i in range(1, n):
        thr = atr_mult * (atr[i] if not np.isnan(atr[i]) else 1.0)

        if highs[i] > highs[candidate_high_idx]:
            candidate_high_idx = i
        if lows[i] < lows[candidate_low_idx]:
            candidate_low_idx = i

        if direction is None:
            if closes[i] >= pivot_price + thr:
                direction = "up"
                candidate_high_idx = i
            elif closes[i] <= pivot_price - thr:
                direction = "down"
                candidate_low_idx = i

        elif direction == "up":
            if closes[i] <= highs[candidate_high_idx] - thr:
                swing_highs.append((int(candidate_high_idx), int(i)))
                direction = "down"
                candidate_low_idx = i
                pivot_price = closes[i]

        elif direction == "down":
            if closes[i] >= lows[candidate_low_idx] + thr:
                swing_lows.append((int(candidate_low_idx), int(i)))
                direction = "up"
                candidate_high_idx = i
                pivot_price = closes[i]

    return swing_highs, swing_lows


# =========================
# Divergence detection
# =========================
@dataclass
class Divergence:
    kind: str          # "bullish" or "bearish"
    pivot1_idx: int
    pivot2_idx: int
    confirm_idx: int
    basis: str         # "close" or "extreme"


def detect_divergences_multi(df: pd.DataFrame,
                             lookback_bars: int,
                             second_point_lookback: int,
                             atr_period: int,
                             atr_mult: float,
                             basis: str = "close",
                             use_recency_filter: bool = True) -> List[Divergence]:
    n = len(df)
    if n < 50:
        return []

    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    rsi = df["RSI14"].to_numpy()

    hi_sw, lo_sw = zigzag_atr_confirmed(df, atr_period=atr_period, atr_mult=atr_mult)

    if use_recency_filter:
        cutoff_recent = n - max(1, second_point_lookback)
        cutoff_lookback = max(0, n - max(1, lookback_bars))
        second_cut = max(cutoff_recent, cutoff_lookback)
    else:
        second_cut = 0

    divs: List[Divergence] = []

    # Bearish: swing highs
    series_hi = highs if basis == "extreme" else closes
    for k in range(1, len(hi_sw)):
        (p1, _c1) = hi_sw[k - 1]
        (p2, c2) = hi_sw[k]
        if c2 < second_cut:
            continue
        if np.isnan(rsi[p1]) or np.isnan(rsi[p2]):
            continue
        if series_hi[p2] > series_hi[p1] and rsi[p2] < rsi[p1]:
            divs.append(Divergence("bearish", int(p1), int(p2), int(c2), basis))

    # Bullish: swing lows
    series_lo = lows if basis == "extreme" else closes
    for k in range(1, len(lo_sw)):
        (p1, _c1) = lo_sw[k - 1]
        (p2, c2) = lo_sw[k]
        if c2 < second_cut:
            continue
        if np.isnan(rsi[p1]) or np.isnan(rsi[p2]):
            continue
        if series_lo[p2] < series_lo[p1] and rsi[p2] > rsi[p1]:
            divs.append(Divergence("bullish", int(p1), int(p2), int(c2), basis))

    return divs


# =========================
# IO helpers
# =========================
def _list_daily_files(dir_path: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(dir_path, "ohlcv_????-??-??.csv.gz")))
    return files


def _date_from_fname(path: str) -> pd.Timestamp:
    base = os.path.basename(path)
    ds = base.replace("ohlcv_", "").replace(".csv.gz", "")
    return pd.to_datetime(ds, format="%Y-%m-%d", errors="raise").normalize()


def load_daily_store_last_n(dir_path: str, last_n_files: int) -> Tuple[pd.DataFrame, pd.Timestamp]:
    files = _list_daily_files(dir_path)
    if not files:
        raise FileNotFoundError(f"No ohlcv_YYYY-MM-DD.csv.gz files under: {dir_path}")

    files = files[-int(last_n_files):]
    target_day = _date_from_fname(files[-1])

    parts = []
    for fp in files:
        df = pd.read_csv(fp, compression="gzip")
        df.columns = [c.strip().lower() for c in df.columns]
        needed = {"date","symbol","open","high","low","close","volume"}
        if not needed.issubset(set(df.columns)):
            raise RuntimeError(f"Bad schema in {fp}. Have {df.columns}. Need {sorted(list(needed))}")
        df = df[["date","symbol","open","high","low","close","volume"]].copy()
        parts.append(df)

    all_df = pd.concat(parts, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"], format="%Y-%m-%d", errors="coerce").dt.normalize()
    all_df["symbol"] = all_df["symbol"].astype(str).str.upper().str.strip()

    for c in ["open","high","low","close","volume"]:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    all_df = all_df.dropna(subset=["date","symbol","close","high","low"]).copy()
    return all_df, target_day


# =========================
# Main divergence build
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df_all, target_day = load_daily_store_last_n(OHLCV_DAILY_DIR, LOAD_LAST_FILES)
    # filter index-like (optional)
    if EXCLUDE_INDEX_LIKE:
        df_all = df_all[~df_all["symbol"].str.contains(INDEX_LIKE_RE)].copy()

    # determine which symbols are eligible on target day
    df_td = df_all[df_all["date"] == target_day].copy()
    if df_td.empty:
        raise RuntimeError(f"No rows for target_day={target_day.date()} in loaded window.")
    if EXCLUDE_ZERO_VOLUME_ON_TARGET_DAY:
        elig_syms = set(df_td.loc[df_td["volume"].fillna(0).astype(float) > 0, "symbol"].unique().tolist())
    else:
        elig_syms = set(df_td["symbol"].unique().tolist())

    # keep only eligible symbols in full window
    df_all = df_all[df_all["symbol"].isin(elig_syms)].copy()

    # rename to match your logic
    df_all = df_all.rename(columns={
        "date": "Date",
        "symbol": "Ticker",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    df_all = df_all.sort_values(["Ticker","Date"]).reset_index(drop=True)

    out_rows: List[Dict] = []
    n_syms = df_all["Ticker"].nunique()
    processed = 0
    found = 0

    for sym, g in df_all.groupby("Ticker", sort=False):
        processed += 1
        g = g.sort_values("Date").reset_index(drop=True)

        # Must end at target day to produce signals for that day
        if g["Date"].iloc[-1] != target_day:
            continue

        if len(g) < 60:
            continue

        # Keep last LOOKBACK_BARS + warmup
        if len(g) > (LOOKBACK_BARS + 120):
            g = g.iloc[-(LOOKBACK_BARS + 120):].reset_index(drop=True)

        g["RSI14"] = compute_rsi(g["Close"], 14)
        g["ATR14"] = atr_wilder(g.rename(columns={"Date":"date"}), ATR_PERIOD)  # uses High/Low/Close columns

        use_recency = (MODE == "latest")

        divs = []
        divs += detect_divergences_multi(g, LOOKBACK_BARS, SECOND_POINT_LOOKBACK, ATR_PERIOD, ATR_MULT, basis="close", use_recency_filter=use_recency)
        divs += detect_divergences_multi(g, LOOKBACK_BARS, SECOND_POINT_LOOKBACK, ATR_PERIOD, ATR_MULT, basis="extreme", use_recency_filter=use_recency)

        if not divs:
            if processed % 150 == 0:
                print(f"[PROGRESS] {processed}/{n_syms} processed, found={found}")
            continue

        # Keep only divergences CONFIRMED on the target day
        # (signal time = confirm_idx; only act when it confirms)
        keep = []
        for d in divs:
            if 0 <= d.confirm_idx < len(g) and g.at[d.confirm_idx, "Date"] == target_day:
                keep.append(d)

        if not keep:
            if processed % 150 == 0:
                print(f"[PROGRESS] {processed}/{n_syms} processed, found={found}")
            continue

        # If multiple, keep the latest by confirm_idx (should be same day anyway)
        d = max(keep, key=lambda x: x.confirm_idx)

        p1, p2, c2 = d.pivot1_idx, d.pivot2_idx, d.confirm_idx
        entry_idx = c2 + 1
        signal_dt = pd.Timestamp(g.at[c2, "Date"]).normalize()
        entry_dt = (pd.Timestamp(g.at[entry_idx, "Date"]).normalize() if entry_idx < len(g) else (signal_dt + BDay(1)))

        # price basis for “price higher high / lower low” comparison
        if d.basis == "extreme":
            price1 = float(g.at[p1, "Low"] if d.kind == "bullish" else g.at[p1, "High"])
            price2 = float(g.at[p2, "Low"] if d.kind == "bullish" else g.at[p2, "High"])
        else:
            price1 = float(g.at[p1, "Close"])
            price2 = float(g.at[p2, "Close"])

        out_rows.append(dict(
            Ticker=str(sym),
            Kind=str(d.kind),
            Basis=str(d.basis),
            PivotIdx1=int(p1),
            PivotIdx2=int(p2),
            ConfirmIdx=int(c2),
            SignalDate=str(signal_dt.date()),
            EntryIdx=int(entry_idx),
            EntryDate=str(pd.Timestamp(entry_dt).date()),
            PriceAtPivot1=float(price1),
            PriceAtPivot2=float(price2),
            RSIAtPivot1=float(g.at[p1, "RSI14"]) if pd.notna(g.at[p1, "RSI14"]) else np.nan,
            RSIAtPivot2=float(g.at[p2, "RSI14"]) if pd.notna(g.at[p2, "RSI14"]) else np.nan,
            CloseConfirm=float(g.at[c2, "Close"]),
            VolumeConfirm=float(g.at[c2, "Volume"]),
            ATRMult=float(ATR_MULT),
            ATRPeriod=int(ATR_PERIOD),
            LookbackBars=int(LOOKBACK_BARS),
            SecondPointLookback=int(SECOND_POINT_LOOKBACK),
        ))
        found += 1

        if processed % 150 == 0:
            print(f"[PROGRESS] {processed}/{n_syms} processed, found={found}")

    out = pd.DataFrame(out_rows).sort_values(["Kind","Ticker"]).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, f"divergences_{target_day.date()}.csv")
    latest_path = os.path.join(OUT_DIR, "divergences_latest.csv")
    out.to_csv(out_path, index=False)
    out.to_csv(latest_path, index=False)

    print("==========================================================")
    print(f"[DONE] target_day={target_day.date()} symbols_loaded={n_syms} signals={len(out)}")
    print(f"[OUT] {out_path}")
    print(f"[OUT] {latest_path}")
    print("==========================================================")


if __name__ == "__main__":
    main()
