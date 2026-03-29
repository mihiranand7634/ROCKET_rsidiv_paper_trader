#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rsidiv_daily_retrain_paper_trader.py

Goal
- Daily paper-trading runner that:
  (1) Retrains the ExpR (expectancy) model EVERY RUN (daily), using all CLOSED training trades up to cutoff=(RUN_DATE-1).
  (2) Loads divergence signals (divergences_latest.csv) and creates RR candidates (stop_atr, target_atr).
  (3) Scores candidates with ROCKET+static features using the freshly retrained model.
  (4) Applies finite-capital constraints and Rolling wR throttle gate (persistent across days).
  (5) Writes orders_to_place.csv and persists paper state (equity/open_positions/ledger/gate state).

Notes
- No argparse. Edit CONFIG.
- Uses causal gate: today's gate uses performance known up to yesterday (cutoff).
- Training uses only labels fully realized by cutoff: exit_date <= cutoff.

Expected repo layout (suggested)
- signals/divergences_latest.csv
- data/ohlcv.csv (or kaggle-downloaded file)
- data/baseline/portfolio_run/... (optional; you can train from baseline trades)
- paper_trader_out/ (generated; commit state/ if you want persistence across GitHub runs)

If you do NOT have baseline portfolio_run data available:
- Set USE_BASELINE_TRADESET=False and provide TRAIN_TRADES_CSV (your own historical closed-trade table).

"""

import os
import re
import sys
import json
import time
import math
import zipfile
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from collections import deque, Counter

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import RidgeClassifier, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


# =========================================================
# CONFIG (EDIT THESE)
# =========================================================

# --- Run date ---
# If RUN_DATE_ENV is set (YYYY-MM-DD), that is used.
# Otherwise uses "today" in Asia/Kolkata timezone.
RUN_DATE_ENV = "RUN_DATE"

# --- Inputs ---
DIVERGENCES_CSV = r"signals/divergences_latest.csv"

# OHLCV file must have at least: symbol/date/open/high/low/close/volume
# Column names can vary; loader auto-detects.
OHLCV_CSV = r"data/ohlcv_master.csv.gz"

# Training source:
# Option A: Use baseline "portfolio_run" trades + equity curves (original setup)
USE_BASELINE_TRADESET = True
BASELINE_RESULTS_PATH = os.environ.get("BASELINE_RESULTS_PATH", r"data/rsi-divergence-portfolio-144").strip()   # directory or .zip containing portfolio_run/
EXCLUDE_ETF = True

# Option B: If you have your own historical closed-trade CSV, set USE_BASELINE_TRADESET=False and give this:
TRAIN_TRADES_CSV = r"data/train_trades_closed.csv"  # must include entry_date/exit_date/symbol/side/rmult/stop_atr/target_atr

# --- Output root (persist this folder across runs for correct gating + open positions) ---
OUT_DIR = "paper_trader_out"

# --- Training window for DAILY retrain ---
TRAIN_LOOKBACK_YEARS = 5           # trailing window by ENTRY_DATE (and EXIT_DATE must be <= cutoff)
MIN_TRAIN_ROWS = 2000
MIN_VAL_ROWS = 1000
VALID_FRAC = 0.20
VALID_DAYS_MAX = 90
VALID_DAYS_MIN = 20
MAX_TRAIN_SAMPLES_PER_BUCKET = 160_000
MAX_VAL_SAMPLES_PER_BUCKET   = 60_000

# --- Feature engineering ---
SUSP_GAP_DAYS = 7

ROCKET_WINDOWS = [7, 15, 21]
ROCKET_CHANNELS = ["log_ret_close", "hl_range", "oc_move", "vol_chg"]
ROCKET_CFG = dict(kernels={7: 512, 15: 512, 21: 1024}, max_dilation=4)
ROCKET_RANDOM_SEED = 4242
ROCKET_BATCH_SIZE = 5000

ADD_STATIC_FEATURES = True

FEATURE_DTYPE = np.float16
TRAIN_DTYPE = np.float32
PRED_BATCH = 60_000

# --- Models ---
RIDGE_ALPHA = 1.0
W_PWIN_RIDGE = 0.50
W_PWIN_RF    = 0.50

RF_PWIN_PARAMS = dict(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=25,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

MAG_BIN_QUANTILES = [0.50, 0.80, 0.95, 0.99]
MAG_RF_PARAMS = dict(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=20,
    max_features="sqrt",
    n_jobs=-1,
    random_state=123,
)

R_ABS_MAX_SANITY = 20.0

# Missing bucket handling at scoring time:
# - "baseline": replace missing-bucket expr scores with BASELINE_PROXY_SCORE (notebook parity default)
# - "drop": drop missing-bucket candidates
# - "zero": legacy option (kept for backwards compatibility)
MISSING_BUCKET_FALLBACK = "baseline"   # "baseline" or "drop" (legacy: "zero")
SENTINEL_SCORE = -1e9

# RR policy (optional; used to pick one RR per signal)
RR_POLICY_ENABLE = True
RR_POLICY_BUCKET_AWARE = True
RR_POLICY_GLOBAL_FALLBACK = True
RR_POLICY_RIDGE_ALPHA = 100.0
RR_POLICY_RIDGE_SOLVER = "lsqr"
RR_POLICY_MIN_ROWS_PER_PAIR = 2500
RR_POLICY_MAX_ROWS_PER_PAIR = 120_000
RR_POLICY_PRED_CLIP = R_ABS_MAX_SANITY
RR_POLICY_BASELINE_WEIGHT = 0.25   # blend baseline_score into rr_choice_score when rr_pred is available

# --- Candidate generation from divergence signals ---
# stop ATR multiple: if STOP_ATR_LIST empty, uses ATRMult from the signal row.
STOP_ATR_LIST = []  # e.g., [0.5, 0.75, 1.0]
# targets are computed as target = stop * rr
RR_LIST = [1.0, 1.5, 2.0, 3.0]

# For live/paper trade exits:
ATR_PERIOD_DEFAULT = 14
MAX_HOLD_DAYS = 60              # time-exit if neither stop nor target hit within this many days
SAME_DAY_BOTH_HIT_POLICY = "stop_first"  # "stop_first" (conservative) or "target_first"

# --- Selection / portfolio constraints ---
START_EQUITY = 1_000_000.0
RISK_PCT_PER_TRADE = 0.005
MAX_OPEN_POSITIONS = 180
MAX_GROSS_RISK_PCT = 0.35
MAX_NEW_ENTRIES_PER_DAY = 120

SELECTION_MODE = "top_k_per_day"   # or "expr_threshold"
EXPR_THRESHOLD = 0.0

ENFORCE_ONE_RR_PER_SIGNAL = True

# --- Rolling wR throttle gate (persistent across days) ---
WR_GATE_ENABLE = True
WR_GATE_WINDOW_DAYS = 60
WR_GATE_MIN_TRADES = 60
WR_GATE_MIN_DAYS = 20
WR_GATE_USE_RISK_WEIGHT = True

WR_GATE_KILL_TH = -0.05
WR_GATE_REVIVE_TH = +0.05
WR_GATE_KILL_STREAK = 3
WR_GATE_REVIVE_STREAK = 5

WR_GATE_THROTTLE_MULT = 0.25
WR_GATE_SCORE_PENALTY = 0.00

# =========================================================
# LOGGING
# =========================================================
os.makedirs(OUT_DIR, exist_ok=True)
for sub in ["caches", "features", "models", "run", "state"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rsidiv_daily_retrain_paper")


# =========================================================
# DATE HELPERS (datetime64[ns] normalized)
# =========================================================
_MONTH_RE = re.compile(r"[A-Za-z]")

def s_day(series: pd.Series, fmt: Optional[str] = None, default_dayfirst: bool = False) -> pd.Series:
    if fmt is not None:
        dt = pd.to_datetime(series, format=fmt, errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(series, errors="coerce", dayfirst=default_dayfirst, utc=True)
    dt = dt.dt.tz_convert(None)
    return dt.dt.normalize()

def assert_day_dt(df: pd.DataFrame, col: str, name: str = ""):
    if col not in df.columns:
        raise RuntimeError(f"[ASSERT] missing column: {col}")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        raise RuntimeError(f"[ASSERT] {name}{col} must be datetime64[ns]. Got {df[col].dtype}")
    bad = (df[col].dt.hour != 0) | (df[col].dt.minute != 0) | (df[col].dt.second != 0) | (df[col].dt.nanosecond != 0)
    if bad.any():
        raise RuntimeError(f"[ASSERT] {name}{col} contains non-midnight times.")

def _date_part(s: str) -> str:
    s = str(s).strip()
    if not s:
        return s
    if "T" in s:
        s = s.split("T", 1)[0]
    if " " in s:
        s = s.split(" ", 1)[0]
    return s.strip()

def _tokenize_numeric_date(dp: str):
    dp = dp.strip()
    if not dp:
        return None, None
    for sep in ("-", "/", "."):
        if sep in dp:
            toks = [t.strip() for t in dp.split(sep) if t.strip() != ""]
            return sep, toks
    return None, None

def infer_date_format_from_samples(samples, default_dayfirst=False):
    cleaned = []
    for x in samples:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        s = str(x).strip()
        if not s:
            continue
        cleaned.append(_date_part(s))
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return None, default_dayfirst, "no_samples"

    if any(_MONTH_RE.search(c) for c in cleaned[:200]):
        month_formats = [
            "%d-%b-%Y", "%d-%b-%y", "%d-%B-%Y", "%d-%B-%y",
            "%b-%d-%Y", "%b-%d-%y", "%B-%d-%Y", "%B-%d-%y",
        ]
        best, best_ok = None, -1
        probe = cleaned[:200]
        for fmt in month_formats:
            ok = 0
            for v in probe:
                try:
                    datetime.strptime(v, fmt)
                    ok += 1
                except Exception:
                    pass
            if ok > best_ok:
                best_ok = ok
                best = fmt
        if best_ok > 0:
            return best, default_dayfirst, f"month_name_fmt_ok={best_ok}/{len(probe)}"
        return None, default_dayfirst, "month_name_unrecognized"

    sep = None
    token_lists = []
    for v in cleaned[:500]:
        ssep, toks = _tokenize_numeric_date(v)
        if ssep is None or toks is None:
            continue
        if sep is None:
            sep = ssep
        if ssep != sep:
            continue
        if len(toks) == 3:
            token_lists.append(toks)
    if not token_lists or sep is None:
        return None, default_dayfirst, "numeric_unrecognized"

    year_pos_votes = {0: 0, 2: 0}
    year_is_2digit_votes = 0
    for toks in token_lists[:200]:
        t0, _, t2 = toks
        if len(t0) == 4 and t0.isdigit():
            year_pos_votes[0] += 2
        if len(t2) == 4 and t2.isdigit():
            year_pos_votes[2] += 2
        if len(t2) == 2 and t2.isdigit():
            year_is_2digit_votes += 1

    year_pos = 0 if year_pos_votes[0] > year_pos_votes[2] else 2
    year_fmt = "%Y"
    if year_pos == 2 and year_is_2digit_votes > 0:
        year_fmt = "%y"

    decided = None
    for toks in token_lists[:500]:
        try:
            if year_pos == 2:
                a, b = int(toks[0]), int(toks[1])
            else:
                a, b = int(toks[1]), int(toks[2])
        except Exception:
            continue
        if b > 12 and a <= 12:
            decided = "md"
            break
        if a > 12 and b <= 12:
            decided = "dm"
            break
    if decided is None:
        decided = "dm" if default_dayfirst else "md"

    if year_pos == 2:
        return (f"%d{sep}%m{sep}{year_fmt}", True, "year_last_dayfirst") if decided == "dm" else (f"%m{sep}%d{sep}{year_fmt}", False, "year_last_monthfirst")
    else:
        return (f"%Y{sep}%d{sep}%m", True, "year_first_ydm") if decided == "dm" else (f"%Y{sep}%m{sep}%d", False, "year_first_ymd")

def parse_date_series_with_inference(series: pd.Series, default_dayfirst=False, sample_n=300):
    s = series.astype(str).map(lambda x: x.strip() if isinstance(x, str) else str(x))
    s = s.replace("nan", "").replace("None", "")
    s_date = s.map(_date_part)
    non_empty = [v for v in s_date.head(sample_n).tolist() if v and v.lower() not in {"nan", "none"}]
    fmt, dayfirst_used, note = infer_date_format_from_samples(non_empty, default_dayfirst=default_dayfirst)
    if fmt is not None:
        dt = s_day(s_date, fmt=fmt, default_dayfirst=dayfirst_used)
        return dt, fmt, note
    dt1 = s_day(s_date, fmt="%Y-%m-%d", default_dayfirst=False)
    if dt1.notna().mean() > 0.9:
        return dt1, "%Y-%m-%d", "fallback_iso"
    dt2 = s_day(s_date, fmt=None, default_dayfirst=default_dayfirst)
    return dt2, None, f"fallback_pandas_dayfirst={default_dayfirst}"


# =========================================================
# OHLCV LOADER (subset) + ROCKET series_map
# =========================================================
@dataclass
class SymSeries:
    dates64: np.ndarray
    seg_id: np.ndarray
    feats: np.ndarray
    ohlc: Optional[np.ndarray] = None  # columns: open,high,low,close (float32)
    atr: Optional[np.ndarray] = None   # ATR values (float32)

def _is_etf_symbol(sym):
    s = str(sym).upper()
    return ("ETF" in s) or ("BEES" in s)

def _find_ohlcv_columns(head_cols):
    cols = {c.lower().strip(): c for c in head_cols}
    sym_col  = cols.get("tradingsymbol") or cols.get("symbol") or cols.get("ticker")
    date_col = cols.get("date") or cols.get("timestamp") or cols.get("datetime")
    o_col = cols.get("open") or cols.get("open_price") or cols.get("openprice")
    h_col = cols.get("high") or cols.get("high_price") or cols.get("highprice")
    l_col = cols.get("low")  or cols.get("low_price") or cols.get("lowprice")
    c_col = cols.get("close") or cols.get("close_price") or cols.get("closeprice")
    v_col = cols.get("volume") or cols.get("ttl_trd_qnty") or cols.get("vol")
    return sym_col, date_col, o_col, h_col, l_col, c_col, v_col

def dedupe_duplicate_dates_closest_close(df_sym):
    if df_sym.empty:
        return df_sym
    df_sym = df_sym.sort_values(["date"]).copy()
    keep_idx = []
    prev_close = np.nan

    for d, g in df_sym.groupby("date", sort=True):
        if len(g) == 1:
            idx = g.index[0]
            row = g.iloc[0]
        else:
            gg = g[pd.notna(g["close"])].copy()
            if gg.empty:
                idx = g.index[-1]
                row = g.loc[idx]
            else:
                if pd.isna(prev_close):
                    if "volume" in gg.columns and gg["volume"].notna().any():
                        idx = gg.sort_values(["volume"], ascending=False).index[0]
                        row = gg.loc[idx]
                    else:
                        idx = gg.index[-1]
                        row = gg.loc[idx]
                else:
                    gg["absdiff"] = (gg["close"].astype(float) - float(prev_close)).abs()
                    sort_cols = ["absdiff"]
                    asc = [True]
                    if "volume" in gg.columns:
                        sort_cols.append("volume")
                        asc.append(False)
                    idx = gg.sort_values(sort_cols, ascending=asc).index[0]
                    row = gg.loc[idx]

        keep_idx.append(idx)
        if pd.notna(row["close"]):
            prev_close = float(row["close"])

    return df_sym.loc[keep_idx].copy().sort_values(["date"])

def add_gap_segments(px: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_values(["symbol", "date"]).copy()
    assert_day_dt(px, "date", name="[OHLCV] ")
    px["gap_days"] = px.groupby("symbol", sort=False)["date"].diff().dt.days
    px["seg_id"] = px["gap_days"].ge(SUSP_GAP_DAYS).groupby(px["symbol"]).cumsum().fillna(0).astype(int)
    return px

def compute_atr_from_ohlc(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    period = int(period)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    # Wilder ATR (RMA)
    atr = np.empty_like(tr, dtype=np.float32)
    atr[0] = tr[0]
    alpha = 1.0 / max(1, period)
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    return atr.astype(np.float32)

def load_ohlcv_subset_build_series(csv_path: str, symbols_needed: List[str], atr_period_default: int):
    """
    Builds:
    - series_map[sym] with dates64, seg_id, feats (ROCKET channels), and also OHLC + ATR arrays for paper exits.
    - seg_map dataframe for gap skip usage (symbol,date,seg_id)
    Cached by (gap_days, n_symbols, csv basename).
    """
    symbols_needed = [s.upper() for s in symbols_needed]
    symset = set(symbols_needed)

    base_name = os.path.basename(csv_path).replace(" ", "_").replace("(", "").replace(")", "")
    cache_series = os.path.join(OUT_DIR, "caches", f"series_cache_{base_name}_gap{SUSP_GAP_DAYS}_n{len(symset)}.pkl")
    cache_seg = os.path.join(OUT_DIR, "caches", f"seg_map_{base_name}_gap{SUSP_GAP_DAYS}_n{len(symset)}.pkl")

    if os.path.exists(cache_series) and os.path.exists(cache_seg):
        log.info(f"[OHLCV][CACHE] loading series={cache_series} seg={cache_seg}")
        series_map = joblib.load(cache_series)
        seg_map = pd.read_pickle(cache_seg)
        assert_day_dt(seg_map, "date", name="[SEG_MAP] ")
        return series_map, seg_map

    head = pd.read_csv(csv_path, nrows=5)
    head.columns = [c.strip() for c in head.columns]
    sym_col, date_col, o_col, h_col, l_col, c_col, v_col = _find_ohlcv_columns(head.columns)
    if sym_col is None or date_col is None or c_col is None:
        raise RuntimeError("OHLCV missing required columns (symbol/date/close).")

    usecols = [sym_col, date_col, c_col]
    for x in (o_col, h_col, l_col, v_col):
        if x is not None:
            usecols.append(x)
    usecols = list(dict.fromkeys(usecols))

    sample = pd.read_csv(csv_path, usecols=[date_col], nrows=2500)
    _parsed_s, fmt, note = parse_date_series_with_inference(sample[date_col], default_dayfirst=False, sample_n=400)
    log.info(f"[OHLCV][DATEFMT] fmt={fmt} note={note}")

    parts = []
    t0 = time.time()
    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=700_000), 1):
        chunk.columns = [c.strip() for c in chunk.columns]
        chunk = chunk.rename(columns={sym_col: "symbol", date_col: "date_raw", c_col: "close"})
        if o_col is not None and o_col in chunk.columns: chunk = chunk.rename(columns={o_col: "open"})
        if h_col is not None and h_col in chunk.columns: chunk = chunk.rename(columns={h_col: "high"})
        if l_col is not None and l_col in chunk.columns: chunk = chunk.rename(columns={l_col: "low"})
        if v_col is not None and v_col in chunk.columns: chunk = chunk.rename(columns={v_col: "volume"})

        chunk["symbol"] = chunk["symbol"].astype(str).str.upper().str.strip()
        chunk = chunk[chunk["symbol"].isin(symset)].copy()
        if chunk.empty:
            continue

        if fmt is not None:
            chunk["date"] = s_day(chunk["date_raw"].astype(str).map(_date_part), fmt=fmt, default_dayfirst=False)
        else:
            parsed, _, _ = parse_date_series_with_inference(chunk["date_raw"], default_dayfirst=False, sample_n=400)
            chunk["date"] = parsed

        for col in ["open", "high", "low", "close", "volume"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk.dropna(subset=["symbol", "date", "close"]).copy()
        if chunk.empty:
            continue

        for col in ["open", "high", "low", "volume"]:
            if col not in chunk.columns:
                chunk[col] = 0.0 if col == "volume" else chunk["close"]

        assert_day_dt(chunk, "date", name="[OHLCV_CHUNK] ")
        parts.append(chunk[["symbol", "date", "open", "high", "low", "close", "volume"]])

        if i % 8 == 0:
            log.info(f"[OHLCV] chunks={i} parts={len(parts)} elapsed={time.time()-t0:.1f}s")

    if not parts:
        raise RuntimeError("No OHLCV rows found for requested symbols.")
    px = pd.concat(parts, ignore_index=True)
    log.info(f"[OHLCV] subset rows={len(px):,} symbols={px['symbol'].nunique():,}")

    deduped = []
    for sym, g in px.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        if g.duplicated(subset=["date"]).any():
            g = dedupe_duplicate_dates_closest_close(g)
        deduped.append(g)
    px2 = pd.concat(deduped, ignore_index=True).sort_values(["symbol", "date"])
    log.info(f"[OHLCV] after dedupe rows={len(px2):,} dup={int(px2.duplicated(['symbol','date']).sum()):,}")

    px2 = add_gap_segments(px2)

    series_map: Dict[str, SymSeries] = {}
    seg_rows = []

    for sym, g in px2.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g = g[(g["open"] > 0) & (g["high"] > 0) & (g["low"] > 0) & (g["close"] > 0)].copy()
        if len(g) < 30:
            continue

        close = g["close"].astype(np.float32).values
        open_ = g["open"].astype(np.float32).values
        high = g["high"].astype(np.float32).values
        low  = g["low"].astype(np.float32).values
        vol  = g["volume"].fillna(0.0).astype(np.float32).values

        prev_close = np.roll(close, 1); prev_close[0] = close[0]
        log_ret_close = np.log(close / np.maximum(1e-12, prev_close)).astype(np.float32)
        hl_range = np.log(high / np.maximum(1e-12, low)).astype(np.float32)
        oc_move  = np.log(close / np.maximum(1e-12, open_)).astype(np.float32)
        vlog = np.log1p(np.maximum(0.0, vol)).astype(np.float32)
        vol_chg = np.diff(vlog, prepend=vlog[0]).astype(np.float32)

        feats = np.vstack([log_ret_close, hl_range, oc_move, vol_chg]).T.astype(np.float32)
        dates64 = g["date"].values.astype("datetime64[D]")
        seg_id = g["seg_id"].astype(np.int32).values

        # ATR for exits
        atr = compute_atr_from_ohlc(high, low, close, period=int(atr_period_default))

        ohlc = np.vstack([open_, high, low, close]).T.astype(np.float32)
        series_map[sym] = SymSeries(dates64=dates64, seg_id=seg_id, feats=feats, ohlc=ohlc, atr=atr)
        seg_rows.append(pd.DataFrame({"symbol": sym, "date": g["date"].values, "seg_id": seg_id}))

    if not seg_rows:
        raise RuntimeError("No seg_map rows built.")
    seg_map = pd.concat(seg_rows, ignore_index=True)
    assert_day_dt(seg_map, "date", name="[SEG_MAP_BUILD] ")

    seg_map.to_pickle(cache_seg)
    joblib.dump(series_map, cache_series)
    log.info(f"[OHLCV][CACHE] saved series={cache_series} seg={cache_seg}")

    return series_map, seg_map


# =========================================================
# ROCKET (deterministic kernels)
# =========================================================
@dataclass
class RocketKernel:
    channel: int
    length: int
    dilation: int
    weights: np.ndarray
    bias: np.float32
    idx_cols: List[np.ndarray]

class RocketFeaturizer:
    def __init__(self, window, n_channels, n_kernels, max_dilation, seed):
        self.window = int(window)
        self.n_channels = int(n_channels)
        self.n_kernels = int(n_kernels)
        self.max_dilation = int(max_dilation)
        self.seed = int(seed)
        self.kernels: List[RocketKernel] = []

    def _sample_dilation(self, rng, L, k_len):
        candidates = []
        d = 1
        while d <= self.max_dilation:
            if (k_len - 1) * d < L:
                candidates.append(d)
            d *= 2
        return int(rng.choice(candidates)) if candidates else 1

    def _sample_length(self, rng, L):
        max_len = min(11, L)
        lens = [2, 3, 5, 7, 9, 11]
        lens = [k for k in lens if k <= max_len]
        return int(rng.choice(lens)) if lens else min(2, L)

    def build(self):
        rng = np.random.RandomState(self.seed)
        L = self.window
        C = self.n_channels
        tries = 0
        while len(self.kernels) < self.n_kernels and tries < self.n_kernels * 50:
            tries += 1
            ch = int(rng.randint(0, C))
            k_len = self._sample_length(rng, L)
            d = self._sample_dilation(rng, L, k_len)
            pos_count = L - (k_len - 1) * d
            if pos_count <= 0:
                continue
            w = rng.normal(0, 1, size=k_len).astype(np.float32)
            w = (w - w.mean()).astype(np.float32)
            b = np.float32(rng.normal(0, 1))
            pos = np.arange(pos_count, dtype=np.int32)
            idx_cols = [pos + (j * d) for j in range(k_len)]
            self.kernels.append(RocketKernel(ch, k_len, d, w, b, idx_cols))
        return self

    def transform_batch(self, X):
        X = X.astype(np.float32, copy=False)
        n, L, _ = X.shape
        out = np.empty((n, 2 * len(self.kernels)), dtype=np.float32)
        for ki, k in enumerate(self.kernels):
            x = X[:, :, k.channel]
            y = np.full((n, k.idx_cols[0].shape[0]), k.bias, dtype=np.float32)
            for j, idx in enumerate(k.idx_cols):
                y += k.weights[j] * x[:, idx]
            out[:, 2 * ki] = y.max(axis=1)
            out[:, 2 * ki + 1] = (y > 0).mean(axis=1).astype(np.float32)
        return out

def normalize_kernels_dict(kernels) -> Dict[int, int]:
    return {int(k): int(v) for k, v in dict(kernels).items()}

def normalize_rocket_cfg(cfg: dict) -> dict:
    cfg2 = dict(cfg)
    cfg2["kernels"] = normalize_kernels_dict(cfg2["kernels"])
    cfg2["max_dilation"] = int(cfg2["max_dilation"])
    return cfg2

def build_featurizers(rocket_cfg: dict, seed: int) -> Dict[int, RocketFeaturizer]:
    cfg = normalize_rocket_cfg(rocket_cfg)
    C = len(ROCKET_CHANNELS)
    fe = {}
    for L in ROCKET_WINDOWS:
        nk = int(cfg["kernels"][L])
        fe[L] = RocketFeaturizer(L, C, nk, cfg["max_dilation"], seed + 1000 * L).build()
    return fe

def rocket_feature_dim(rocket_cfg: dict) -> int:
    cfg = normalize_rocket_cfg(rocket_cfg)
    return sum(2 * int(cfg["kernels"][L]) for L in ROCKET_WINDOWS)

def build_windows_for_batch_rows(symbols: np.ndarray,
                                entry_dates64: np.ndarray,
                                series_map: Dict[str, SymSeries],
                                L: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(symbols)
    C = len(ROCKET_CHANNELS)
    X = np.zeros((n, L, C), dtype=np.float32)
    ok = np.zeros((n,), dtype=np.uint8)

    uniq_syms = pd.unique(symbols)
    ar = np.arange(L, dtype=np.int64)[None, :]

    for sym in uniq_syms:
        s = series_map.get(sym)
        if s is None:
            continue
        pos = np.where(symbols == sym)[0]
        if pos.size == 0:
            continue

        dates64 = s.dates64
        seg = s.seg_id
        feats = s.feats

        ent = entry_dates64[pos]
        p = np.searchsorted(dates64, ent, side="left") - 1
        valid = p >= (L - 1)
        if not np.any(valid):
            continue

        pos_v = pos[valid]
        p_v = p[valid].astype(np.int64)
        start = p_v - (L - 1)

        seg_ok = (seg[p_v] == seg[start])
        if not np.any(seg_ok):
            continue

        pos_ok = pos_v[seg_ok]
        start_ok = start[seg_ok]
        take_idx = start_ok[:, None] + ar
        X[pos_ok, :, :] = feats[take_idx, :]
        ok[pos_ok] = 1

    return X, ok

def compute_rocket_features_for_df(df: pd.DataFrame, series_map: Dict[str, SymSeries], rocket_cfg: dict, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - F: (n, rocket_dim) float16
    - ok_all: (n,) uint8 where 1 means all windows OK across ROCKET_WINDOWS
    """
    cfg = normalize_rocket_cfg(rocket_cfg)
    n = len(df)
    d = rocket_feature_dim(cfg)
    F = np.zeros((n, d), dtype=FEATURE_DTYPE)
    ok_all = np.ones((n,), dtype=np.uint8)
    featurizers = build_featurizers(cfg, seed)

    offsets = {}
    off = 0
    for L in ROCKET_WINDOWS:
        span = 2 * int(cfg["kernels"][L])
        offsets[L] = (off, off + span)
        off += span

    symbols_all = df["symbol"].values.astype(str)
    entry_dates64_all = df["entry_date"].values.astype("datetime64[D]")

    for start in range(0, n, ROCKET_BATCH_SIZE):
        end = min(n, start + ROCKET_BATCH_SIZE)
        sy = symbols_all[start:end]
        ed = entry_dates64_all[start:end]

        ok_windows = []
        for L in ROCKET_WINDOWS:
            Xw, okw = build_windows_for_batch_rows(sy, ed, series_map, L=L)
            Fw = featurizers[L].transform_batch(Xw).astype(FEATURE_DTYPE)
            a, b = offsets[L]
            F[start:end, a:b] = Fw
            ok_windows.append(okw)

        okb = ok_windows[0].copy()
        for k in ok_windows[1:]:
            okb = (okb & k).astype(np.uint8)
        ok_all[start:end] = okb

    return F, ok_all

def add_cs_z_rank_inplace(df: pd.DataFrame, cols: List[str], group_col: str) -> List[str]:
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    g = df[group_col]
    mean = X.groupby(g, sort=False).transform("mean")
    std = X.groupby(g, sort=False).transform(lambda s: s.std(ddof=0))
    z = (X - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)
    rk = X.groupby(g, sort=False).rank(pct=True, method="average")
    out_cols = []
    for c in cols:
        zc = f"{c}_cs_z"
        rc = f"{c}_cs_rank"
        df[zc] = z[c].astype(np.float32)
        df[rc] = rk[c].astype(np.float32)
        out_cols += [zc, rc]
    return out_cols


# =========================================================
# Platt calibration
# =========================================================
@dataclass
class PlattCalibrator:
    a: float
    b: float
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        z = np.clip(self.a * scores + self.b, -50, 50)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

def fit_platt(scores: np.ndarray, y: np.ndarray) -> Optional[PlattCalibrator]:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(scores.reshape(-1, 1), y)
    return PlattCalibrator(a=float(lr.coef_.ravel()[0]), b=float(lr.intercept_.ravel()[0]))

def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -50, 50)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


# =========================================================
# Stage1 / Stage2 models
# =========================================================
@dataclass
class Stage2MagModel:
    edges: np.ndarray
    reps: np.ndarray
    clf: RandomForestClassifier

def _make_bins_from_quantiles(x: np.ndarray, qs: List[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 200:
        edges = [0.0, np.quantile(x, 0.7) if x.size else 1.0, np.quantile(x, 0.9) if x.size else 2.0, np.inf]
        return np.array(edges, dtype=float)
    cuts = [float(np.quantile(x, q)) for q in qs]
    cuts = [c for c in cuts if c > 0 and np.isfinite(c)]
    edges = [0.0] + sorted(list(dict.fromkeys(cuts))) + [np.inf]
    edges2 = [edges[0]]
    for v in edges[1:]:
        if v > edges2[-1] * 1.0000001:
            edges2.append(v)
    if len(edges2) < 4:
        m = float(np.median(x)) if x.size else 1.0
        edges2 = [0.0, m, m * 1.5 + 1e-6, np.inf]
    return np.array(edges2, dtype=float)

def _bin_indices(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = np.maximum(0.0, np.asarray(x, dtype=float))
    k = len(edges) - 1
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, k - 1).astype(int)

def _bin_representatives(x: np.ndarray, ybin: np.ndarray, k: int) -> np.ndarray:
    reps = np.zeros((k,), dtype=np.float32)
    for i in range(k):
        vals = x[ybin == i]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            reps[i] = 0.0
        elif vals.size < 50:
            reps[i] = float(np.median(vals))
        else:
            reps[i] = float(np.mean(vals))
    return reps.astype(np.float32)

def train_mag_bin_model(x_mag: np.ndarray, X: np.ndarray) -> Optional[Stage2MagModel]:
    x_mag = np.asarray(x_mag, dtype=float)
    x_mag = x_mag[np.isfinite(x_mag)]
    if x_mag.size < 200:
        return None
    edges = _make_bins_from_quantiles(x_mag, MAG_BIN_QUANTILES)
    ybin = _bin_indices(x_mag, edges)
    clf = RandomForestClassifier(**MAG_RF_PARAMS)
    clf.fit(X, ybin)
    reps = _bin_representatives(x_mag, ybin, k=(len(edges) - 1))
    return Stage2MagModel(edges=edges, reps=reps, clf=clf)

def predict_mag_expectation(model: Optional[Stage2MagModel], X: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros((X.shape[0],), dtype=np.float32)
    proba = model.clf.predict_proba(X).astype(np.float32)
    reps = model.reps.reshape(1, -1).astype(np.float32)
    return (proba * reps).sum(axis=1).astype(np.float32)

@dataclass
class Stage1Model:
    ridge: RidgeClassifier
    cal: Optional[PlattCalibrator]
    rf: RandomForestClassifier

def train_stage1(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, sample_weight: np.ndarray) -> Stage1Model:
    ridge = RidgeClassifier(alpha=RIDGE_ALPHA, random_state=42)
    ridge.fit(X_tr, y_tr, sample_weight=sample_weight)
    s_va = ridge.decision_function(X_va).astype(np.float32)
    cal = fit_platt(s_va, y_va)

    rf = RandomForestClassifier(**RF_PWIN_PARAMS)
    rf.fit(X_tr, y_tr, sample_weight=sample_weight)
    return Stage1Model(ridge=ridge, cal=cal, rf=rf)

def predict_pwin(stage1: Stage1Model, X: np.ndarray) -> np.ndarray:
    s = stage1.ridge.decision_function(X).astype(np.float32)
    p_r = stage1.cal.predict_proba(s) if stage1.cal is not None else sigmoid(s)
    p_f = stage1.rf.predict_proba(X)[:, 1].astype(np.float32)
    p = (W_PWIN_RIDGE * p_r + W_PWIN_RF * p_f).astype(np.float32)
    return np.clip(p, 1e-5, 1.0 - 1e-5)

@dataclass
class BucketModels:
    stage1: Stage1Model
    win_mag: Optional[Stage2MagModel]
    loss_mag: Optional[Stage2MagModel]


# =========================================================
# RR policy (optional)
# =========================================================
@dataclass
class RRPolicyPack:
    model: Ridge
    mean_r: float
    n_train: int


# =========================================================
# Rolling wR throttle gate (persistent)
# =========================================================
PFKey = Tuple[str, str, str, float, float]  # (regime, side, side_mode, stop_r, tgt_r)

@dataclass
class GateEvent:
    date: str
    key: str
    prev_mult: float
    new_mult: float
    wr: float
    n_trades: int
    n_days: int
    kill_streak: int
    revive_streak: int
    reason: str

@dataclass
class GateState:
    mult: float
    kill_streak: int
    revive_streak: int
    dq: deque  # of (exit_day_ts, r_used, w)
    day_counts: Counter
    sum_w: float
    sum_wr: float
    last_wr: float
    last_eval_cutoff: Optional[pd.Timestamp]

class RollingWRThrottleGate:
    def __init__(self, name: str):
        self.name = str(name)
        self.window_days = int(WR_GATE_WINDOW_DAYS)
        self.min_trades = int(WR_GATE_MIN_TRADES)
        self.min_days = int(WR_GATE_MIN_DAYS)
        self.kill_th = float(WR_GATE_KILL_TH)
        self.revive_th = float(WR_GATE_REVIVE_TH)
        self.kill_streak_req = int(WR_GATE_KILL_STREAK)
        self.revive_streak_req = int(WR_GATE_REVIVE_STREAK)
        self.throttle_mult = float(WR_GATE_THROTTLE_MULT)
        self.use_risk_weight = bool(WR_GATE_USE_RISK_WEIGHT)

        self.states: Dict[PFKey, GateState] = {}
        self.daily_rows: List[dict] = []
        self.events: List[GateEvent] = []
        self._exits_seen = 0

    @staticmethod
    def key_str(k: PFKey) -> str:
        r, s, sm, st, tg = k
        return f"{r}|{s}|{sm}|s{st:g}|t{tg:g}"

    def get_mult(self, key: PFKey) -> float:
        st = self.states.get(key)
        return float(st.mult) if st is not None else 1.0

    def add_exit(self, key: PFKey, exit_day: pd.Timestamp, r_used: float, risk_amt: float):
        exit_day = pd.Timestamp(exit_day).normalize()
        w = float(risk_amt) if self.use_risk_weight else 1.0
        r = float(r_used)

        st = self.states.get(key)
        if st is None:
            st = GateState(
                mult=1.0,
                kill_streak=0,
                revive_streak=0,
                dq=deque(),
                day_counts=Counter(),
                sum_w=0.0,
                sum_wr=0.0,
                last_wr=np.nan,
                last_eval_cutoff=None,
            )
            self.states[key] = st

        st.dq.append((exit_day, r, w))
        st.day_counts[exit_day] += 1
        st.sum_w += w
        st.sum_wr += (w * r)
        self._exits_seen += 1

    def _prune_to_cutoff(self, st: GateState, cutoff: pd.Timestamp):
        cutoff = pd.Timestamp(cutoff).normalize()
        lo = cutoff - pd.Timedelta(days=self.window_days - 1)

        while st.dq and st.dq[0][0] < lo:
            d0, r0, w0 = st.dq.popleft()
            st.sum_w -= float(w0)
            st.sum_wr -= float(w0) * float(r0)
            st.day_counts[d0] -= 1
            if st.day_counts[d0] <= 0:
                del st.day_counts[d0]

    def update_states_for_cutoff(self, trade_day: pd.Timestamp):
        trade_day = pd.Timestamp(trade_day).normalize()
        cutoff = (trade_day - pd.Timedelta(days=1)).normalize()

        n_keys = len(self.states)
        eval_wrs = []
        n_eval = 0
        n_throttled = 0
        n_events = 0

        for key, st in self.states.items():
            self._prune_to_cutoff(st, cutoff)

            n_tr = len(st.dq)
            n_d = len(st.day_counts)

            if n_tr < self.min_trades or n_d < self.min_days or st.sum_w <= 0:
                continue

            wr = float(st.sum_wr / max(1e-12, st.sum_w))
            st.last_wr = wr
            st.last_eval_cutoff = cutoff
            n_eval += 1
            eval_wrs.append(wr)

            if wr <= self.kill_th:
                st.kill_streak += 1
            else:
                st.kill_streak = 0

            if wr >= self.revive_th:
                st.revive_streak += 1
            else:
                st.revive_streak = 0

            prev = float(st.mult)
            new = prev

            if prev >= 0.999 and st.kill_streak >= self.kill_streak_req:
                new = self.throttle_mult
                st.revive_streak = 0
                self.events.append(GateEvent(
                    date=str(trade_day.date()),
                    key=self.key_str(key),
                    prev_mult=prev,
                    new_mult=new,
                    wr=wr,
                    n_trades=n_tr,
                    n_days=n_d,
                    kill_streak=st.kill_streak,
                    revive_streak=st.revive_streak,
                    reason="THROTTLE",
                ))
                n_events += 1

            if prev < 0.999 and st.revive_streak >= self.revive_streak_req:
                new = 1.0
                st.kill_streak = 0
                self.events.append(GateEvent(
                    date=str(trade_day.date()),
                    key=self.key_str(key),
                    prev_mult=prev,
                    new_mult=new,
                    wr=wr,
                    n_trades=n_tr,
                    n_days=n_d,
                    kill_streak=st.kill_streak,
                    revive_streak=st.revive_streak,
                    reason="UNTHROTTLE",
                ))
                n_events += 1

            st.mult = float(new)
            if st.mult < 0.999:
                n_throttled += 1

        if eval_wrs:
            arr = np.array(eval_wrs, dtype=float)
            row = dict(
                trade_day=str(trade_day.date()),
                cutoff_day=str(cutoff.date()),
                keys_total=int(n_keys),
                keys_evaluated=int(n_eval),
                keys_throttled=int(n_throttled),
                wr_min=float(np.min(arr)),
                wr_p25=float(np.quantile(arr, 0.25)),
                wr_median=float(np.median(arr)),
                wr_p75=float(np.quantile(arr, 0.75)),
                wr_max=float(np.max(arr)),
                exits_seen_total=int(self._exits_seen),
                events_today=int(n_events),
            )
        else:
            row = dict(
                trade_day=str(trade_day.date()),
                cutoff_day=str(cutoff.date()),
                keys_total=int(n_keys),
                keys_evaluated=0,
                keys_throttled=int(n_throttled),
                wr_min=np.nan, wr_p25=np.nan, wr_median=np.nan, wr_p75=np.nan, wr_max=np.nan,
                exits_seen_total=int(self._exits_seen),
                events_today=int(n_events),
            )
        self.daily_rows.append(row)


# =========================================================
# Paper state (open positions + ledger)
# =========================================================
@dataclass
class OpenPos:
    pos_id: int
    entry_date: str           # YYYY-MM-DD
    symbol: str
    side: str                 # long/short
    regime: str               # bull/bear
    side_mode: str            # both
    stop_r: float
    tgt_r: float
    entry_price: float
    stop_price: float
    target_price: float
    risk_amt: float
    gate_mult: float
    score: float
    last_mark_date: str       # last date checked for exit evaluation (YYYY-MM-DD)

def _pf_key(regime: str, side: str, side_mode: str, stop_r: float, tgt_r: float) -> PFKey:
    return (str(regime), str(side), str(side_mode), float(stop_r), float(tgt_r))

def _daily_capacity(equity: float, open_positions: List[OpenPos]) -> Tuple[int, float]:
    slots = max(0, int(MAX_OPEN_POSITIONS - len(open_positions)))
    open_risk = float(sum(p.risk_amt for p in open_positions))
    max_risk_amt = float(MAX_GROSS_RISK_PCT * equity)
    rem_risk_amt = max(0.0, max_risk_amt - open_risk)
    r_per = float(RISK_PCT_PER_TRADE * equity)
    by_risk = int(rem_risk_amt // max(1e-9, r_per))
    cap = int(min(MAX_NEW_ENTRIES_PER_DAY, slots, by_risk))
    return cap, open_risk / max(1e-9, equity)

def load_state():
    state_path = os.path.join(OUT_DIR, "state", "state.json")
    if os.path.exists(state_path):
        st = json.load(open(state_path, "r"))
    else:
        st = dict(equity=float(START_EQUITY), pos_counter=0, last_run_date="")
        json.dump(st, open(state_path, "w"), indent=2)
    return st

def save_state(st: dict):
    state_path = os.path.join(OUT_DIR, "state", "state.json")
    json.dump(st, open(state_path, "w"), indent=2)

def load_open_positions() -> List[OpenPos]:
    fp = os.path.join(OUT_DIR, "state", "open_positions.csv")
    if not os.path.exists(fp):
        return []
    df = pd.read_csv(fp)
    out = []
    for r in df.to_dict("records"):
        out.append(OpenPos(**r))
    return out

def save_open_positions(ops: List[OpenPos]):
    fp = os.path.join(OUT_DIR, "state", "open_positions.csv")
    if not ops:
        pd.DataFrame(columns=list(OpenPos.__annotations__.keys())).to_csv(fp, index=False)
        return
    pd.DataFrame([asdict(p) for p in ops]).to_csv(fp, index=False)

def append_ledger(rows: List[dict]):
    fp = os.path.join(OUT_DIR, "state", "trade_ledger.csv")
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if os.path.exists(fp):
        df_old = pd.read_csv(fp)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(fp, index=False)

def load_ledger() -> pd.DataFrame:
    fp = os.path.join(OUT_DIR, "state", "trade_ledger.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    return pd.read_csv(fp)


# =========================================================
# Training data loader
# =========================================================
_RE_TR = re.compile(r"^trades_(bull|bear)(?:_(both|long|short))?_s([0-9_.]+)_t([0-9_.]+)\.(parquet|csv)$", re.IGNORECASE)

def _num_to_float(s: str) -> float:
    return float(str(s).replace("_", "."))

def resolve_root_dir(path_in, extract_to_dir):
    if os.path.isdir(path_in):
        return path_in
    if os.path.isfile(path_in) and path_in.lower().endswith(".zip"):
        if os.path.isdir(extract_to_dir) and any(os.scandir(extract_to_dir)):
            return extract_to_dir
        os.makedirs(extract_to_dir, exist_ok=True)
        with zipfile.ZipFile(path_in, "r") as zf:
            zf.extractall(extract_to_dir)
        return extract_to_dir
    raise FileNotFoundError(f"Path must be a directory or a .zip file. Got: {path_in}")

def locate_portfolio_run_dir(root_dir):
    cand = os.path.join(root_dir, "portfolio_run")
    if os.path.isdir(cand):
        return cand
    for r, _d, _f in os.walk(root_dir):
        if os.path.basename(r).lower() == "portfolio_run":
            return r
    raise FileNotFoundError(f"Could not locate 'portfolio_run' under: {root_dir}")

def locate_trades_dir(portfolio_run_dir):
    cand = os.path.join(portfolio_run_dir, "trades")
    if os.path.isdir(cand):
        return cand
    for r, _d, _f in os.walk(portfolio_run_dir):
        if os.path.basename(r).lower() == "trades":
            return r
    raise FileNotFoundError(f"Could not locate 'trades' under: {portfolio_run_dir}")

def load_all_trades_from_dir(trades_dir: str) -> pd.DataFrame:
    parqs = [os.path.join(trades_dir, f) for f in os.listdir(trades_dir) if f.lower().endswith(".parquet")]
    csvs  = [os.path.join(trades_dir, f) for f in os.listdir(trades_dir) if f.lower().endswith(".csv")]
    if not parqs and not csvs:
        raise RuntimeError(f"No trade files found in {trades_dir}")

    dfs = []
    if parqs:
        for p in sorted(parqs):
            df = pd.read_parquet(p)
            df["__srcfile"] = os.path.basename(p)
            dfs.append(df)
    else:
        for p in sorted(csvs):
            df = pd.read_csv(p)
            df["__srcfile"] = os.path.basename(p)
            dfs.append(df)

    t = pd.concat(dfs, ignore_index=True)
    t.columns = [c.strip() for c in t.columns]

    def pick(names):
        for n in names:
            if n in t.columns:
                return n
        return None

    c_entry = pick(["EntryDate", "entry_date", "entry_dt"])
    c_exit  = pick(["ExitDate", "exit_date", "exit_dt"])
    c_sym   = pick(["Symbol", "symbol", "Ticker", "ticker", "tradingsymbol", "TradingSymbol"])
    c_r     = pick(["Rmult", "rmult", "RMultiple"])
    c_reg   = pick(["Regime", "regime"])
    c_side  = pick(["Side", "side"])
    c_smode = pick(["SideMode", "side_mode", "PortfolioSide", "portfolio_side"])
    c_stop  = pick(["StopATR", "stop_atr"])
    c_tgt   = pick(["TargetATR", "target_atr"])

    missing = [k for k, v in [("EntryDate", c_entry), ("ExitDate", c_exit), ("Ticker/Symbol", c_sym), ("Rmult", c_r)] if v is None]
    if missing:
        raise RuntimeError(f"Trade logs missing required columns: {missing}")

    t["entry_date"] = s_day(t[c_entry], fmt=None, default_dayfirst=False)
    t["exit_date"]  = s_day(t[c_exit],  fmt=None, default_dayfirst=False)
    t["symbol"]     = t[c_sym].astype(str).str.upper().str.strip()
    t["rmult"]      = pd.to_numeric(t[c_r], errors="coerce")

    t["regime"] = t[c_reg].astype(str).str.lower().str.strip() if c_reg else "bull"
    t["side"] = t[c_side].astype(str).str.lower().str.strip() if c_side else "long"
    t["side_mode"] = t[c_smode].astype(str).str.lower().str.strip() if c_smode else "both"
    t["stop_atr"] = pd.to_numeric(t[c_stop], errors="coerce") if c_stop else np.nan
    t["target_atr"] = pd.to_numeric(t[c_tgt], errors="coerce") if c_tgt else np.nan

    t = t.dropna(subset=["symbol", "entry_date", "exit_date", "rmult", "stop_atr", "target_atr"]).copy()
    if EXCLUDE_ETF:
        t = t[~t["symbol"].map(_is_etf_symbol)].copy()

    t["stop_r"] = pd.to_numeric(t["stop_atr"], errors="coerce").round(6)
    t["tgt_r"]  = pd.to_numeric(t["target_atr"], errors="coerce").round(6)
    t["y_win"] = (pd.to_numeric(t["rmult"], errors="coerce") > 0).astype(int)

    assert_day_dt(t, "entry_date", name="[TRADES] ")
    assert_day_dt(t, "exit_date",  name="[TRADES] ")

    return t[["symbol","entry_date","exit_date","regime","side","side_mode","stop_atr","target_atr","stop_r","tgt_r","rmult","y_win"]].copy()


# =========================================================
# Gap skip for training trades (uses seg_id at entry and exit)
# =========================================================
def apply_gap_skip(trades: pd.DataFrame, seg_map: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    assert_day_dt(t, "entry_date", name="[GAP/TRADES] ")
    assert_day_dt(t, "exit_date",  name="[GAP/TRADES] ")
    assert_day_dt(seg_map, "date", name="[GAP/SEG] ")

    t = t.merge(seg_map.rename(columns={"date": "entry_date", "seg_id": "entry_seg"}),
                on=["symbol", "entry_date"], how="left")
    t = t.merge(seg_map.rename(columns={"date": "exit_date", "seg_id": "exit_seg"}),
                on=["symbol", "exit_date"], how="left")

    t["gap_skip"] = 0
    t.loc[t["entry_seg"].isna() | t["exit_seg"].isna(), "gap_skip"] = 1
    t.loc[(t["entry_seg"].notna()) & (t["exit_seg"].notna()) & (t["entry_seg"] != t["exit_seg"]), "gap_skip"] = 1
    return t


# =========================================================
# Training split helper
# =========================================================
def pick_val_dates(train_dates: np.ndarray) -> set:
    n = len(train_dates)
    if n <= 0:
        return set()
    vd = int(round(n * float(VALID_FRAC)))
    vd = max(VALID_DAYS_MIN, min(VALID_DAYS_MAX, vd))
    vd = min(vd, max(10, n // 2))
    return set(train_dates[-vd:])

def subsample_stratified(df: pd.DataFrame, ycol: str, nmax: int, seed: int) -> pd.DataFrame:
    if len(df) <= nmax:
        return df
    rng = np.random.RandomState(seed)
    y = df[ycol].astype(int).values
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        take = rng.choice(np.arange(len(df)), size=nmax, replace=False)
        return df.iloc[take].copy()
    n1 = min(idx1.size, nmax // 2)
    n0 = min(idx0.size, nmax - n1)
    take = np.concatenate([
        rng.choice(idx1, size=n1, replace=False),
        rng.choice(idx0, size=n0, replace=False),
    ])
    rng.shuffle(take)
    return df.iloc[take].copy()

def subsample_random(df: pd.DataFrame, nmax: int, seed: int) -> pd.DataFrame:
    if len(df) <= nmax:
        return df
    rng = np.random.RandomState(seed)
    take = rng.choice(df.index.values, size=nmax, replace=False)
    return df.loc[take].copy()


# =========================================================
# Model training (daily retrain)
# =========================================================
def build_training_window(run_date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    cutoff = (run_date - pd.Timedelta(days=1)).normalize()
    start = (cutoff - pd.Timedelta(days=int(365.25 * TRAIN_LOOKBACK_YEARS))).normalize()
    return start, cutoff

def build_static_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (F_static float16, cols)
    """
    if not ADD_STATIC_FEATURES or df.empty:
        return np.zeros((len(df), 0), dtype=FEATURE_DTYPE), []

    df = df.copy()
    df["stop_atr_f"] = pd.to_numeric(df["stop_atr"], errors="coerce").fillna(0.0).astype(np.float32)
    df["target_atr_f"] = pd.to_numeric(df["target_atr"], errors="coerce").fillna(0.0).astype(np.float32)
    df["rr_ratio"] = (df["target_atr_f"] / np.maximum(1e-6, df["stop_atr_f"])).astype(np.float32)
    df["rr_diff"]  = (df["target_atr_f"] - df["stop_atr_f"]).astype(np.float32)
    df["side_is_long"] = (df["side"] == "long").astype(np.float32)

    base_cols = ["stop_atr_f","target_atr_f","rr_ratio","rr_diff"]
    cs_cols = add_cs_z_rank_inplace(df, cols=base_cols, group_col="entry_date")
    static_cols = base_cols + cs_cols + ["side_is_long"]

    F = df[static_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(np.float32).values.astype(FEATURE_DTYPE)
    return F, static_cols

def score_expr(bucket_models: Dict[Tuple[str,str], BucketModels], X: np.ndarray, regimes: np.ndarray, sides: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    out = np.full((n,), SENTINEL_SCORE, dtype=np.float32)
    valid = np.zeros((n,), dtype=np.uint8)
    for regime in ["bull","bear"]:
        for side in ["long","short"]:
            mask = (regimes == regime) & (sides == side)
            if not np.any(mask):
                continue
            pack = bucket_models.get((regime, side))
            if pack is None:
                continue
            Xp = X[mask]
            pwin = predict_pwin(pack.stage1, Xp)
            Ew = predict_mag_expectation(pack.win_mag, Xp)
            El = predict_mag_expectation(pack.loss_mag, Xp)
            out[mask] = (pwin * Ew) - ((1.0 - pwin) * El)
            valid[mask] = 1
    return out, valid

def train_bucket_models_daily(train_df: pd.DataFrame, series_map: Dict[str, SymSeries], run_tag: str) -> Tuple[Dict[Tuple[str,str], BucketModels], pd.DataFrame]:
    """
    Trains 4 bucket models (bull/bear x long/short) on train_df.
    Uses internal no-leak val split based on last VALID_FRAC of entry dates,
    with the key constraint: train exits < val_min.
    """
    if train_df.empty:
        raise RuntimeError("train_df empty")

    # features
    F_rocket, ok_all = compute_rocket_features_for_df(train_df, series_map, ROCKET_CFG, ROCKET_RANDOM_SEED)
    F_static, static_cols = build_static_features(train_df)

    # join features
    X_all = np.asarray(F_rocket, dtype=TRAIN_DTYPE)
    if F_static.shape[1] > 0:
        X_all = np.hstack([X_all, np.asarray(F_static, dtype=TRAIN_DTYPE)]).astype(TRAIN_DTYPE, copy=False)

    df = train_df.copy()
    df["rocket_ok_all"] = ok_all.astype(int)
    df["_idx"] = np.arange(len(df), dtype=np.int64)

    models: Dict[Tuple[str,str], BucketModels] = {}
    stats_rows = []

    for regime in ["bull","bear"]:
        for side in ["long","short"]:
            sub = df[(df["regime"] == regime) & (df["side"] == side)].copy()
            if sub.empty:
                stats_rows.append(dict(run=run_tag, regime=regime, side=side, status="no_rows"))
                continue

            ok_rate = float((sub["rocket_ok_all"] == 1).mean())
            sub_ok = sub[sub["rocket_ok_all"] == 1].copy()
            if len(sub_ok) >= max(MIN_TRAIN_ROWS, int(0.6 * len(sub))):
                sub = sub_ok

            dates = np.array(sorted(sub["entry_date"].unique()))
            if dates.size < 5:
                stats_rows.append(dict(run=run_tag, regime=regime, side=side, status="too_few_dates", n_total=int(len(sub))))
                continue
            val_dates = pick_val_dates(dates)
            vd = max(VALID_DAYS_MIN, min(len(val_dates), len(dates)))
            tail = set(dates[-vd:])
            val_min = pd.Timestamp(dates[-vd]).normalize()

            tr_base = sub[sub["exit_date"] < val_min].copy()
            va = sub[sub["entry_date"].isin(tail)].copy()

            if len(tr_base) < MIN_TRAIN_ROWS or len(va) < MIN_VAL_ROWS or tr_base["y_win"].nunique() < 2 or va["y_win"].nunique() < 2:
                stats_rows.append(dict(
                    run=run_tag, regime=regime, side=side, status="split_infeasible",
                    n_total=int(len(sub)), ok_rate=float(ok_rate),
                    n_train=int(len(tr_base)), n_val=int(len(va))
                ))
                continue

            tr_base = subsample_stratified(tr_base, "y_win", MAX_TRAIN_SAMPLES_PER_BUCKET, seed=101)
            va = subsample_stratified(va, "y_win", MAX_VAL_SAMPLES_PER_BUCKET, seed=202)

            idx_tr = tr_base["_idx"].values.astype(np.int64)
            idx_va = va["_idx"].values.astype(np.int64)

            X_tr = X_all[idx_tr]
            X_va = X_all[idx_va]
            y_tr = tr_base["y_win"].astype(int).values
            y_va = va["y_win"].astype(int).values

            pos = int((y_tr == 1).sum())
            neg = int((y_tr == 0).sum())
            w_pos = 0.5 / max(1, pos)
            w_neg = 0.5 / max(1, neg)
            sw = np.where(y_tr == 1, w_pos, w_neg).astype(np.float32)

            stage1 = train_stage1(X_tr, y_tr, X_va, y_va, sample_weight=sw)

            r_tr = pd.to_numeric(tr_base["rmult"], errors="coerce").values.astype(float)
            r_tr = np.clip(r_tr, -R_ABS_MAX_SANITY, R_ABS_MAX_SANITY)
            win_mask = (r_tr > 0)
            loss_mask = (r_tr < 0)

            win_mag = train_mag_bin_model(r_tr[win_mask], X_tr[win_mask]) if win_mask.sum() >= 400 else None
            loss_mag = train_mag_bin_model(np.abs(r_tr[loss_mask]), X_tr[loss_mask]) if loss_mask.sum() >= 400 else None

            pwin_va = predict_pwin(stage1, X_va)
            pr = float(average_precision_score(y_va, pwin_va)) if len(np.unique(y_va)) > 1 else np.nan
            roc = float(roc_auc_score(y_va, pwin_va)) if len(np.unique(y_va)) > 1 else np.nan

            models[(regime, side)] = BucketModels(stage1=stage1, win_mag=win_mag, loss_mag=loss_mag)
            stats_rows.append(dict(
                run=run_tag, regime=regime, side=side, status="ok",
                n_total=int(len(sub)), ok_rate=float(ok_rate),
                n_train=int(len(tr_base)), n_val=int(len(va)),
                pr_auc=float(pr) if pd.notna(pr) else np.nan,
                roc_auc=float(roc) if pd.notna(roc) else np.nan,
            ))

            log.info(f"[TRAIN {run_tag}][{regime}/{side}] ok_rate={ok_rate:.3f} n_train={len(tr_base):,} n_val={len(va):,} PR={pr:.4f} ROC={roc:.4f}")

    return models, pd.DataFrame(stats_rows)

def train_rr_policy_daily(train_df: pd.DataFrame, series_map: Dict[str, SymSeries], run_tag: str) -> Tuple[Dict[Tuple[str,str,float,float], RRPolicyPack], Dict[Tuple[float,float], RRPolicyPack], pd.DataFrame]:
    if not RR_POLICY_ENABLE:
        return {}, {}, pd.DataFrame()

    # features
    F_rocket, ok_all = compute_rocket_features_for_df(train_df, series_map, ROCKET_CFG, ROCKET_RANDOM_SEED)
    F_static, _ = build_static_features(train_df)
    X_all = np.asarray(F_rocket, dtype=TRAIN_DTYPE)
    if F_static.shape[1] > 0:
        X_all = np.hstack([X_all, np.asarray(F_static, dtype=TRAIN_DTYPE)]).astype(TRAIN_DTYPE, copy=False)

    df = train_df.copy()
    df["rocket_ok_all"] = ok_all.astype(int)
    df["_idx"] = np.arange(len(df), dtype=np.int64)

    # prefer rocket-ok rows if enough
    df_ok = df[df["rocket_ok_all"] == 1].copy()
    if len(df_ok) >= max(MIN_TRAIN_ROWS, int(0.6 * len(df))):
        df = df_ok

    r = pd.to_numeric(df["rmult"], errors="coerce").fillna(0.0).values.astype(np.float32)
    r = np.clip(r, -RR_POLICY_PRED_CLIP, RR_POLICY_PRED_CLIP).astype(np.float32)
    df["_r_target"] = r

    bucket_rr: Dict[Tuple[str,str,float,float], RRPolicyPack] = {}
    global_rr: Dict[Tuple[float,float], RRPolicyPack] = {}
    stats = []

    if RR_POLICY_BUCKET_AWARE:
        for (regime, side, stop_r, tgt_r), g in df.groupby(["regime","side","stop_r","tgt_r"], sort=False):
            n = len(g)
            if n < RR_POLICY_MIN_ROWS_PER_PAIR:
                continue
            g2 = subsample_random(g, RR_POLICY_MAX_ROWS_PER_PAIR, seed=777)
            idx = g2["_idx"].values.astype(np.int64)
            X = X_all[idx]
            y = g2["_r_target"].values.astype(np.float32)
            mdl = Ridge(alpha=float(RR_POLICY_RIDGE_ALPHA), fit_intercept=True, solver=str(RR_POLICY_RIDGE_SOLVER))
            mdl.fit(X, y)
            pack = RRPolicyPack(model=mdl, mean_r=float(np.mean(y)), n_train=int(len(g2)))
            bucket_rr[(str(regime), str(side), float(stop_r), float(tgt_r))] = pack
            stats.append(dict(run=run_tag, level="bucket", regime=regime, side=side, stop_r=float(stop_r), tgt_r=float(tgt_r), n_train=int(len(g2))))

    if RR_POLICY_GLOBAL_FALLBACK:
        for (stop_r, tgt_r), g in df.groupby(["stop_r","tgt_r"], sort=False):
            n = len(g)
            if n < RR_POLICY_MIN_ROWS_PER_PAIR:
                continue
            g2 = subsample_random(g, RR_POLICY_MAX_ROWS_PER_PAIR, seed=778)
            idx = g2["_idx"].values.astype(np.int64)
            X = X_all[idx]
            y = g2["_r_target"].values.astype(np.float32)
            mdl = Ridge(alpha=float(RR_POLICY_RIDGE_ALPHA), fit_intercept=True, solver=str(RR_POLICY_RIDGE_SOLVER))
            mdl.fit(X, y)
            pack = RRPolicyPack(model=mdl, mean_r=float(np.mean(y)), n_train=int(len(g2)))
            global_rr[(float(stop_r), float(tgt_r))] = pack
            stats.append(dict(run=run_tag, level="global", stop_r=float(stop_r), tgt_r=float(tgt_r), n_train=int(len(g2))))

    return bucket_rr, global_rr, pd.DataFrame(stats)


def compute_baseline_proxy_score(train_df: pd.DataFrame) -> float:
    """
    Notebook parity proxy for missing-bucket expr scores.
    Uses a simple "winner-slice" analogue: mean positive R-multiple from training winners.
    Falls back to overall mean R, then 0.0 if unavailable.
    """
    if train_df is None or train_df.empty or "rmult" not in train_df.columns:
        return 0.0
    r = pd.to_numeric(train_df["rmult"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if r.notna().sum() == 0:
        return 0.0
    wins = r[r > 0]
    if len(wins) > 0:
        return float(np.nanmean(wins.values.astype(np.float32)))
    return float(np.nanmean(r.values.astype(np.float32)))

def rr_policy_predict(df: pd.DataFrame, bucket_rr: Dict[Tuple[str,str,float,float], RRPolicyPack], global_rr: Dict[Tuple[float,float], RRPolicyPack],
                      X: np.ndarray) -> np.ndarray:
    n = len(df)
    out = np.full((n,), np.nan, dtype=np.float32)
    if n == 0 or (not RR_POLICY_ENABLE):
        return out

    regimes = df["regime"].astype(str).values
    sides = df["side"].astype(str).values
    stop_r = pd.to_numeric(df["stop_r"], errors="coerce").values.astype(float)
    tgt_r  = pd.to_numeric(df["tgt_r"], errors="coerce").values.astype(float)

    keys = list(zip(regimes, sides, stop_r, tgt_r))
    uniq = {}
    for i, k in enumerate(keys):
        uniq.setdefault(k, []).append(i)

    for k, pos in uniq.items():
        regime, side, s_r, t_r = k
        if not np.isfinite(s_r) or not np.isfinite(t_r):
            continue
        pack = None
        if RR_POLICY_BUCKET_AWARE:
            pack = bucket_rr.get((str(regime), str(side), float(round(s_r, 6)), float(round(t_r, 6))))
        if pack is None and RR_POLICY_GLOBAL_FALLBACK:
            pack = global_rr.get((float(round(s_r, 6)), float(round(t_r, 6))))
        if pack is None:
            continue
        pos = np.array(pos, dtype=np.int64)
        yhat = pack.model.predict(X[pos]).astype(np.float32)
        yhat = np.clip(yhat, -RR_POLICY_PRED_CLIP, RR_POLICY_PRED_CLIP).astype(np.float32)
        out[pos] = yhat

    return out


# =========================================================
# Divergence signals -> RR candidates
# =========================================================
def load_divergences(fp: str) -> pd.DataFrame:
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    d = pd.read_csv(fp)
    d.columns = [c.strip() for c in d.columns]
    req = ["Ticker","Kind","SignalDate","EntryDate","ATRMult","ATRPeriod"]
    for r in req:
        if r not in d.columns:
            raise RuntimeError(f"Divergences file missing required column: {r}. Columns={d.columns.tolist()}")
    d["symbol"] = d["Ticker"].astype(str).str.upper().str.strip()
    d["kind"] = d["Kind"].astype(str).str.lower().str.strip()
    d["signal_date"] = s_day(d["SignalDate"], fmt=None, default_dayfirst=False)
    d["entry_date"] = s_day(d["EntryDate"], fmt=None, default_dayfirst=False)
    d["atr_mult"] = pd.to_numeric(d["ATRMult"], errors="coerce").fillna(1.0).astype(np.float32)
    d["atr_period"] = pd.to_numeric(d["ATRPeriod"], errors="coerce").fillna(ATR_PERIOD_DEFAULT).astype(int)
    d = d.dropna(subset=["symbol","signal_date","entry_date"]).copy()
    assert_day_dt(d, "signal_date", name="[DIV] ")
    assert_day_dt(d, "entry_date", name="[DIV] ")
    return d

def kind_to_side_regime(kind: str) -> Tuple[str, str]:
    # Simple mapping for paper trading:
    # bullish -> long + bull ; bearish -> short + bear
    # (If you have a separate regime detector, replace this.)
    k = str(kind).lower()
    if "bear" in k:
        return "short", "bear"
    if "bull" in k:
        return "long", "bull"
    # fallback
    return "long", "bull"

def make_rr_candidates(div_df: pd.DataFrame, run_date: pd.Timestamp) -> pd.DataFrame:
    """
    Builds candidates only for EntryDate == run_date.
    Expands RR combos: (stop_r, tgt_r).
    """
    d0 = div_df[div_df["entry_date"] == pd.Timestamp(run_date).normalize()].copy()
    if d0.empty:
        return d0

    rows = []
    for r in d0.itertuples(index=False):
        stop_list = STOP_ATR_LIST[:] if STOP_ATR_LIST else [float(r.atr_mult)]
        for stop_r in stop_list:
            for rr in RR_LIST:
                tgt_r = float(stop_r) * float(rr)
                side, regime = kind_to_side_regime(r.kind)
                rows.append(dict(
                    symbol=str(r.symbol),
                    signal_date=pd.Timestamp(r.signal_date).normalize(),
                    entry_date=pd.Timestamp(r.entry_date).normalize(),
                    side=side,
                    regime=regime,
                    side_mode="both",
                    stop_atr=float(stop_r),
                    target_atr=float(tgt_r),
                    stop_r=float(round(stop_r, 6)),
                    tgt_r=float(round(tgt_r, 6)),
                    atr_period=int(r.atr_period),
                ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    assert_day_dt(out, "entry_date", name="[CAND] ")
    assert_day_dt(out, "signal_date", name="[CAND] ")
    return out


# =========================================================
# Price helpers for paper trades (entry/stop/target + exit simulation)
# =========================================================
def get_ohlc_at(series: SymSeries, day: pd.Timestamp) -> Optional[Tuple[float,float,float,float,float]]:
    """
    Returns (open, high, low, close, atr) for the given day, or None if missing.
    """
    day64 = np.datetime64(pd.Timestamp(day).normalize().date())
    i = np.searchsorted(series.dates64, day64)
    if i >= len(series.dates64) or series.dates64[i] != day64:
        return None
    o,h,l,c = series.ohlc[i]
    atr = series.atr[i] if series.atr is not None else np.nan
    return float(o), float(h), float(l), float(c), float(atr)

def simulate_exit_from_entry(
    series: SymSeries,
    entry_day: pd.Timestamp,
    side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    max_hold_days: int,
    cutoff_day: pd.Timestamp,
) -> Tuple[Optional[pd.Timestamp], Optional[str], Optional[float]]:
    """
    Walk forward from entry_day to min(entry_day+max_hold_days, cutoff_day) (calendar days),
    using DAILY OHLC to decide stop/target/time exits.

    Exit policy:
      1) OPEN gap exits take priority:
         - LONG: if Open <= stop => exit at Open; if Open >= target => exit at Open
         - SHORT: if Open >= stop => exit at Open; if Open <= target => exit at Open
      2) Intrabar (same-day) stop/target:
         - If both stop and target touched in the same bar, use SAME_DAY_BOTH_HIT_POLICY
         - Exit price is the stop/target level (paper assumption)
      3) TIME exit at the last evaluated day close.

    Returns: (exit_day, exit_reason, exit_price). If no exit by cutoff_day => (None,None,None).
    """
    entry_day = pd.Timestamp(entry_day).normalize()
    cutoff_day = pd.Timestamp(cutoff_day).normalize()
    side = str(side).lower()

    day64 = np.datetime64(entry_day.date())
    i0 = np.searchsorted(series.dates64, day64)
    if i0 >= len(series.dates64) or series.dates64[i0] != day64:
        return None, None, None

    last_day = min(cutoff_day, entry_day + pd.Timedelta(days=int(max_hold_days)))
    last64 = np.datetime64(last_day.date())
    i1 = np.searchsorted(series.dates64, last64, side="right") - 1
    if i1 < i0:
        return None, None, None

    for i in range(i0, i1 + 1):
        d = pd.Timestamp(str(series.dates64[i])).normalize()
        o, h, l, c = series.ohlc[i]

        # 1) OPEN gap exits (priority)
        if side == "long":
            if o <= stop_price:
                return d, "OPEN_STOP", float(o)
            if o >= target_price:
                return d, "OPEN_TARGET", float(o)
        else:
            if o >= stop_price:
                return d, "OPEN_STOP", float(o)
            if o <= target_price:
                return d, "OPEN_TARGET", float(o)

        # 2) Intrabar stop/target
        if side == "long":
            hit_stop = (l <= stop_price)
            hit_tgt = (h >= target_price)
        else:
            hit_stop = (h >= stop_price)
            hit_tgt = (l <= target_price)

        if hit_stop and hit_tgt:
            if SAME_DAY_BOTH_HIT_POLICY == "target_first":
                return d, "TARGET_AND_STOP_TARGET_FIRST", float(target_price)
            return d, "TARGET_AND_STOP_STOP_FIRST", float(stop_price)

        if hit_stop:
            return d, "STOP", float(stop_price)
        if hit_tgt:
            return d, "TARGET", float(target_price)

    # 3) TIME exit at last_day close
    if last_day <= cutoff_day:
        px = get_ohlc_at(series, last_day)
        if px is None:
            return None, None, None
        return last_day, "TIME", float(px[3])

    return None, None, None
  
def rmult_from_exit(side: str, entry_price: float, stop_price: float, target_price: float, exit_price: float) -> float:
    side = str(side).lower()
    if side == "long":
        risk_per_share = max(1e-12, entry_price - stop_price)
        return float((exit_price - entry_price) / risk_per_share)
    else:
        risk_per_share = max(1e-12, stop_price - entry_price)
        return float((entry_price - exit_price) / risk_per_share)

def qty_from_risk(risk_amt: float, entry_price: float, stop_price: float, side: str) -> int:
    side = str(side).lower()
    if side == "long":
        per_share = max(1e-9, entry_price - stop_price)
    else:
        per_share = max(1e-9, stop_price - entry_price)
    q = int(math.floor(float(risk_amt) / float(per_share)))
    return max(0, q)


# =========================================================
# Main daily run
# =========================================================
def get_run_date() -> pd.Timestamp:
    env = os.environ.get(RUN_DATE_ENV, "").strip()
    if env:
        return pd.Timestamp(env).normalize()
    # fallback: now in Asia/Kolkata
    try:
        return pd.Timestamp.now(tz="Asia/Kolkata").tz_convert(None).normalize()
    except Exception:
        return pd.Timestamp.now().normalize()

def load_or_init_gate() -> RollingWRThrottleGate:
    fp = os.path.join(OUT_DIR, "state", "wr_gate.pkl")
    if os.path.exists(fp):
        try:
            g = joblib.load(fp)
            if isinstance(g, RollingWRThrottleGate):
                return g
        except Exception:
            pass
    return RollingWRThrottleGate("paper")

def save_gate(g: RollingWRThrottleGate):
    fp = os.path.join(OUT_DIR, "state", "wr_gate.pkl")
    joblib.dump(g, fp)

def main():
    t0 = time.time()
    run_date = get_run_date()
    cutoff = (run_date - pd.Timedelta(days=1)).normalize()
    run_tag = run_date.strftime("%Y%m%d")
    run_dir = os.path.join(OUT_DIR, "run", run_tag)
    os.makedirs(run_dir, exist_ok=True)

    log.info(f"[RUN] run_date={run_date.date()} cutoff={cutoff.date()}")

    # load paper state
    st = load_state()
    equity = float(st.get("equity", START_EQUITY))
    pos_counter = int(st.get("pos_counter", 0))
    open_positions = load_open_positions()

    # load divergence signals
    div = load_divergences(DIVERGENCES_CSV)

    # training dataset: baseline trades OR user-provided closed trades
    if USE_BASELINE_TRADESET:
        baseline_extract = os.path.join(OUT_DIR, "caches", "_baseline_extracted")
        root = resolve_root_dir(BASELINE_RESULTS_PATH, baseline_extract)
        pr = locate_portfolio_run_dir(root)
        trades_dir = locate_trades_dir(pr)
        train_trades = load_all_trades_from_dir(trades_dir)
    else:
        if not os.path.exists(TRAIN_TRADES_CSV):
            raise FileNotFoundError(TRAIN_TRADES_CSV)
        tt = pd.read_csv(TRAIN_TRADES_CSV)
        tt.columns = [c.strip() for c in tt.columns]
        # normalize expected cols
        need = ["symbol","entry_date","exit_date","side","rmult","stop_atr","target_atr"]
        for c in need:
            if c not in tt.columns:
                raise RuntimeError(f"TRAIN_TRADES_CSV missing col: {c}")
        tt["symbol"] = tt["symbol"].astype(str).str.upper().str.strip()
        tt["entry_date"] = s_day(tt["entry_date"], fmt=None, default_dayfirst=False)
        tt["exit_date"]  = s_day(tt["exit_date"], fmt=None, default_dayfirst=False)
        tt["side"] = tt["side"].astype(str).str.lower().str.strip()
        tt["regime"] = np.where(tt["side"].eq("short"), "bear", "bull")
        tt["side_mode"] = "both"
        tt["stop_atr"] = pd.to_numeric(tt["stop_atr"], errors="coerce")
        tt["target_atr"] = pd.to_numeric(tt["target_atr"], errors="coerce")
        tt["stop_r"] = pd.to_numeric(tt["stop_atr"], errors="coerce").round(6)
        tt["tgt_r"]  = pd.to_numeric(tt["target_atr"], errors="coerce").round(6)
        tt["y_win"] = (pd.to_numeric(tt["rmult"], errors="coerce") > 0).astype(int)
        train_trades = tt.dropna(subset=["symbol","entry_date","exit_date","rmult","stop_atr","target_atr"]).copy()

    # append closed PAPER trades into training set (optional; improves daily adaptation)
    led = load_ledger()
    if not led.empty:
        # required columns if present
        # We store from this script: symbol, entry_date, exit_date, side, stop_r, tgt_r, rmult_used
        cols = {c.lower(): c for c in led.columns}
        if "rmult_used" in cols and "entry_date" in cols and "exit_date" in cols:
            p = led.copy()
            p.columns = [c.strip() for c in p.columns]
            p["symbol"] = p["symbol"].astype(str).str.upper().str.strip()
            p["entry_date"] = s_day(p["entry_date"], fmt=None, default_dayfirst=False)
            p["exit_date"]  = s_day(p["exit_date"], fmt=None, default_dayfirst=False)
            p["side"] = p["side"].astype(str).str.lower().str.strip()
            p["regime"] = p.get("regime", np.where(p["side"].eq("short"), "bear", "bull"))
            p["side_mode"] = p.get("side_mode", "both")
            p["stop_atr"] = pd.to_numeric(p.get("stop_r", np.nan), errors="coerce")
            p["target_atr"] = pd.to_numeric(p.get("tgt_r", np.nan), errors="coerce")
            p["stop_r"] = pd.to_numeric(p["stop_atr"], errors="coerce").round(6)
            p["tgt_r"]  = pd.to_numeric(p["target_atr"], errors="coerce").round(6)
            p["rmult"] = pd.to_numeric(p["rmult_used"], errors="coerce")
            p["y_win"] = (p["rmult"] > 0).astype(int)
            p = p.dropna(subset=["symbol","entry_date","exit_date","rmult","stop_atr","target_atr"]).copy()
            # avoid duplicates
            key_cols = ["symbol","entry_date","exit_date","side","stop_r","tgt_r","rmult"]
            p = p.drop_duplicates(subset=key_cols, keep="last").copy()
            train_trades = pd.concat([train_trades, p[train_trades.columns]], ignore_index=True)

    # restrict train window
    train_start, train_end = build_training_window(run_date)
    train_trades = train_trades[(train_trades["entry_date"] >= train_start) & (train_trades["entry_date"] <= train_end) & (train_trades["exit_date"] <= train_end)].copy()
    if len(train_trades) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Not enough training rows after window/cutoff: {len(train_trades)} (min {MIN_TRAIN_ROWS})")
    log.info(f"[TRAINSET] rows={len(train_trades):,} symbols={train_trades['symbol'].nunique():,} window={train_start.date()}..{train_end.date()}")

    # Build OHLCV series_map for symbols needed:
    symbols_needed = sorted(set(train_trades["symbol"].unique().tolist()) | set(div["symbol"].unique().tolist()) | set([p.symbol for p in open_positions]))
    series_map, seg_map = load_ohlcv_subset_build_series(OHLCV_CSV, symbols_needed, atr_period_default=ATR_PERIOD_DEFAULT)

    # gap skip training trades
    t_gap = apply_gap_skip(train_trades, seg_map)
    forced = int((t_gap["gap_skip"] == 1).sum())
    train_trades = t_gap[t_gap["gap_skip"] == 0].drop(columns=["entry_seg","exit_seg","gap_skip"]).copy()
    log.info(f"[GAP] forced_skips={forced:,} kept_train={len(train_trades):,}")
    baseline_proxy_score = compute_baseline_proxy_score(train_trades)
    log.info(f"[BASELINE] proxy_expr_score={baseline_proxy_score:.6f}")

    # daily retrain models
    bucket_models, train_stats = train_bucket_models_daily(train_trades, series_map, run_tag=run_tag)
    rr_bucket, rr_global, rr_stats = train_rr_policy_daily(train_trades, series_map, run_tag=run_tag)

    # save models (daily snapshot)
    model_dir = os.path.join(OUT_DIR, "models", "daily", run_tag)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(bucket_models, os.path.join(model_dir, "bucket_models.pkl"))
    joblib.dump(rr_bucket, os.path.join(model_dir, "rr_bucket.pkl"))
    joblib.dump(rr_global, os.path.join(model_dir, "rr_global.pkl"))
    train_stats.to_csv(os.path.join(model_dir, "train_stats.csv"), index=False)
    if not rr_stats.empty:
        rr_stats.to_csv(os.path.join(model_dir, "rr_stats.csv"), index=False)
    json.dump(dict(run_date=str(run_date.date()), cutoff=str(cutoff.date()), train_start=str(train_start.date()), train_end=str(train_end.date()),
                   rocket_cfg=normalize_rocket_cfg(ROCKET_CFG), rocket_windows=ROCKET_WINDOWS, rocket_channels=ROCKET_CHANNELS),
              open(os.path.join(model_dir, "meta.json"), "w"), indent=2)

    # ---- Gate update: close positions up to cutoff, then update for run_date ----
    gate = load_or_init_gate()
    closed_rows = []
    still_open = []

    for p in open_positions:
        sym = p.symbol
        s = series_map.get(sym)
        if s is None or s.ohlc is None:
            still_open.append(p)
            continue
        last_mark = pd.Timestamp(p.last_mark_date).normalize() if p.last_mark_date else pd.Timestamp(p.entry_date).normalize()
        # evaluate exits from max(last_mark, entry_date) to cutoff
        entry_day = pd.Timestamp(p.entry_date).normalize()
        start_eval = max(entry_day, last_mark)
        if start_eval > cutoff:
            still_open.append(p)
            continue

        exit_day, reason, exit_price = simulate_exit_from_entry(
            s, entry_day=entry_day, side=p.side,
            entry_price=float(p.entry_price), stop_price=float(p.stop_price), target_price=float(p.target_price),
            max_hold_days=int(MAX_HOLD_DAYS), cutoff_day=cutoff
        )

        if exit_day is None:
            p.last_mark_date = str(cutoff.date())
            still_open.append(p)
            continue

        r_used = float(np.clip(rmult_from_exit(p.side, p.entry_price, p.stop_price, p.target_price, exit_price), -R_ABS_MAX_SANITY, R_ABS_MAX_SANITY))
        pnl = float(p.risk_amt * r_used)
        eq_before = float(equity)
        equity = float(equity + pnl)

        gate.add_exit(_pf_key(p.regime, p.side, p.side_mode, p.stop_r, p.tgt_r), exit_day, r_used=r_used, risk_amt=float(p.risk_amt))

        closed_rows.append(dict(
            pos_id=p.pos_id,
            symbol=p.symbol,
            side=p.side,
            regime=p.regime,
            side_mode=p.side_mode,
            stop_r=float(p.stop_r),
            tgt_r=float(p.tgt_r),
            entry_date=str(entry_day.date()),
            exit_date=str(exit_day.date()),
            exit_reason=str(reason),
            entry_price=float(p.entry_price),
            exit_price=float(exit_price),
            rmult_used=float(r_used),
            risk_amt=float(p.risk_amt),
            gate_mult=float(p.gate_mult),
            score=float(p.score),
            pnl=float(pnl),
            equity_before=float(eq_before),
            equity_after=float(equity),
            marked_to=str(cutoff.date()),
            model_run=str(run_date.date()),
        ))

    open_positions = still_open
    if closed_rows:
        append_ledger(closed_rows)
        log.info(f"[EXITS] closed={len(closed_rows):,} equity_now={equity:,.2f}")

    if WR_GATE_ENABLE:
        gate.update_states_for_cutoff(run_date)
    save_gate(gate)

    # ---- Build candidates for today ----
    cand = make_rr_candidates(div, run_date=run_date)
    if cand.empty:
        log.info("[CAND] no candidates for today; writing empty outputs.")
        save_open_positions(open_positions)
        st["equity"] = float(equity)
        st["pos_counter"] = int(pos_counter)
        st["last_run_date"] = str(run_date.date())
        save_state(st)

        pd.DataFrame().to_csv(os.path.join(run_dir, "candidates_scored.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(run_dir, "selected_trades.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(run_dir, "orders_to_place.csv"), index=False)
        return

    # price + stops + targets
    entry_prices = []
    stop_prices = []
    target_prices = []
    atr_vals = []
    ok_px = []
    for r in cand.itertuples(index=False):
        s = series_map.get(r.symbol)
        px = get_ohlc_at(s, r.entry_date) if s is not None else None
        if px is None:
            entry_prices.append(np.nan); stop_prices.append(np.nan); target_prices.append(np.nan); atr_vals.append(np.nan); ok_px.append(0)
            continue
        o,h,l,c,atr = px
        atr_use = atr if np.isfinite(atr) and atr > 0 else np.nan
        if not np.isfinite(atr_use) or atr_use <= 0:
            entry_prices.append(np.nan); stop_prices.append(np.nan); target_prices.append(np.nan); atr_vals.append(np.nan); ok_px.append(0)
            continue
        entry = float(o)
        stop_dist = float(r.stop_atr) * float(atr_use)
        tgt_dist  = float(r.target_atr) * float(atr_use)
        if r.side == "long":
            stop_p = entry - stop_dist
            tgt_p  = entry + tgt_dist
        else:
            stop_p = entry + stop_dist
            tgt_p  = entry - tgt_dist

        entry_prices.append(entry)
        stop_prices.append(stop_p)
        target_prices.append(tgt_p)
        atr_vals.append(atr_use)
        ok_px.append(1)

    cand["entry_price"] = np.array(entry_prices, dtype=float)
    cand["stop_price"] = np.array(stop_prices, dtype=float)
    cand["target_price"] = np.array(target_prices, dtype=float)
    cand["atr_value"] = np.array(atr_vals, dtype=float)
    cand["px_ok"] = np.array(ok_px, dtype=int)
    cand = cand[cand["px_ok"] == 1].copy()
    if cand.empty:
        log.info("[CAND] no candidates with price/ATR available.")
        save_open_positions(open_positions)
        st["equity"] = float(equity)
        st["pos_counter"] = int(pos_counter)
        st["last_run_date"] = str(run_date.date())
        save_state(st)
        pd.DataFrame().to_csv(os.path.join(run_dir, "candidates_scored.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(run_dir, "selected_trades.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(run_dir, "orders_to_place.csv"), index=False)
        return

    # features for candidates
    F_rocket_c, ok_all_c = compute_rocket_features_for_df(cand, series_map, ROCKET_CFG, ROCKET_RANDOM_SEED)
    F_static_c, static_cols_c = build_static_features(cand)
    Xc = np.asarray(F_rocket_c, dtype=TRAIN_DTYPE)
    if F_static_c.shape[1] > 0:
        Xc = np.hstack([Xc, np.asarray(F_static_c, dtype=TRAIN_DTYPE)]).astype(TRAIN_DTYPE, copy=False)

    regimes = cand["regime"].astype(str).values
    sides = cand["side"].astype(str).values
    expr, valid = score_expr(bucket_models, Xc, regimes, sides)
    cand["expr_score_raw"] = expr.astype(np.float32)
    cand["expr_valid"] = valid.astype(int)

    missing_cnt = int((cand["expr_valid"].values == 0).sum())
    fallback_mode = str(MISSING_BUCKET_FALLBACK).strip().lower()
    if fallback_mode == "drop":
        cand = cand[cand["expr_valid"].values == 1].copy()
        cand["expr_score"] = cand["expr_score_raw"].astype(np.float32)
    elif fallback_mode == "zero":
        cand["expr_score"] = np.where(cand["expr_valid"].values == 1, cand["expr_score_raw"].values, 0.0).astype(np.float32)
    else:
        cand["expr_score"] = np.where(
            cand["expr_valid"].values == 1,
            cand["expr_score_raw"].values,
            np.float32(baseline_proxy_score),
        ).astype(np.float32)
        fallback_mode = "baseline"
    log.info(f"[CAND] missing_bucket_scores={missing_cnt:,} fallback_mode={fallback_mode}")

    # RR choice score
    # Baseline score (notebook parity): use the candidate-level selection score as baseline fallback.
    cand["baseline_score"] = cand["expr_score"].astype(np.float32)
    rr_pred = rr_policy_predict(cand, rr_bucket, rr_global, Xc)
    cand["rr_pred"] = rr_pred.astype(np.float32)
    base = cand["baseline_score"].values.astype(np.float32)
    cand["rr_choice_score"] = np.where(
        np.isfinite(rr_pred),
        rr_pred.astype(np.float32) + float(RR_POLICY_BASELINE_WEIGHT) * base,
        base,
    ).astype(np.float32)

    # one RR per signal: group by (symbol, entry_date, side) and pick best rr_choice_score
    if ENFORCE_ONE_RR_PER_SIGNAL:
        cand = cand.sort_values(["rr_choice_score","baseline_score","expr_score"], ascending=False).groupby(["symbol","entry_date","side"], sort=False).head(1).copy()

    # gate-adjusted selection penalty (ranking only)
    if WR_GATE_ENABLE and WR_GATE_SCORE_PENALTY > 0:
        penalty = float(WR_GATE_SCORE_PENALTY)
        mults = []
        for r in cand.itertuples(index=False):
            mults.append(gate.get_mult(_pf_key(r.regime, r.side, r.side_mode, r.stop_r, r.tgt_r)))
        mults = np.array(mults, dtype=np.float32)
        cand["gate_mult_now"] = mults
        cand["expr_score_adj"] = cand["expr_score"].astype(np.float32) - np.where(mults < 0.999, penalty, 0.0).astype(np.float32)
    else:
        cand["gate_mult_now"] = 1.0
        cand["expr_score_adj"] = cand["expr_score"].astype(np.float32)

    cap, open_risk_pct = _daily_capacity(equity, open_positions)
    if cap <= 0:
        log.info(f"[SELECT] cap=0 open_positions={len(open_positions)} open_risk_pct={open_risk_pct:.3f}")
        cand.to_csv(os.path.join(run_dir, "candidates_scored.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(run_dir, "selected_trades.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(run_dir, "orders_to_place.csv"), index=False)
        save_open_positions(open_positions)
        st["equity"] = float(equity)
        st["pos_counter"] = int(pos_counter)
        st["last_run_date"] = str(run_date.date())
        save_state(st)
        return

    if SELECTION_MODE == "expr_threshold":
        cand_sel = cand[cand["expr_score"] >= float(EXPR_THRESHOLD)].copy()
        cand_sel = cand_sel.sort_values("expr_score_adj", ascending=False).head(cap).copy()
    else:
        cand_sel = cand.sort_values("expr_score_adj", ascending=False).head(cap).copy()

    # open new positions with gate sizing
    orders = []
    for r in cand_sel.itertuples(index=False):
        gm = float(gate.get_mult(_pf_key(r.regime, r.side, r.side_mode, r.stop_r, r.tgt_r))) if WR_GATE_ENABLE else 1.0
        base_risk_amt = float(RISK_PCT_PER_TRADE * equity)
        risk_amt = float(base_risk_amt * gm)
        q = qty_from_risk(risk_amt, r.entry_price, r.stop_price, r.side)

        if q <= 0:
            continue

        pos_counter += 1
        open_positions.append(OpenPos(
            pos_id=int(pos_counter),
            entry_date=str(pd.Timestamp(r.entry_date).normalize().date()),
            symbol=str(r.symbol),
            side=str(r.side),
            regime=str(r.regime),
            side_mode=str(r.side_mode),
            stop_r=float(r.stop_r),
            tgt_r=float(r.tgt_r),
            entry_price=float(r.entry_price),
            stop_price=float(r.stop_price),
            target_price=float(r.target_price),
            risk_amt=float(risk_amt),
            gate_mult=float(gm),
            score=float(r.expr_score),
            last_mark_date=str(cutoff.date()),
        ))

        orders.append(dict(
            pos_id=int(pos_counter),
            symbol=str(r.symbol),
            side=str(r.side),
            entry_date=str(pd.Timestamp(r.entry_date).normalize().date()),
            entry_price=float(r.entry_price),
            qty=int(q),
            stop_price=float(r.stop_price),
            target_price=float(r.target_price),
            stop_r=float(r.stop_r),
            tgt_r=float(r.tgt_r),
            risk_amt=float(risk_amt),
            gate_mult=float(gm),
            expr_score=float(r.expr_score),
            expr_score_adj=float(r.expr_score_adj),
            rr_pred=float(r.rr_pred) if np.isfinite(r.rr_pred) else np.nan,
            model_run=str(run_date.date()),
        ))

    # write outputs
    cand.to_csv(os.path.join(run_dir, "candidates_scored.csv"), index=False)
    cand_sel.to_csv(os.path.join(run_dir, "selected_trades.csv"), index=False)
    pd.DataFrame(orders).to_csv(os.path.join(run_dir, "orders_to_place.csv"), index=False)

    # persist state
    save_open_positions(open_positions)
    st["equity"] = float(equity)
    st["pos_counter"] = int(pos_counter)
    st["last_run_date"] = str(run_date.date())
    save_state(st)

    # gate daily snapshot
    pd.DataFrame(gate.daily_rows).tail(200).to_csv(os.path.join(OUT_DIR, "state", "wr_gate_daily_tail.csv"), index=False)
    if gate.events:
        pd.DataFrame([e.__dict__ for e in gate.events]).tail(500).to_csv(os.path.join(OUT_DIR, "state", "wr_gate_events_tail.csv"), index=False)

    log.info(f"[DONE] orders={len(orders):,} open_positions={len(open_positions):,} equity={equity:,.2f} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
