#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step B: Update OHLCV store in GitHub repo (keeps last ~N years; default ~7y) — CONCURRENT FETCH.

- Reads Kite access_token from KITE_TOKEN_PATH JSON (generated earlier in workflow).
- Builds NSE EQ universe from kite.instruments("NSE").
- Finds the latest completed trading day by checking a reference symbol's daily candle.
- Fetches DAILY candle (O/H/L/C/V) for that day for each symbol using historical_data().
- Writes: data/ohlcv_daily/ohlcv_YYYY-MM-DD.csv.gz
- Prunes files older than RETENTION_DAYS.

Concurrency:
- Uses ThreadPoolExecutor to overlap network latency.
- Uses a GLOBAL rate limiter so total request rate stays near HIST_RPS (safe).
- You can tune OHLCV_WORKERS, HIST_RPS via env vars.

No argparse. Configure via env vars.
"""

import os
import re
import csv
import json
import gzip
import time
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from kiteconnect import KiteConnect


# -----------------------------
# ENV / CONFIG
# -----------------------------
TOKEN_PATH = os.environ.get("KITE_TOKEN_PATH", "./secrets/kite_access_token.json")
OUT_DIR = os.environ.get("OHLCV_OUT_DIR", "./data/ohlcv_daily")

# Keep ~7 years by default
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", str(7 * 366)))

# Total allowed requests/sec across ALL threads (global)
HIST_RPS = float(os.environ.get("HIST_RPS", "2.5"))

# Number of concurrent worker threads (overlaps latency; does NOT bypass RPS limit)
OHLCV_WORKERS = int(os.environ.get("OHLCV_WORKERS", "24"))

EXCLUDE_ETF = os.environ.get("EXCLUDE_ETF", "1").strip().lower() not in {"0", "false", "no"}

MAX_RETRIES = 4
BACKOFF_BASE = 1.6  # exponential

# Progress prints
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "200"))


# -----------------------------
# Helpers
# -----------------------------
def _is_etf_symbol(sym: str) -> bool:
    s = str(sym).upper()
    return ("ETF" in s) or ("BEES" in s)


def _read_token(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_ohlcv_fname(fn: str):
    m = re.match(r"^ohlcv_(\d{4}-\d{2}-\d{2})\.csv\.gz$", fn)
    if not m:
        return None
    return m.group(1)


def _prune_old_files(out_dir: str, keep_days: int):
    os.makedirs(out_dir, exist_ok=True)
    today = datetime.utcnow().date()
    cutoff = today - timedelta(days=int(keep_days))

    removed = 0
    kept = 0
    for fn in os.listdir(out_dir):
        ds = _parse_ohlcv_fname(fn)
        if not ds:
            continue
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        fp = os.path.join(out_dir, fn)
        if d < cutoff:
            try:
                os.remove(fp)
                removed += 1
            except Exception:
                pass
        else:
            kept += 1
    return kept, removed


def _retry_call(fn, *args, **kwargs):
    last_err = None
    for k in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            sleep_s = (BACKOFF_BASE ** k) + (0.05 * k)
            time.sleep(sleep_s)
    raise last_err


def _find_reference_token(inst_rows):
    pref = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    by_sym = {r["tradingsymbol"]: r for r in inst_rows}
    for s in pref:
        if s in by_sym:
            return by_sym[s]["instrument_token"], s
    r0 = inst_rows[0]
    return r0["instrument_token"], r0["tradingsymbol"]


def _latest_completed_trading_day(kite: KiteConnect, ref_token: int) -> datetime.date:
    now = datetime.utcnow()
    frm = (now - timedelta(days=20)).replace(hour=0, minute=0, second=0, microsecond=0)
    to = now.replace(hour=23, minute=59, second=59, microsecond=0)

    candles = _retry_call(kite.historical_data, ref_token, frm, to, "day", False, False)
    if not candles:
        raise RuntimeError("Could not determine latest trading day (no candles returned).")
    return candles[-1]["date"].date()


def _write_gz_csv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
        for r in rows:
            w.writerow([r["date"], r["symbol"], r["open"], r["high"], r["low"], r["close"], r["volume"]])


# -----------------------------
# Global rate limiter (thread-safe)
# -----------------------------
class GlobalRateLimiter:
    """
    Leaky-bucket style limiter:
    - Ensures average request spacing >= 1/rps across all threads.
    """
    def __init__(self, rps: float):
        self.rps = float(max(0.0, rps))
        self.min_dt = 0.0 if self.rps <= 0 else 1.0 / self.rps
        self._lock = threading.Lock()
        self._next_ts = 0.0

    def acquire(self):
        if self.min_dt <= 0:
            return
        while True:
            with self._lock:
                now = time.time()
                if now >= self._next_ts:
                    self._next_ts = now + self.min_dt
                    return
                sleep_for = self._next_ts - now
            if sleep_for > 0:
                time.sleep(sleep_for)


# -----------------------------
# Per-thread Kite client (avoid shared session issues)
# -----------------------------
_tls = threading.local()

def _get_kite_client(api_key: str, access_token: str) -> KiteConnect:
    kc = getattr(_tls, "kite", None)
    if kc is None:
        kc = KiteConnect(api_key=api_key)
        kc.set_access_token(access_token)
        _tls.kite = kc
    return kc


def _fetch_one_symbol_daily(
    api_key: str,
    access_token: str,
    limiter: GlobalRateLimiter,
    symbol: str,
    itok: int,
    frm: datetime,
    to: datetime,
):
    """
    Returns dict row or None if skipped.
    """
    limiter.acquire()
    kite = _get_kite_client(api_key, access_token)

    candles = _retry_call(kite.historical_data, itok, frm, to, "day", False, False)
    if not candles:
        return None

    c = candles[-1]
    dt = c["date"].date().isoformat()
    return {
        "date": dt,
        "symbol": symbol,
        "open": float(c["open"]),
        "high": float(c["high"]),
        "low": float(c["low"]),
        "close": float(c["close"]),
        "volume": int(c.get("volume", 0) or 0),
    }


def main():
    tok = _read_token(TOKEN_PATH)
    api_key = tok.get("api_key") or os.environ.get("KITE_API_KEY")
    access_token = tok.get("access_token")
    if not api_key or not access_token:
        raise RuntimeError("TOKEN_PATH JSON must contain api_key and access_token (or set KITE_API_KEY).")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Use a single client for instruments + reference-day discovery
    kite0 = KiteConnect(api_key=api_key)
    kite0.set_access_token(access_token)

    inst = kite0.instruments("NSE")
    if not inst:
        raise RuntimeError("kite.instruments('NSE') returned empty list.")

    eq = []
    for r in inst:
        try:
            if str(r.get("exchange", "")).upper() != "NSE":
                continue
            if str(r.get("instrument_type", "")).upper() != "EQ":
                continue
            sym = str(r.get("tradingsymbol", "")).upper().strip()
            if not sym:
                continue
            if EXCLUDE_ETF and _is_etf_symbol(sym):
                continue
            tok_ = r.get("instrument_token", None)
            if tok_ is None:
                continue
            eq.append({"tradingsymbol": sym, "instrument_token": int(tok_)})
        except Exception:
            continue

    if not eq:
        raise RuntimeError("No NSE EQ instruments found after filtering.")

    # Stable ordering
    eq.sort(key=lambda x: x["tradingsymbol"])

    ref_token, ref_sym = _find_reference_token(eq)
    target_day = _latest_completed_trading_day(kite0, ref_token)

    out_file = os.path.join(OUT_DIR, f"ohlcv_{target_day.strftime('%Y-%m-%d')}.csv.gz")
    if os.path.exists(out_file):
        print(f"[OHLCV] Already exists: {out_file} (skipping fetch)")
        kept, removed = _prune_old_files(OUT_DIR, RETENTION_DAYS)
        print(f"[PRUNE] kept={kept} removed={removed}")
        return

    frm = datetime(target_day.year, target_day.month, target_day.day, 0, 0, 0)
    to = datetime(target_day.year, target_day.month, target_day.day, 23, 59, 59)

    limiter = GlobalRateLimiter(HIST_RPS)

    total = len(eq)
    print(
        f"[UNIVERSE] NSE EQ count={total} ref={ref_sym} target_day={target_day.isoformat()} "
        f"retention_days={RETENTION_DAYS} hist_rps={HIST_RPS} workers={OHLCV_WORKERS}"
    )

    rows = []
    skipped = 0
    done = 0

    t0 = time.time()

    # Submit all tasks
    with ThreadPoolExecutor(max_workers=max(1, OHLCV_WORKERS)) as ex:
        futs = []
        for r in eq:
            futs.append(
                ex.submit(
                    _fetch_one_symbol_daily,
                    api_key,
                    access_token,
                    limiter,
                    r["tradingsymbol"],
                    r["instrument_token"],
                    frm,
                    to,
                )
            )

        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
                if res is None:
                    skipped += 1
                else:
                    rows.append(res)
            except Exception:
                skipped += 1

            if (done % PROGRESS_EVERY) == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / max(1e-9, elapsed)
                print(f"[PROGRESS] {done}/{total} rows={len(rows)} skipped={skipped} elapsed={elapsed:.1f}s eff_rps={rate:.2f}")

    if not rows:
        raise RuntimeError("No OHLCV rows collected (all skipped).")

    # Deterministic output sort
    rows.sort(key=lambda x: x["symbol"])

    _write_gz_csv(out_file, rows)
    print(f"[WRITE] {out_file} rows={len(rows)} skipped={skipped}")

    meta_path = os.path.join(OUT_DIR, "_meta.json")
    meta = {
        "target_day": target_day.isoformat(),
        "universe_count": total,
        "rows_written": len(rows),
        "skipped": skipped,
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "hist_rps": HIST_RPS,
        "workers": OHLCV_WORKERS,
        "exclude_etf": EXCLUDE_ETF,
        "retention_days": RETENTION_DAYS,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    kept, removed = _prune_old_files(OUT_DIR, RETENTION_DAYS)
    print(f"[PRUNE] kept={kept} removed={removed}")


if __name__ == "__main__":
    main()
