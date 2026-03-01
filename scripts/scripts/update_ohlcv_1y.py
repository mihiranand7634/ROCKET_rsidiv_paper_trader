#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step B: Update 1-year OHLCV store in GitHub repo.

- Reads Kite access_token from KITE_TOKEN_PATH JSON (generated earlier in workflow).
- Builds NSE EQ universe from kite.instruments("NSE").
- Finds the latest completed trading day by checking a reference symbol's daily candle.
- Fetches DAILY candle (O/H/L/C/V) for that day for each symbol using historical_data().
- Writes: data/ohlcv_daily/ohlcv_YYYY-MM-DD.csv.gz
- Prunes files older than RETENTION_DAYS.
- No argparse. Configure via env vars.
"""

import os
import re
import csv
import json
import gzip
import time
import math
from datetime import datetime, timedelta

from kiteconnect import KiteConnect


TOKEN_PATH = os.environ.get("KITE_TOKEN_PATH", "./secrets/kite_access_token.json")
OUT_DIR = os.environ.get("OHLCV_OUT_DIR", "./data/ohlcv_daily")
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "370"))
HIST_RPS = float(os.environ.get("HIST_RPS", "3.5"))
EXCLUDE_ETF = os.environ.get("EXCLUDE_ETF", "1").strip() not in {"0", "false", "False"}

# Safety buffer for retries
MAX_RETRIES = 4
BACKOFF_BASE = 1.6  # exponential


def _is_etf_symbol(sym: str) -> bool:
    s = str(sym).upper()
    return ("ETF" in s) or ("BEES" in s)


def _read_token(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_ohlcv_fname(fn: str):
    # ohlcv_YYYY-MM-DD.csv.gz
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


def _throttle(last_ts: float, rps: float) -> float:
    if rps <= 0:
        return time.time()
    min_dt = 1.0 / rps
    now = time.time()
    dt = now - last_ts
    if dt < min_dt:
        time.sleep(min_dt - dt)
    return time.time()


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
    # Prefer a stable, liquid equity
    pref = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    by_sym = {r["tradingsymbol"]: r for r in inst_rows}
    for s in pref:
        if s in by_sym:
            return by_sym[s]["instrument_token"], s
    # fallback to first
    r0 = inst_rows[0]
    return r0["instrument_token"], r0["tradingsymbol"]


def _latest_completed_trading_day(kite: KiteConnect, ref_token: int) -> datetime.date:
    # Pull last ~14 calendar days of daily candles, take the last candle date.
    # Historical candle API provides daily candles (Timestamp, O/H/L/C/V). :contentReference[oaicite:5]{index=5}
    now = datetime.utcnow()
    frm = (now - timedelta(days=20)).replace(hour=0, minute=0, second=0, microsecond=0)
    to = now.replace(hour=23, minute=59, second=59, microsecond=0)

    candles = _retry_call(kite.historical_data, ref_token, frm, to, "day", False, False)
    if not candles:
        raise RuntimeError("Could not determine latest trading day (no candles returned).")
    # candles have 'date' as datetime
    last_dt = candles[-1]["date"]
    return last_dt.date()


def _write_gz_csv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
        for r in rows:
            w.writerow([r["date"], r["symbol"], r["open"], r["high"], r["low"], r["close"], r["volume"]])


def main():
    tok = _read_token(TOKEN_PATH)
    api_key = tok.get("api_key") or os.environ.get("KITE_API_KEY")
    access_token = tok.get("access_token")
    if not api_key or not access_token:
        raise RuntimeError("TOKEN_PATH JSON must contain api_key and access_token.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Build EQ universe from instruments("NSE") list.
    # instruments() contains fields like exchange, tradingsymbol, instrument_token, instrument_type, segment. :contentReference[oaicite:6]{index=6}
    inst = kite.instruments("NSE")
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

    ref_token, ref_sym = _find_reference_token(eq)
    target_day = _latest_completed_trading_day(kite, ref_token)

    out_file = os.path.join(OUT_DIR, f"ohlcv_{target_day.strftime('%Y-%m-%d')}.csv.gz")
    if os.path.exists(out_file):
        print(f"[OHLCV] Already exists: {out_file} (skipping fetch)")
        kept, removed = _prune_old_files(OUT_DIR, RETENTION_DAYS)
        print(f"[PRUNE] kept={kept} removed={removed}")
        return

    # Fetch daily candle for target_day for each instrument token
    frm = datetime(target_day.year, target_day.month, target_day.day, 0, 0, 0)
    to = datetime(target_day.year, target_day.month, target_day.day, 23, 59, 59)

    rows = []
    skipped = 0
    last_ts = 0.0

    print(f"[UNIVERSE] NSE EQ count={len(eq)} ref={ref_sym} target_day={target_day.isoformat()}")
    for i, r in enumerate(eq, 1):
        last_ts = _throttle(last_ts, HIST_RPS)

        sym = r["tradingsymbol"]
        itok = r["instrument_token"]

        try:
            candles = _retry_call(kite.historical_data, itok, frm, to, "day", False, False)
            if not candles:
                skipped += 1
                continue
            c = candles[-1]
            dt = c["date"].date().isoformat()
            rows.append({
                "date": dt,
                "symbol": sym,
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": int(c.get("volume", 0) or 0),
            })
        except Exception:
            skipped += 1

        if i % 200 == 0:
            print(f"[PROGRESS] {i}/{len(eq)} rows_written={len(rows)} skipped={skipped}")

    if not rows:
        raise RuntimeError("No OHLCV rows collected (all skipped).")

    _write_gz_csv(out_file, rows)
    print(f"[WRITE] {out_file} rows={len(rows)} skipped={skipped}")

    # Write meta (committed to repo for traceability)
    meta_path = os.path.join(OUT_DIR, "_meta.json")
    meta = {
        "target_day": target_day.isoformat(),
        "universe_count": len(eq),
        "rows_written": len(rows),
        "skipped": skipped,
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "hist_rps": HIST_RPS,
        "exclude_etf": EXCLUDE_ETF,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    kept, removed = _prune_old_files(OUT_DIR, RETENTION_DAYS)
    print(f"[PRUNE] kept={kept} removed={removed}")


if __name__ == "__main__":
    main()
