#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-time OHLCV backfill (daily) for last ~N years (default ~7y).

- Reads Kite access_token from KITE_TOKEN_PATH JSON (generated in same workflow).
- Builds NSE EQ universe from kite.instruments("NSE").
- Finds latest completed trading day using a reference symbol's daily candles.
- For each symbol, fetches DAILY candles for [start_date .. end_date] in chunks.
  (Day interval supports up to ~2000 days per request; script chunks for safety.)  # see your note
- Writes one gzipped CSV per date:
    data/ohlcv_daily/ohlcv_YYYY-MM-DD.csv.gz
- Prunes files older than RETENTION_DAYS.

No argparse: configure via env vars.
"""

import os
import re
import csv
import json
import gzip
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple

from kiteconnect import KiteConnect

TOKEN_PATH = os.environ.get("KITE_TOKEN_PATH", "./secrets/kite_access_token.json")
OUT_DIR = os.environ.get("OHLCV_OUT_DIR", "./data/ohlcv_daily")

# Backfill window (calendar days)
BACKFILL_DAYS = int(os.environ.get("BACKFILL_DAYS", str(7 * 366)))      # ~7 years
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", str(7 * 366)))    # keep ~7 years

# Kite day-candle request span safety: keep < 2000 days to avoid failures
MAX_DAYS_PER_REQUEST = int(os.environ.get("MAX_DAYS_PER_REQUEST", "1900"))

# Throttle to avoid 429s
HIST_RPS = float(os.environ.get("HIST_RPS", "2.0"))

EXCLUDE_ETF = os.environ.get("EXCLUDE_ETF", "1").strip().lower() not in {"0", "false", "no"}

MAX_RETRIES = 5
BACKOFF = 1.7


def _is_etf_symbol(sym: str) -> bool:
    s = str(sym).upper()
    return ("ETF" in s) or ("BEES" in s)


def _read_token(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mkdirp(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _throttle(last_ts: float, rps: float) -> float:
    if rps <= 0:
        return time.time()
    min_dt = 1.0 / rps
    now = time.time()
    dt = now - last_ts
    if dt < min_dt:
        time.sleep(min_dt - dt)
    return time.time()


def _retry(fn, *args, **kwargs):
    err = None
    for k in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = e
            time.sleep((BACKOFF ** k) + 0.1 * k)
    raise err


def _parse_ohlcv_fname(fn: str):
    m = re.match(r"^ohlcv_(\d{4}-\d{2}-\d{2})\.csv\.gz$", fn)
    return m.group(1) if m else None


def _prune_old_files(out_dir: str, keep_days: int):
    _mkdirp(out_dir)
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


def _find_reference_token(eq_rows):
    pref = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    m = {r["tradingsymbol"]: r for r in eq_rows}
    for s in pref:
        if s in m:
            return m[s]["instrument_token"], s
    return eq_rows[0]["instrument_token"], eq_rows[0]["tradingsymbol"]


def _latest_completed_trading_day(kite: KiteConnect, ref_token: int) -> date:
    now = datetime.utcnow()
    frm = (now - timedelta(days=25)).replace(hour=0, minute=0, second=0, microsecond=0)
    to = now.replace(hour=23, minute=59, second=59, microsecond=0)
    candles = _retry(kite.historical_data, ref_token, frm, to, "day", False, False)
    if not candles:
        raise RuntimeError("No candles returned for reference token; cannot infer latest trading day.")
    return candles[-1]["date"].date()


def _write_daily_files(out_dir: str, by_date_rows: dict):
    _mkdirp(out_dir)
    dates = sorted(by_date_rows.keys())
    for d in dates:
        out_file = os.path.join(out_dir, f"ohlcv_{d}.csv.gz")
        rows = by_date_rows[d]
        # overwrite to ensure idempotent backfill
        with gzip.open(out_file, "wt", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
            for r in rows:
                w.writerow([r["date"], r["symbol"], r["open"], r["high"], r["low"], r["close"], r["volume"]])


def _iter_windows(start_day: date, end_day: date, max_days: int):
    """Generate inclusive [a..b] windows with length <= max_days."""
    cur = start_day
    while cur <= end_day:
        b = min(end_day, cur + timedelta(days=max_days - 1))
        yield cur, b
        cur = b + timedelta(days=1)


def main():
    tok = _read_token(TOKEN_PATH)
    api_key = tok.get("api_key") or os.environ.get("KITE_API_KEY")
    access_token = tok.get("access_token")
    if not api_key or not access_token:
        raise RuntimeError("TOKEN_PATH must contain api_key and access_token.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    _mkdirp(OUT_DIR)

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
            itok = r.get("instrument_token", None)
            if itok is None:
                continue
            eq.append({"tradingsymbol": sym, "instrument_token": int(itok)})
        except Exception:
            continue

    if not eq:
        raise RuntimeError("No NSE EQ instruments after filtering.")

    ref_token, ref_sym = _find_reference_token(eq)
    end_day = _latest_completed_trading_day(kite, ref_token)
    start_day = end_day - timedelta(days=BACKFILL_DAYS)

    windows = list(_iter_windows(start_day, end_day, MAX_DAYS_PER_REQUEST))
    print(
        f"[BACKFILL] universe={len(eq)} ref={ref_sym} start={start_day} end={end_day} "
        f"windows={len(windows)} max_days_per_req={MAX_DAYS_PER_REQUEST} rps={HIST_RPS}"
    )

    by_date_rows: Dict[str, List[dict]] = {}
    last_ts = 0.0
    skipped_symbols = 0
    total_requests = 0

    for i, r in enumerate(eq, 1):
        sym = r["tradingsymbol"]
        itok = r["instrument_token"]

        sym_had_any = False
        sym_failed = False

        for (a, b) in windows:
            frm = datetime(a.year, a.month, a.day, 0, 0, 0)
            to = datetime(b.year, b.month, b.day, 23, 59, 59)

            last_ts = _throttle(last_ts, HIST_RPS)
            total_requests += 1

            try:
                candles = _retry(kite.historical_data, itok, frm, to, "day", False, False)
                if not candles:
                    continue
                sym_had_any = True
                for c in candles:
                    ds = c["date"].date().isoformat()
                    by_date_rows.setdefault(ds, []).append({
                        "date": ds,
                        "symbol": sym,
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": int(c.get("volume", 0) or 0),
                    })
            except Exception:
                sym_failed = True

        if (not sym_had_any) or sym_failed:
            skipped_symbols += 1

        if i % 200 == 0:
            print(
                f"[PROGRESS] {i}/{len(eq)} dates={len(by_date_rows)} "
                f"skipped_syms={skipped_symbols} requests={total_requests}"
            )

    if not by_date_rows:
        raise RuntimeError("Backfill collected 0 rows.")

    _write_daily_files(OUT_DIR, by_date_rows)
    print(f"[WRITE] wrote_files={len(by_date_rows)} date_range=[{min(by_date_rows)}..{max(by_date_rows)}]")

    meta = {
        "mode": "backfill",
        "start_day": start_day.isoformat(),
        "end_day": end_day.isoformat(),
        "universe_count": len(eq),
        "dates_written": len(by_date_rows),
        "skipped_symbols": skipped_symbols,
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "hist_rps": HIST_RPS,
        "exclude_etf": EXCLUDE_ETF,
        "backfill_days": BACKFILL_DAYS,
        "retention_days": RETENTION_DAYS,
        "max_days_per_request": MAX_DAYS_PER_REQUEST,
        "windows": len(windows),
        "requests_total": total_requests,
    }
    with open(os.path.join(OUT_DIR, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    kept, removed = _prune_old_files(OUT_DIR, RETENTION_DAYS)
    print(f"[PRUNE] kept={kept} removed={removed}")


if __name__ == "__main__":
    main()
