#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build (or rebuild) a single OHLCV master file from your per-day OHLCV store.

You currently have:
  data/ohlcv_daily/ohlcv_YYYY-MM-DD.csv.gz
(one file per day; produced by backfill_ohlcv_1y.py and update_ohlcv_1y.py).

This script:
- Scans OHLCV_DAILY_DIR for daily files
- Keeps only the last KEEP_DAYS (default ~7 years)
- Writes a gzipped master CSV (date-order):
    data/ohlcv_master.csv.gz
- Also writes a small meta JSON next to it.

No argparse. Configure via env vars.

NOTE:
- This is a *streaming* build (does not load everything into RAM).
- It does not attempt to sort within a day or re-dedupe; the daily files are assumed
  to have at most 1 row per (date,symbol). Your downstream pipeline already sorts
  per-symbol and applies dedupe rules where needed.
"""

import os
import re
import csv
import json
import gzip
from datetime import datetime, timedelta

OHLCV_DAILY_DIR = os.environ.get("OHLCV_DAILY_DIR", "./data/ohlcv_daily")
OUT_MASTER_PATH = os.environ.get("OHLCV_MASTER_PATH", "./data/ohlcv_master.csv.gz")
KEEP_DAYS = int(os.environ.get("KEEP_DAYS", str(7 * 370)))  # ~7y with buffer

# If 1, overwrite OUT_MASTER_PATH. If 0, write to a temp and then atomically replace.
ATOMIC_REPLACE = os.environ.get("ATOMIC_REPLACE", "1").strip().lower() not in {"0", "false", "no"}

_RE = re.compile(r"^ohlcv_(\d{4}-\d{2}-\d{2})\.csv\.gz$")


def _list_daily_files(daily_dir: str):
    if not os.path.isdir(daily_dir):
        raise RuntimeError(f"OHLCV_DAILY_DIR not found: {daily_dir}")
    out = []
    for fn in os.listdir(daily_dir):
        m = _RE.match(fn)
        if not m:
            continue
        ds = m.group(1)
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        out.append((d, os.path.join(daily_dir, fn)))
    out.sort(key=lambda x: x[0])
    return out


def _iter_rows_from_daily(fp: str):
    with gzip.open(fp, "rt", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return
        # Expect: date,symbol,open,high,low,close,volume
        for row in r:
            if not row:
                continue
            yield row


def main():
    daily = _list_daily_files(OHLCV_DAILY_DIR)
    if not daily:
        raise RuntimeError(f"No daily files found in {OHLCV_DAILY_DIR}")

    last_day = daily[-1][0]
    cutoff = last_day - timedelta(days=KEEP_DAYS)
    keep = [(d, fp) for (d, fp) in daily if d >= cutoff]
    if not keep:
        raise RuntimeError("After cutoff, no files remain. Increase KEEP_DAYS.")

    out_dir = os.path.dirname(os.path.abspath(OUT_MASTER_PATH))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = OUT_MASTER_PATH + ".tmp" if ATOMIC_REPLACE else OUT_MASTER_PATH

    n_rows = 0
    n_files = 0

    with gzip.open(tmp_path, "wt", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])

        for d, fp in keep:
            n_files += 1
            for row in _iter_rows_from_daily(fp):
                # Basic sanity: if row has fewer columns, skip.
                if len(row) < 7:
                    continue
                w.writerow(row[:7])
                n_rows += 1

            if n_files % 60 == 0:
                print(f"[BUILD] files={n_files}/{len(keep)} rows={n_rows:,} last={d.isoformat()}")

    if ATOMIC_REPLACE:
        os.replace(tmp_path, OUT_MASTER_PATH)

    meta = {
        "daily_dir": os.path.abspath(OHLCV_DAILY_DIR),
        "master_path": os.path.abspath(OUT_MASTER_PATH),
        "keep_days": int(KEEP_DAYS),
        "start_day": keep[0][0].isoformat(),
        "end_day": keep[-1][0].isoformat(),
        "files_included": int(len(keep)),
        "rows_written": int(n_rows),
        "generated_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }

    meta_path = os.path.splitext(os.path.splitext(OUT_MASTER_PATH)[0])[0] + "_meta.json"  # ohlcv_master_meta.json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] master={OUT_MASTER_PATH} files={len(keep)} rows={n_rows:,} range={keep[0][0]}..{keep[-1][0]}")
    print(f"[META] {meta_path}")


if __name__ == "__main__":
    main()
