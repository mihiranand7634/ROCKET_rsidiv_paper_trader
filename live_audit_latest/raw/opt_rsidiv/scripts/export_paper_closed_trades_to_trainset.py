#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

LEDGER_PATH = "paper_trader_out/state/trade_ledger.csv"
OUT_PATH = "data/train_trades_closed.csv"

def norm_day(x):
    return pd.to_datetime(x, errors="coerce").dt.normalize()

def main():
    if not os.path.exists(LEDGER_PATH):
        raise FileNotFoundError(f"Missing {LEDGER_PATH}. Run paper trader first.")

    led = pd.read_csv(LEDGER_PATH)
    led.columns = [c.strip() for c in led.columns]

    # required columns from your paper ledger
    req = ["pos_id","symbol","entry_date","exit_date","regime","side","side_mode","stop_r","tgt_r","rmult_used"]
    missing = [c for c in req if c not in led.columns]
    if missing:
        raise RuntimeError(f"Ledger missing columns: {missing}")

    df = led.copy()
    df["EntryDate"] = norm_day(df["entry_date"])
    df["ExitDate"]  = norm_day(df["exit_date"])
    df["Symbol"]    = df["symbol"].astype(str).str.upper().str.strip()
    df["Regime"]    = df["regime"].astype(str).str.lower().str.strip()
    df["Side"]      = df["side"].astype(str).str.lower().str.strip()
    df["SideMode"]  = df["side_mode"].astype(str).str.lower().str.strip()
    df["StopATR"]   = pd.to_numeric(df["stop_r"], errors="coerce")
    df["TargetATR"] = pd.to_numeric(df["tgt_r"], errors="coerce")
    df["Rmult"]     = pd.to_numeric(df["rmult_used"], errors="coerce")

    # keep only valid closed trades
    df = df.dropna(subset=["EntryDate","ExitDate","Symbol","Regime","Side","SideMode","StopATR","TargetATR","Rmult"]).copy()

    # unique id so appends are deduped safely
    df["__uid"] = df["pos_id"].astype(str)

    out_new = df[["EntryDate","ExitDate","Symbol","Rmult","Regime","Side","SideMode","StopATR","TargetATR","__uid"]].copy()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if os.path.exists(OUT_PATH):
        old = pd.read_csv(OUT_PATH)
        old.columns = [c.strip() for c in old.columns]
        if "__uid" not in old.columns:
            # fallback if older file existed without uid
            old["__uid"] = (
                old["Symbol"].astype(str) + "|" +
                old["EntryDate"].astype(str) + "|" +
                old["ExitDate"].astype(str) + "|" +
                old["Side"].astype(str) + "|" +
                old["StopATR"].astype(str) + "|" +
                old["TargetATR"].astype(str)
            )
        combo = pd.concat([old, out_new], ignore_index=True)
        combo = combo.drop_duplicates(subset=["__uid"], keep="last")
    else:
        combo = out_new.drop_duplicates(subset=["__uid"], keep="last")

    combo.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}: {len(combo):,} rows (added/updated {len(out_new):,})")

if __name__ == "__main__":
    main()
