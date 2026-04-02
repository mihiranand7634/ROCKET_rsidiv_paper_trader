
#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

check_paper_trader_summary.py



Purpose

- Inspect paper trader state

- Check pending orders / open positions / trade ledger

- Summarize closed-trade performance so far

- Show latest run files

- Be robust to missing / empty files

- No argparse; edit CONFIG below if needed

"""



from __future__ import annotations



import json

from pathlib import Path

from typing import Optional, Tuple, List



import pandas as pd





# =========================

# CONFIG

# =========================

ROOT = Path("/opt/rsidiv")

STATE_DIR = ROOT / "paper_trader_out" / "state"

RUN_DIR = ROOT / "paper_trader_out" / "run"

START_EQUITY = 1_000_000.0





# =========================

# HELPERS

# =========================

def safe_read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    if path.stat().st_size == 0:

        return pd.DataFrame()

    try:

        return pd.read_csv(path)

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    except Exception as e:

        print(f"[WARN] Could not read CSV: {path} -> {e}")

        return pd.DataFrame()





def safe_read_json(path: Path) -> dict:

    if not path.exists():

        return {}

    try:

        with open(path, "r", encoding="utf-8") as f:

            return json.load(f)

    except Exception as e:

        print(f"[WARN] Could not read JSON: {path} -> {e}")

        return {}





def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:

    for c in candidates:

        if c in df.columns:

            return c

    return None





def to_num(s: pd.Series) -> pd.Series:

    return pd.to_numeric(s, errors="coerce")





def print_df_head(df: pd.DataFrame, title: str, n: int = 10) -> None:

    print(f"\n=== {title} ===")

    if df.empty:

        print("EMPTY")

        return

    print(f"rows={len(df)} cols={list(df.columns)}")

    print(df.head(n).to_string(index=False))





def latest_run_dir(run_dir: Path) -> Optional[Path]:

    dirs = sorted([p for p in run_dir.glob("*") if p.is_dir()])

    if not dirs:

        return None

    return dirs[-1]





def summarize_ledger(ledger_df: pd.DataFrame) -> Tuple[dict, Optional[str], Optional[str]]:

    out = {

        "rows": len(ledger_df),

        "wins": None,

        "losses": None,

        "win_rate": None,

        "total_pnl": None,

        "avg_pnl": None,

        "median_pnl": None,

        "min_pnl": None,

        "max_pnl": None,

        "total_R": None,

        "avg_R": None,

        "median_R": None,

        "min_R": None,

        "max_R": None,

        "equity_last": None,

        "equity_return_pct": None,

    }



    if ledger_df.empty:

        return out, None, None



    pnl_col = pick_first_existing(ledger_df, ["realized_pnl", "net_pnl", "pnl"])

    r_col = pick_first_existing(ledger_df, ["r_multiple", "rr_realized", "R", "r", "rmult_used"])



    if pnl_col is not None:

        pnl = to_num(ledger_df[pnl_col]).dropna()

        if len(pnl) > 0:

            out["wins"] = int((pnl > 0).sum())

            out["losses"] = int((pnl < 0).sum())

            out["win_rate"] = float((pnl > 0).mean())

            out["total_pnl"] = float(pnl.sum())

            out["avg_pnl"] = float(pnl.mean())

            out["median_pnl"] = float(pnl.median())

            out["min_pnl"] = float(pnl.min())

            out["max_pnl"] = float(pnl.max())



    if r_col is not None:

        r = to_num(ledger_df[r_col]).dropna()

        if len(r) > 0:

            out["total_R"] = float(r.sum())

            out["avg_R"] = float(r.mean())

            out["median_R"] = float(r.median())

            out["min_R"] = float(r.min())

            out["max_R"] = float(r.max())



    if "equity_after" in ledger_df.columns:

        eq = to_num(ledger_df["equity_after"]).dropna()

        if len(eq) > 0:

            out["equity_last"] = float(eq.iloc[-1])

            out["equity_return_pct"] = float((eq.iloc[-1] / START_EQUITY - 1.0) * 100.0)



    return out, pnl_col, r_col





def print_kv(d: dict) -> None:

    for k, v in d.items():

        print(f"{k}: {v}")





# =========================

# MAIN

# =========================

def main() -> None:

    print("========================================")

    print("PAPER TRADER SUMMARY")

    print("========================================")

    print(f"ROOT: {ROOT}")

    print(f"STATE_DIR: {STATE_DIR}")

    print(f"RUN_DIR: {RUN_DIR}")

    print(f"START_EQUITY: {START_EQUITY}")



    print("\n=== STATE FILES ===")

    if STATE_DIR.exists():

        any_file = False

        for p in sorted(STATE_DIR.glob("*")):

            any_file = True

            print(f"- {p.name} ({p.stat().st_size} bytes)")

        if not any_file:

            print("STATE DIR EXISTS BUT IS EMPTY")

    else:

        print("STATE DIR MISSING")



    pending_path = STATE_DIR / "pending_orders.csv"

    open_path = STATE_DIR / "open_positions.csv"

    ledger_path = STATE_DIR / "trade_ledger.csv"

    state_json_path = STATE_DIR / "state.json"



    pending_df = safe_read_csv(pending_path)

    open_df = safe_read_csv(open_path)

    ledger_df = safe_read_csv(ledger_path)

    state_json = safe_read_json(state_json_path)



    print_df_head(pending_df, "PENDING ORDERS", n=10)

    print_df_head(open_df, "OPEN POSITIONS", n=10)

    print_df_head(ledger_df, "TRADE LEDGER", n=10)



    print("\n=== POSITION / ORDER COUNTS ===")

    print(f"pending_orders_rows: {len(pending_df)}")

    print(f"open_positions_rows: {len(open_df)}")

    print(f"trade_ledger_rows: {len(ledger_df)}")



    print("\n=== STATE.JSON ===")

    if state_json:

        print(json.dumps(state_json, indent=2, ensure_ascii=False))

    else:

        print("MISSING OR EMPTY")



    print("\n=== CLOSED-TRADE PERFORMANCE SUMMARY ===")

    ledger_summary, pnl_col, r_col = summarize_ledger(ledger_df)

    print(f"pnl_col_used: {pnl_col}")

    print(f"r_col_used: {r_col}")

    print_kv(ledger_summary)



    if not ledger_df.empty:

        for col in ["exit_reason", "side", "regime", "side_mode", "symbol"]:

            if col in ledger_df.columns:

                print(f"\n=== LEDGER value_counts({col}) ===")

                print(ledger_df[col].value_counts(dropna=False).head(20).to_string())



    if not open_df.empty:

        for col in ["side", "regime", "side_mode", "symbol"]:

            if col in open_df.columns:

                print(f"\n=== OPEN POSITIONS value_counts({col}) ===")

                print(open_df[col].value_counts(dropna=False).head(20).to_string())



    if not pending_df.empty:

        for col in ["entry_date", "side", "regime", "side_mode", "symbol"]:

            if col in pending_df.columns:

                print(f"\n=== PENDING ORDERS value_counts({col}) ===")

                print(pending_df[col].value_counts(dropna=False).head(20).to_string())



    latest = latest_run_dir(RUN_DIR)

    print("\n=== LATEST RUN DIRECTORY ===")

    if latest is None:

        print("NO RUN DIRECTORIES FOUND")

    else:

        print(latest)

        for name in ["selected_trades.csv", "orders_to_place.csv", "candidates_scored.csv"]:

            p = latest / name

            df = safe_read_csv(p)

            print(f"\n{name}: exists={p.exists()} size={p.stat().st_size if p.exists() else 'NA'}")

            if df.empty:

                print("EMPTY / MISSING")

            else:

                print(f"rows={len(df)} cols={list(df.columns)}")

                print(df.head(10).to_string(index=False))



    print("\n=== PORTFOLIO VIEW ===")

    realized_pnl = ledger_summary["total_pnl"] if ledger_summary["total_pnl"] is not None else 0.0

    realized_equity = START_EQUITY + realized_pnl



    print(f"realized_pnl: {realized_pnl}")

    print(f"realized_equity: {realized_equity}")

    print(f"realized_return_pct: {(realized_equity / START_EQUITY - 1.0) * 100.0:.4f}")



    if ledger_summary["equity_last"] is not None:

        print(f"ledger_last_equity_after: {ledger_summary['equity_last']}")

        print(f"ledger_last_equity_return_pct: {ledger_summary['equity_return_pct']:.4f}")



    print("\n=== NOTE ===")

    print("This summary reliably reports CLOSED-TRADE / REALIZED performance from trade_ledger.csv.")

    print("Unrealized MTM is not computed here unless your open_positions.csv explicitly stores it.")

    print("If you want, a separate script can be written to estimate open-position MTM from daily close data.")



    print("\n========================================")

    print("DONE")

    print("========================================")





if __name__ == "__main__":

    main()

