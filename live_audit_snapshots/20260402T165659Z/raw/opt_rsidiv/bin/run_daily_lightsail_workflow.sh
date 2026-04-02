
#!/usr/bin/env bash

set -euo pipefail

shopt -s nullglob



source /opt/rsidiv/bin/common.sh



load_env

ensure_git_ready

sync_repo



log "========== DAILY WORKFLOW START =========="



run_token_refresh



if [[ "${DO_BACKFILL:-0}" == "1" ]]; then

  log "Running BACKFILL stage"

  python scripts/backfill_ohlcv_7y.py

else

  log "Running DAILY OHLCV UPDATE stage"

  python scripts/update_ohlcv_7y.py

fi



log "Building ohlcv_master.csv.gz"

python scripts/build_ohlcv_master_from_daily.py



git add data/ohlcv_daily data/ohlcv_master.csv.gz || true

commit_and_push_staged "Update OHLCV daily store + master"



download_baseline_release_if_needed



log "Building divergence signals"

python -u scripts/build_divergence_signals_from_daily_store.py

test -f signals/divergences_latest.csv



log "Running paper trader"

python -u scripts/rsidiv_daily_retrain_paper_trader_option1.py



if [[ -s paper_trader_out/state/trade_ledger.csv ]]; then

  log "Exporting closed paper trades to rolling trainset"

  python -u scripts/export_paper_closed_trades_to_trainset.py

else

  log "No paper_trader_out/state/trade_ledger.csv yet; skipping exporter"

fi



log "Pruning old run folders older than 45 days"

if [[ -d paper_trader_out/run ]]; then

  find paper_trader_out/run -mindepth 1 -maxdepth 1 -type d -mtime +45 -print -exec rm -rf {} \; || true

fi



rm -f data/portfolio_run.zip || true



if [[ -f signals/divergences_latest.csv ]]; then

  git add signals/divergences_latest.csv || true

fi



if [[ -d paper_trader_out/state ]]; then

  git add paper_trader_out/state || true

fi



for f in paper_trader_out/run/*/orders_to_place.csv; do

  git add "$f" || true

done



for f in paper_trader_out/run/*/selected_trades.csv; do

  git add "$f" || true

done



for f in paper_trader_out/run/*/candidates_scored.csv; do

  git add "$f" || true

done



if [[ -f data/train_trades_closed.csv ]]; then

  git add data/train_trades_closed.csv || true

fi



commit_and_push_staged "paper-trade: update signals/state/orders"



date '+%F %T %Z' > /opt/rsidiv/paper_trader_out/state/last_daily_success.txt



log "========== DAILY WORKFLOW SUCCESS =========="

