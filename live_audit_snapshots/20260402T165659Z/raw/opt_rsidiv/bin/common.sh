
#!/usr/bin/env bash

set -euo pipefail



export RSIDIV_ROOT="/opt/rsidiv"



log() {

  printf '%s | %s\n' "$(date '+%F %T')" "$*"

}



load_env() {

  cd "$RSIDIV_ROOT"



  if [[ ! -f "$RSIDIV_ROOT/.env" ]]; then

    echo "Missing $RSIDIV_ROOT/.env" >&2

    exit 1

  fi



  source "$RSIDIV_ROOT/.venv/bin/activate"



  set -a

  source "$RSIDIV_ROOT/.env"

  if [[ -f "$RSIDIV_ROOT/.env.local" ]]; then

    source "$RSIDIV_ROOT/.env.local"

  fi

  set +a



  export TZ="${TZ:-Asia/Kolkata}"

  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

  export PATH="$HOME/.local/bin:$PATH"



  mkdir -p "$RSIDIV_ROOT/logs"

  mkdir -p "$RSIDIV_ROOT/debug_login"

  mkdir -p "$RSIDIV_ROOT/data/ohlcv_daily"

  mkdir -p "$RSIDIV_ROOT/signals"

  mkdir -p "$RSIDIV_ROOT/paper_trader_out/run"

  mkdir -p "$RSIDIV_ROOT/paper_trader_out/state"

  mkdir -p /opt/kite/secrets



  cd "$RSIDIV_ROOT"

}



ensure_git_ready() {

  git config user.name "lightsail-bot"

  git config user.email "lightsail-bot@users.noreply.github.com"

}



sync_repo() {

  log "Syncing repo from ${GIT_REMOTE}/${REPO_BRANCH}"

  git fetch "$GIT_REMOTE" "$REPO_BRANCH"

  git checkout "$REPO_BRANCH"

  git pull --ff-only "$GIT_REMOTE" "$REPO_BRANCH"

}



commit_and_push_staged() {

  local commit_msg="$1"



  if git diff --cached --quiet; then

    log "No staged changes to commit for: $commit_msg"

    return 0

  fi



  git commit -m "$commit_msg"



  if ! git push "$GIT_REMOTE" "HEAD:$REPO_BRANCH"; then

    log "Initial push failed; retrying after rebase"

    git pull --rebase "$GIT_REMOTE" "$REPO_BRANCH"

    git push "$GIT_REMOTE" "HEAD:$REPO_BRANCH"

  fi

}



detect_chrome_if_needed() {

  if [[ -z "${CHROME_BIN:-}" ]]; then

    for candidate in /usr/bin/google-chrome /usr/bin/google-chrome-stable /usr/bin/chromium /usr/bin/chromium-browser; do

      if [[ -x "$candidate" ]]; then

        export CHROME_BIN="$candidate"

        break

      fi

    done

  fi

}



verify_token_json() {

  python - <<'PY'

import json, os

p = os.environ["KITE_TOKEN_PATH"]

with open(p, "r", encoding="utf-8") as f:

    j = json.load(f)



api_key = j.get("api_key")

access_token = j.get("access_token")



if not api_key or not access_token:

    raise SystemExit("Token JSON missing api_key/access_token")



print(

    f"token_ok path={p} user_id={j.get('user_id')} "

    f"generated_at={j.get('generated_at')} token_last4={access_token[-4:]}"

)

PY

}



run_token_refresh() {

  detect_chrome_if_needed

  mkdir -p "$(dirname "$KITE_TOKEN_PATH")" "$DEBUG_DIR"



  log "Refreshing Kite token"

  if [[ "${HEADLESS:-0}" == "0" ]]; then

    xvfb-run -a -s "-screen 0 1280x900x24" python scripts/get_kite_access_token_selenium.py

  else

    python scripts/get_kite_access_token_selenium.py

  fi



  verify_token_json

}



download_baseline_release_if_needed() {

  if [[ "${DOWNLOAD_BASELINE_FROM_RELEASE:-1}" != "1" ]]; then

    if [[ ! -s "${BASELINE_RESULTS_PATH}" ]]; then

      echo "Missing BASELINE_RESULTS_PATH=${BASELINE_RESULTS_PATH}" >&2

      exit 1

    fi

    return 0

  fi



  mkdir -p "$(dirname "$BASELINE_RESULTS_PATH")"



  local asset_name

  asset_name="$(basename "$BASELINE_RESULTS_PATH")"

  local url

  url="https://github.com/${GITHUB_REPOSITORY}/releases/download/${RELEASE_TAG}/${asset_name}"



  log "Downloading ${asset_name} from public release URL"

  curl -L --fail --retry 3 --retry-delay 2 -o "${BASELINE_RESULTS_PATH}" "$url"



  if [[ ! -s "${BASELINE_RESULTS_PATH}" ]]; then

    echo "Baseline release asset download failed: ${BASELINE_RESULTS_PATH}" >&2

    exit 1

  fi



  ls -lh "${BASELINE_RESULTS_PATH}"

}

