
#!/usr/bin/env bash

set -euo pipefail



export DO_BACKFILL=1

exec /opt/rsidiv/bin/run_daily_lightsail_workflow.sh

