#!/usr/bin/env bash
# Reproduces every number in the README into a fresh results directory.
#
# It brings up the demo stack, runs each measurement, kills a provider for the
# failover run, reconciles, and tears the stack down. Everything is offline and
# needs no credentials, so a reader can regenerate the committed numbers and
# check them.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-results/run-$(date -u +%Y%m%d-%H%M%S)}
LABEL=${LABEL:-"$(uname -sm)"}
PRIMARY_ADMIN=${PRIMARY_ADMIN:-9091}
GW=${GW:-http://127.0.0.1:8080}

mkdir -p "$OUT"
echo "bench-all: writing results to $OUT"

cleanup() { ./scripts/stack.sh down >/dev/null 2>&1 || true; }
trap cleanup EXIT

./scripts/stack.sh up

echo "== overhead =="
./bin/gwbench -mode overhead -gateway "$GW" -n 5000 -c 32 -out "$OUT" -label "$LABEL"

echo "== streaming =="
./bin/gwbench -mode stream -gateway "$GW" -n 300 -c 16 -out "$OUT" -label "$LABEL"

echo "== cache =="
./bin/gwbench -mode cache -gateway "$GW" -n 400 -c 16 -out "$OUT" -label "$LABEL"

echo "== failover (kills the primary mid-load) =="
./bin/gwbench -mode failover -gateway "$GW" -admin "http://127.0.0.1:$PRIMARY_ADMIN" \
  -c 16 -kill-after 3s -duration 10s -out "$OUT" -label "$LABEL: primary killed at t=3s"

echo "== reconciliation (ledger vs both provider logs) =="
./bin/gwbench -mode reconcile \
  -ledger data/ledger.jsonl \
  -provider-log "data/requests.jsonl,data/requests-secondary.jsonl" \
  -out "$OUT" -label "$LABEL: post-failover"

echo
echo "bench-all: done. Artifacts in $OUT:"
ls -1 "$OUT"
