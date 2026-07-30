#!/usr/bin/env bash
# Brings up the local demo stack: two mock providers plus the gateway.
#
# Two providers, not one, because a failover demonstration needs somewhere to
# fail over TO. A single-provider stack can only demonstrate that an outage
# produces errors, which is not the claim this project makes.
#
# Everything binds to 127.0.0.1 and needs no credentials, so the whole stack —
# and therefore every measurement in the README — runs offline.

set -euo pipefail

cd "$(dirname "$0")/.."

BIN=bin
RUN=run
GW_PORT=${GW_PORT:-8080}
PRIMARY_PORT=${PRIMARY:-9001}
PRIMARY_ADMIN=${PRIMARY_ADMIN:-9091}
SECONDARY_PORT=${SECONDARY:-9002}
SECOND_ADMIN=${SECOND_ADMIN:-9092}

mkdir -p "$RUN" data

# wait_for_http polls a URL until it answers or the deadline passes.
#
# A fixed `sleep 2` is the usual shortcut here and it is wrong in both
# directions: too short on a loaded machine (the benchmark then measures a
# gateway that is still opening its listeners) and needlessly slow otherwise.
wait_for_http() {
  local url=$1 name=$2 deadline=$((SECONDS + 20))
  until curl -fsS -o /dev/null --max-time 1 "$url" 2>/dev/null; do
    if [ $SECONDS -ge $deadline ]; then
      echo "stack: $name did not become ready at $url within 20s" >&2
      echo "--- last 30 lines of $RUN/$name.log ---" >&2
      tail -n 30 "$RUN/$name.log" >&2 || true
      return 1
    fi
    sleep 0.1
  done
  echo "stack: $name ready at $url"
}

start() {
  if [ ! -x "$BIN/gateway" ] || [ ! -x "$BIN/mockprovider" ]; then
    echo "stack: binaries missing; run 'make build' first" >&2
    exit 1
  fi

  stop_quiet

  # Truncate the logs the reconciliation reads. Appending across runs would make
  # a later reconciliation compare this run's ledger against every previous
  # run's provider log, which shows up as thousands of "missing in ledger" rows
  # and looks like a gateway bug rather than a stale file.
  : > data/requests.jsonl
  : > data/requests-secondary.jsonl
  : > data/ledger.jsonl

  echo "stack: starting primary mock provider on :$PRIMARY_PORT (admin :$PRIMARY_ADMIN)"
  "$BIN/mockprovider" \
    -listen "127.0.0.1:$PRIMARY_PORT" \
    -admin "127.0.0.1:$PRIMARY_ADMIN" \
    -config configs/mockprovider.json \
    -log data/requests.jsonl \
    > "$RUN/primary.log" 2>&1 &
  echo $! > "$RUN/primary.pid"

  echo "stack: starting secondary mock provider on :$SECONDARY_PORT (admin :$SECOND_ADMIN)"
  "$BIN/mockprovider" \
    -listen "127.0.0.1:$SECONDARY_PORT" \
    -admin "127.0.0.1:$SECOND_ADMIN" \
    -config configs/mockprovider.json \
    -log data/requests-secondary.jsonl \
    > "$RUN/secondary.log" 2>&1 &
  echo $! > "$RUN/secondary.pid"

  wait_for_http "http://127.0.0.1:$PRIMARY_ADMIN/admin/health" primary
  wait_for_http "http://127.0.0.1:$SECOND_ADMIN/admin/health" secondary

  echo "stack: starting gateway on :$GW_PORT"
  MOCK_API_KEY=${MOCK_API_KEY:-mock-key} \
  "$BIN/gateway" \
    -config configs/gateway.example.json \
    -listen "127.0.0.1:$GW_PORT" \
    -ledger data/ledger.jsonl \
    > "$RUN/gateway.log" 2>&1 &
  echo $! > "$RUN/gateway.pid"

  wait_for_http "http://127.0.0.1:$GW_PORT/healthz" gateway

  # Readiness is a separate question from liveness: the gateway process can be
  # alive while every upstream is unusable. Waiting for /readyz is what stops a
  # benchmark from starting against a gateway that has no healthy provider yet
  # and recording the resulting 503s as a gateway defect.
  wait_for_http "http://127.0.0.1:$GW_PORT/readyz" gateway-ready

  cat <<EOF

stack up:
  gateway    http://127.0.0.1:$GW_PORT
  metrics    http://127.0.0.1:$GW_PORT/metrics
  primary    http://127.0.0.1:$PRIMARY_PORT   admin http://127.0.0.1:$PRIMARY_ADMIN
  secondary  http://127.0.0.1:$SECONDARY_PORT   admin http://127.0.0.1:$SECOND_ADMIN
  logs       $RUN/{gateway,primary,secondary}.log

kill the primary to watch failover:
  curl -XPOST 'http://127.0.0.1:$PRIMARY_ADMIN/admin/fault?down=true'
EOF
}

stop_quiet() {
  for name in gateway primary secondary; do
    local pidfile="$RUN/$name.pid"
    [ -f "$pidfile" ] || continue
    local pid
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      # SIGTERM, not SIGKILL: the gateway flushes its ledger on graceful
      # shutdown, and a SIGKILL would lose the buffered tail — which would then
      # show up as a reconciliation failure caused by the teardown rather than
      # by the gateway.
      kill -TERM "$pid" 2>/dev/null || true
      local deadline=$((SECONDS + 10))
      while kill -0 "$pid" 2>/dev/null; do
        if [ $SECONDS -ge $deadline ]; then
          echo "stack: $name (pid $pid) ignored SIGTERM, sending SIGKILL" >&2
          kill -KILL "$pid" 2>/dev/null || true
          break
        fi
        sleep 0.1
      done
    fi
    rm -f "$pidfile"
  done
}

stop() {
  stop_quiet
  echo "stack: down"
}

case "${1:-up}" in
  up)      start ;;
  down)    stop ;;
  restart) stop; start ;;
  status)
    for name in gateway primary secondary; do
      if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
        echo "$name: running (pid $(cat "$RUN/$name.pid"))"
      else
        echo "$name: stopped"
      fi
    done
    ;;
  *) echo "usage: $0 {up|down|restart|status}" >&2; exit 2 ;;
esac
