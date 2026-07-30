#!/usr/bin/env bash
# End-to-end self-check: brings up the stack, drives a batch of requests through
# the gateway, and asserts the gateway's ledger reconciles EXACTLY (among settled
# rows) against the mock providers' independent logs.
#
# This is the offline gate CI runs. It needs no credentials and no network: the
# whole thing is the gateway talking to two local mock providers, so a reader —
# or CI — can prove the central claim (cost accounting reconciles against
# provider logs) without trusting a committed number.

set -euo pipefail
cd "$(dirname "$0")/.."

N=${N:-500}
GW=${GW:-http://127.0.0.1:8080}
KEY=${KEY:-bench-key}

fail() { echo "SELFCHECK FAILED: $*" >&2; exit 1; }

cleanup() { ./scripts/stack.sh down >/dev/null 2>&1 || true; }
trap cleanup EXIT

./scripts/stack.sh up

echo "selfcheck: driving $N requests through the gateway"
./bin/gwbench -mode overhead -gateway "$GW" -key "$KEY" -n "$N" -c 16 -warmup 0 >/dev/null

echo "selfcheck: reconciling ledger against provider logs"
# Capture the JSON report and assert on it, so the check is on the numbers rather
# than on a substring of human-readable output.
OUT=$(mktemp -d)
./bin/gwbench -mode reconcile \
  -ledger data/ledger.jsonl \
  -provider-log "data/requests.jsonl,data/requests-secondary.jsonl" \
  -out "$OUT" >/dev/null

python3 - "$OUT/reconcile.json" "$N" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))["reconcile"]
want_min = int(sys.argv[2])

problems = []
if not report["exact_among_settled"]:
    problems.append(f"not exact among settled rows: {report['settled_mismatches']} settled mismatches")
if report["missing_in_ledger"] != 0:
    problems.append(f"{report['missing_in_ledger']} provider rows have no ledger counterpart (lost billing)")
if report["missing_in_provider"] != 0:
    problems.append(f"{report['missing_in_provider']} billable ledger rows have no provider record")
if report["matched"] < want_min:
    problems.append(f"only {report['matched']} rows matched, expected >= {want_min}")

if problems:
    print("RECONCILIATION PROBLEMS:")
    for p in problems:
        print("  -", p)
    sys.exit(1)

print(f"selfcheck OK: {report['matched']} rows reconcile exactly "
      f"(ledger {report['ledger_rows']} vs provider {report['provider_rows']}); "
      f"{report['estimated_mismatches']} estimated-row differences from the harness tail, 0 settled mismatches")
PY
