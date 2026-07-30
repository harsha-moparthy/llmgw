# Every target here runs offline with no credentials. The measurements are taken
# against the local mock provider, which is what makes the numbers in the README
# reproducible by a reader rather than a claim they have to trust.

GO      ?= go
GOFLAGS ?=
BIN     := bin
STAMP   := $(shell date -u +%Y%m%d-%H%M%S)
RESULTS := results/run-$(STAMP)

# Ports the demo stack uses. Two mock providers so failover has somewhere to go.
GW_PORT      ?= 8080
PRIMARY      ?= 9001
PRIMARY_ADMIN?= 9091
SECONDARY    ?= 9002
SECOND_ADMIN ?= 9092

.PHONY: all
all: lint test build

.PHONY: build
build:
	@mkdir -p $(BIN)
	$(GO) build $(GOFLAGS) -o $(BIN)/gateway      ./cmd/gateway
	$(GO) build $(GOFLAGS) -o $(BIN)/mockprovider ./cmd/mockprovider
	$(GO) build $(GOFLAGS) -o $(BIN)/gwbench      ./cmd/gwbench

.PHONY: test
test:
	$(GO) test ./... -count=1

# The race detector is not optional for this project. A gateway is a concurrency
# artifact: the budget reservation path, the circuit breaker, the cache and the
# metrics registry are all touched by every in-flight request simultaneously, and
# a data race there is the kind of bug that shows up as an inexplicable billing
# discrepancy in production rather than as a crash in CI.
.PHONY: race
race:
	$(GO) test ./... -race -count=1

.PHONY: cover
cover:
	$(GO) test ./... -coverprofile=coverage.out -covermode=atomic
	$(GO) tool cover -func=coverage.out | tail -30

.PHONY: lint
lint:
	$(GO) vet ./...
	@gofmt -l . | grep -v '^$$' && { echo "gofmt: files need formatting (run 'make fmt')"; exit 1; } || echo "gofmt: clean"

.PHONY: fmt
fmt:
	gofmt -w .

# Brief fuzzing of the SSE codec. Framing is the one place in this gateway where
# a parser bug silently corrupts every streamed response rather than failing
# loudly, so it gets fuzzed rather than only unit-tested.
.PHONY: fuzz
fuzz:
	$(GO) test ./internal/sse/ -run=Fuzz -fuzz=FuzzRoundTrip -fuzztime=30s

.PHONY: selfcheck
selfcheck: build
	./scripts/selfcheck.sh

# --- the demo stack -------------------------------------------------------

.PHONY: stack-up
stack-up: build
	@./scripts/stack.sh up

.PHONY: stack-down
stack-down:
	@./scripts/stack.sh down

# --- measurements ---------------------------------------------------------

# Reproduces every number in the README into a fresh timestamped directory.
.PHONY: bench
bench: build
	@./scripts/bench-all.sh $(RESULTS)

.PHONY: bench-overhead
bench-overhead: build
	$(BIN)/gwbench -mode overhead -n 3000 -c 32 -out $(RESULTS)

.PHONY: bench-failover
bench-failover: build
	$(BIN)/gwbench -mode failover -admin http://127.0.0.1:$(PRIMARY_ADMIN) -out $(RESULTS)

.PHONY: bench-reconcile
bench-reconcile: build
	$(BIN)/gwbench -mode reconcile -out $(RESULTS)

.PHONY: loadtest
loadtest:
	k6 run scripts/loadtest.js

.PHONY: clean
clean:
	rm -rf $(BIN) coverage.out data/requests.jsonl data/ledger.jsonl run/
