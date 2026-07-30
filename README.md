# llmgw — an LLM gateway: routing, failover, caching, cost control

**One OpenAI-compatible API in front of many providers, in zero-dependency Go. It fails over around a dead provider within a bounded window (demonstrated by killing a provider under 120k requests of live load, with zero client-visible errors), enforces per-tenant budgets with clear rejection semantics, streams responses with a documented mid-stream failover boundary, and keeps a cost ledger that reconciles _exactly_ against the providers' own logs — 120,901 of 120,901 rows, to the picodollar, with zero mismatches in either direction. Added latency is 1.8 ms p99 (gateway self-timed) / 3.0 ms p99 (client-observed upper bound), excluding provider time, at ~19.5k req/s on one laptop.**

> Suggested GitHub repo name: `llmgw`

---

## Why this project exists

Every company calling more than one LLM provider ends up rebuilding the same middleware: one API in front of all of them, failover when one degrades, caching, per-team budgets, and cost visibility accurate enough to bill against. It is the API-gateway category reborn for AI traffic — LiteLLM, Portkey, Kong and Cloudflare all sell a version of it — and it is one of the most-hired-for backend niches right now.

The category is easy to fake and hard to do honestly. A gateway that "supports failover" but splices two models' partial outputs together on a mid-stream retry produces a response no model ever generated. A "cost dashboard" built from `float64` drifts a fraction of a cent per request and has no defence when a customer disputes an invoice. A "reconciliation" that passes does so because someone added an epsilon to stop a nightly page. This project's design commitments are aimed squarely at those failure modes, and the results section reports where they bite.

## Zero dependencies, on purpose

The `go.mod` has no `require` block. SSE framing, the circuit breaker, the Prometheus exposition format, the HTTP surface, the token estimator — all stdlib.

A gateway sits in the request path of every LLM call a company makes, so its dependency tree is its attack surface and its p99 is its reputation. The subset of each dependency a gateway actually needs is small enough to implement correctly and test properly, and doing so removes an entire class of supply-chain and version-skew risk. Where a library would have been the conventional choice — `client_golang` for metrics, a YAML parser for config — the code says so and says why it went the other way.

A practical consequence for reproducibility: the whole thing builds, tests, and benchmarks with `go build ./...` and nothing else. No module download, no lockfile, no network.

## The four things in the title, and how each is kept honest

### Routing and failover

A route is a client-facing model alias mapped to an **ordered chain of targets** (provider instance + upstream model). The router walks the chain, and the two decisions that matter are:

- **What counts as retryable.** Every upstream failure is classified at the adapter — where the status codes and error bodies are understood — into a closed set of classes (`connect`, `timeout`, `rate_limit`, `upstream_5xx`, `overloaded`, `bad_request`, `auth`, `content_filter`, `context_length`, `cancelled`). Retryability is a table, not a chain of `if`s, so it is auditable in one glance. A `400` for a malformed tool schema is **not** retried — walking it down every provider only burns latency to produce the same `400`. A `429` or a `503` is.
- **What counts as evidence about the provider.** A client hammering malformed requests must not be able to trip a provider's circuit breaker and deny service to every other tenant. So the breaker only records outcomes where `Class.CountsAgainstHealth()` is true. A test fires 10,000 consecutive `bad_request` failures and asserts the breaker stays closed; the mirror-image test asserts sustained `5xx` does open it.

The circuit breaker trips on failure **ratio over a sliding window with a minimum sample count** (ratio alone trips on the first blip when the window holds one sample), opens for a jittered cooldown (without jitter, every replica probes a recovering provider at the same instant and re-kills it), and admits a bounded number of half-open trial requests before closing.

### Streaming, and the one boundary this project is most careful about

Once a single content byte has reached the client, the gateway **cannot** transparently retry on another provider: splicing a second model's continuation onto the first's partial output produces a sentence cut mid-clause, a fact then contradicted, a tool call duplicated — worse than an honest error. So streaming failover is defined precisely and the code enforces it:

- **Before the first content byte**, a failure is transparently retryable. The server holds the client's response headers *unwritten* until the first upstream content arrives; until then the failover window is open. This covers the common outages — connection refused, `429`, `500`, a provider that dies before its first token.
- **After the first content byte**, the failure is surfaced to the client as an explicit SSE `error` event (not a silent EOF that reads as a normal end-of-stream), the partial usage is still billed and ledgered, and no silent retry happens.

This is stated in the router's package doc, enforced by the server's SSE sink, and pinned by tests that assert the second provider is *never called* after content began, and that a mid-stream abort produces an `event: error` frame rather than a truncated-but-clean-looking stream.

### Caching

- **Tenant isolation is the default, not an option.** Cache keys are namespaced per tenant unless an operator explicitly names a shared pool. A cache that defaults to shared serves one customer's completion as another's cache hit — a data-isolation breach that is invisible in testing because a hit looks exactly like a correct response. The safe scope is the zero value; sharing has to be asked for by name.
- **The key is length-prefixed, not concatenated.** Concatenating `(model, messages…)` without length prefixes maps `("ab","c")` and `("a","bc")` to the same bytes — a cross-request collision that serves one user's answer to another's question. A test constructs boundary-shifted message sets and asserts they do not collide.
- **The `stream` flag is excluded from the key** (the same question asked streaming and non-streaming has the same answer; including it halves the hit rate), and a streamed client is served a cache hit by *replaying* the stored response as frames — with a comment stating plainly that the replayed chunk boundaries are not the original ones, because they cannot be.
- **Only deterministic requests are cached** (temperature 0), and a truncated response (`finish_reason: length`) is never cached under the default policy — a truncated answer served forever from memory is a permanent bug the client cannot retry its way out of.

### Cost control

Every monetary value is an `int64` count of **picodollars** — never `float64`. The unit is chosen so every realistic list price (quoted per million tokens) divides to an exact integer per token; a price that would not divide exactly is **rejected at config load** rather than silently rounded, which is what lets the rest of the codebase treat cost arithmetic as exact.

Two accounting traps that a naive gateway gets wrong, each with a test that fails if the handling is removed:

1. `completion_tokens` **already includes** reasoning/thinking tokens. Adding `reasoning_tokens` on top double-bills a reasoning model by roughly 10×.
2. `prompt_tokens` **already includes** cached tokens. Billing the full prompt at the input rate *and* the cached tokens at the cached rate over-bills every cache hit.

Budgets use **estimate-then-true-up**: a pre-flight token estimate reserves against the tenant's remaining budget; the real cost is committed after the response and the unused hold released. Concurrent reservations are proven not to collectively over-admit past the limit by a `-race` test firing hundreds of concurrent `Reserve`s at a small budget. A rejection carries the numbers to act on — limit, spent, remaining, reset — because "budget exceeded" with no numbers forces the client to guess.

The estimate is conservative **for admission** in the specific sense that matters: when the client sets `max_tokens`, that is a hard ceiling the provider cannot exceed, so pricing the full cap can only over-reserve. Without a client cap the completion length is genuinely unknowable and the estimate falls back to a per-tenant prior, which *can* come in low — so "conservative" is a property of the capped case, not a guarantee in general. Either way the reservation is trued up against the provider's reported usage, so a tenant is never billed from the estimate when a measurement exists.

## Measured results

All numbers below were measured on an **Apple M4 Pro (14 cores, 48 GB), Go 1.26.5, macOS**, against two local mock providers over loopback. Every figure in the five sections below is reproducible with `make bench`, and its raw JSON report is committed under [`results/`](results/). The k6 numbers further down are the exception and are labelled as such: `make loadtest` runs those scenarios but `bench-all.sh` does not, and no k6 artifact is committed. Where a number is a target rather than a guarantee, it is stated as such.

### Gateway overhead: 1.8 ms p99 self-timed, 3.0 ms p99 client-observed

5,000 requests, concurrency 32, ~19,500 req/s, 0 errors.

| overhead measurement | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|
| **self-reported** (gateway's own timer, excl. upstream) | 0.13 ms | 1.05 ms | **1.84 ms** | 2.56 ms |
| **subtractive** (client total − reported upstream; upper bound) | 0.83 ms | 1.80 ms | **2.95 ms** | 4.09 ms |

The spec asks for "added latency under 5 ms p99 **excluding provider time**", and excluding provider time is the whole subtlety. The overhead is measured **two ways and both are reported**, because reporting only the flattering one would be the easy dishonesty here:

- **self-reported** — the gateway times everything except the upstream round trip and returns it in an `X-LLMGW-Overhead-Us` header. Precise, but it is the gateway grading its own homework: it cannot see kernel time, `net/http`'s own read/write path, or scheduler delay before the handler was entered.
- **subtractive** — client-observed total minus the gateway's reported upstream time. Catches everything the self-report misses, but includes loopback and client cost, so it is an **upper bound**.

The truth is bracketed by the two. Both are under 5 ms p99. Quantiles are **exact order statistics over the full sample** (nearest-rank), not interpolated histogram buckets — a "p99 of 4.8 ms" read off a histogram whose buckets straddle 5 ms would be an artifact of the bucket layout, which is exactly the mistake a headline latency claim must not make. See [`cmd/gwbench`](cmd/gwbench/main.go) for the methodology, stated at length.

### Failover: the outage was absorbed entirely

Steady load of ~16 concurrent workers; the primary provider is **killed mid-flight** via the mock's admin endpoint at t=3 s and left dead.

| metric | value |
|---|---|
| total requests during the run | **120,885** |
| succeeded | **120,885** |
| client-visible failures | **0** |
| requests that failed over to the secondary | 16 |

The primary served the first ~3 s of the 10 s run (roughly 30% of the requests, ~36k). At t=3 s it was killed: the 16 requests in flight to it failed over transparently (they were pre-first-byte), the circuit breaker opened, and the remaining ~85k requests went to the secondary with no per-request failure. **The failover window, measured as the span in which any client saw an error, was zero** — the breaker absorbed the outage. This is the spec's headline demonstration: an outage under load produces no client-visible errors.

### Cost reconciliation: 120,901 of 120,901 rows, exact

The gateway's ledger and the mock provider's request log are produced by **independent code paths with independent price tables**. That independence is the entire point: if both sides computed cost with the same code, a bug in it would cancel out and the reconciliation would pass while every invoice was wrong. Agreement is therefore *evidence*, not a tautology. Matching is on `(request id, attempt)` with **no tolerance** — a single token or picodollar of disagreement is a mismatch.

Over a run that included the failover kill above (120,917 ledger rows vs 120,901 provider rows):

| | count |
|---|---|
| matched exactly (tokens **and** picodollar cost) | **120,901** |
| token mismatches | **0** |
| cost mismatches | **0** |
| billable ledger rows with no provider record | **0** |
| provider rows with no ledger record | **0** |

Zero mismatches and zero orphans in either direction. Two counts deserve explanation rather than being buried:

- **The 16-row ledger/provider gap** is the attempts to the killed primary. A connection that was refused never reached the provider, so it has no provider-side row and is correctly classified non-billable — the reconciler accounts for it as gateway-side rather than as a missing charge.
- **`EXACT (strict)` is still reported as `false`, and that is the reconciler being right.** Four matched rows were built from an *estimate* (a request cancelled after the provider began generating, where no usage was reported). Their numbers happen to agree with the provider's, but `Exact()` deliberately refuses to certify a run containing any estimated row: an estimator that guesses correctly has not turned a guess into a measurement. So the tool reports both — strict `false`, and `EXACT among settled rows: true` — rather than picking the flattering one.

### Caching: 100% hit rate on repeat, 4.8× median speedup

Cold pass of 400 distinct deterministic prompts, then a warm pass of the identical 400.

| | cold p50 | warm p50 | hit rate (warm) | median speedup |
|---|---|---|---|---|
| cache | 1.59 ms | 0.34 ms | **100%** | **4.8×** |

The speedup is modest by construction: the mock provider is *fast* (12 ms TTFB), so the upstream time the cache removes is small. Against a real provider taking 200–800 ms, the same cache removes that entire round trip. The benchmark measures cold and warm as separate passes over distinct prompts precisely so the "speedup" is not the cache measured against itself.

### Streaming: TTFB ≈ 14 ms, 300/300 clean, 0 truncated

300 streaming requests, measured with a real incremental SSE reader (which `k6` cannot do — it buffers the whole response).

| metric | value |
|---|---|
| time to first content frame, p50 | 14.3 ms |
| inter-frame gap, p50 | 1.1 ms |
| clean ends (terminated with `[DONE]`) | **300 / 300** |
| truncated | **0** |

A stream counts as "clean" only if it ended with the `[DONE]` sentinel; anything else is a truncation. That distinction is what makes a silently-cut stream *visible* instead of passing as a success — and it is the same check that turns the mid-stream-abort fault into a detected failure rather than a corrupt response.

### Under concurrent load (k6)

`k6` drives the full HTTP surface. All scenarios pass their thresholds:

- **steady** — 250 req/s for 12 s: 100% success, every response carries content + usage and every error is a well-formed OpenAI envelope.
- **budget** — one small-budget tenant hammered: 354 rejections, every one carrying a self-explaining, retry-informative body.
- **failover** — primary killed mid-run: failovers recorded, `http_req_failed` = 0.00%.

`k6` deliberately does **not** produce the headline overhead number: it runs on the same laptop as the gateway, so under load the two contend for cores and the gateway's self-timed overhead inflates for reasons unrelated to its code. That is documented in the script, and `cmd/gwbench` measures overhead in isolation instead.

## Bugs found and fixed

Each was found by distrusting something that looked fine, and each is now pinned.

**1. An OpenAI stream that ends without `[DONE]` was treated as a clean end.** The streaming adapter returned a normal "done" event on any clean EOF. But an OpenAI-compatible stream is only complete after its `[DONE]` sentinel — a provider that dies after flushing a well-formed final frame produces a *framing-clean* EOF that is nonetheless a truncated response. Serving it would bill a half-generated answer as a whole one. Found by the mid-stream-abort end-to-end test, which got a silent truncation instead of the expected error frame. Now EOF-without-`[DONE]` surfaces as a timeout-class failure carrying the usage seen so far.

**2. Failed-over attempts were logged under the wrong attempt number.** The server stamped `attempt: 1` into the upstream correlation and never varied it per attempt, so a request served on attempt 2 (after failover) was logged by the provider as attempt 1. The reconciliation keys on `(request id, attempt)`, so those rows never matched — a perfectly correct bill failing to reconcile. Found by the **live** failover reconciliation (22 mismatches), invisible until a provider actually failed over under load. Fixed by stamping the real attempt number per attempt; the mismatches dropped from 22 → the estimated-tail residue.

**3. A billable-but-cancelled attempt was recorded as free.** After #2, a residue of mismatches remained: requests the harness cancelled while the provider had already completed them were logged by the gateway with **zero** tokens (`context canceled`), under-counting money the provider actually charged. The ledger package's `ClassifyFailure` explicitly flags this case (`NeedsEstimate`), but the server was ignoring the flag. Now such an attempt records an *estimated* cost, flagged `usage_source: estimated`, and the reconciliation reports estimated-row differences as their own category rather than as clean matches or hidden misses.

**4. The standalone reconcile tool had its own field parser and disagreed with the real one.** `gwbench`'s reconcile mode originally hand-parsed the ledger JSON and read a flat `prompt_tokens` where the ledger nests `tokens.prompt` — so it reported total mismatches while the authoritative `ledger.Reconcile` (used by the tests) passed. A second, subtly-different reconciler is a place for two definitions of "agree" to diverge. Fixed by making `gwbench` delegate to the same `ledger.Reconcile` the tests use; the mock's log schema was aligned to `ledger.ProviderRecord` so no translation step sits between them.

The next ten were found by an adversarial audit pass (parallel reviewers over money, concurrency, streaming, the HTTP surface, and the README itself, each required to demonstrate a failure with a running test rather than assert one). Bugs 5-10 and 13-14 are pinned by regression tests that I verified fail when the fix is reverted. Bugs 11 and 12 are fixed but covered only by the existing suite, not by a dedicated test — stated rather than implied.

**5. A keep-alive frame killed the stream mid-response.** Providers emit keep-alives to hold a connection open through a long generation. `sse.Event.IsComment()` required non-empty comment text, so a bare `":\n\n"` — the cheapest form, and one real providers send — produced an event that was neither comment nor data, fell through to `json.Unmarshal`, and failed with "unexpected end of JSON input". The gateway then reported that as an upstream failure: the client was **truncated mid-response**, and the healthy provider took a health-counting failure that could open its breaker. An empty `data:\n\n` frame did the same. Fixed with an explicit `IsCommentEvent` flag plus an empty-payload guard before the parse.

**6. A billed failure charged the tenant nothing.** The failure path called `Budget.Release()`, freeing the whole hold, on the documented assumption that a failed request is free. But a `5xx` after generation began *is* billed, and the ledger recorded that cost — so the ledger showed spend the budget never saw, and a tenant could burn unlimited budget through a failing upstream and never be rejected. `Budget.Charge()` existed for exactly this case and was never called from the request path. Fixed by committing the cost the ledger actually attributed (`Commit(0)` is still a full release when nothing was billed).

**7. Anthropic responses claimed to be from 1970.** Anthropic's Messages API returns no creation timestamp, but OpenAI's schema has a required `created` field that clients render as the completion time. Both the streaming and non-streaming translations left it at the zero value, so every Anthropic-served response emitted `"created":0`. Fixed by stamping the adapter's injected clock once per response.

**8. The mock provider ignored `max_tokens`, which slandered the estimator.** The mock generated its configured reply length regardless of the client's cap. A real provider cannot exceed `max_tokens` — it stops and reports `finish_reason: "length"`. Because the benchmark sent `max_tokens: 32` and the mock emitted 48, the reconciliation showed the gateway's estimated rows *under*-counting, and an earlier version of this README repeated that as a finding about the estimator. The instrument was wrong, not the estimator: with the cap honoured, the same reconciliation shows **zero** token and cost mismatches. This is the most instructive bug in the list — a measurement rig that lies makes the thing it measures look broken.

**9. A tool-call stream was silently discarded.** The streaming sink decided "is this content?" by testing `delta.content` text only. A tool-call response carries its entire payload in `delta.tool_calls` with **no** text, so nothing ever met the bar, the response never started, and the client received only `data: [DONE]`. The same gate dropped `finish_reason` and usage frames from an empty-completion stream. Fixed with an explicit predicate covering text, tool calls, finish reason and usage — while still excluding the role-only opening frame, since keeping that out is what holds the transparent-failover window open.

**10. A truncated Anthropic stream was served as complete.** The same class of bug as #1, in the other adapter: Anthropic terminates with `message_stop`, and EOF without it means the response was cut short — but the translator emitted a clean `Done`. Fixed to surface a failure carrying the usage accumulated so far.

**11. A client disconnect was blamed on the provider.** When the gateway's own context was cancelled mid-stream (client gone, or request deadline), the read error was classified `upstream_5xx` — a health-counting failure. So every client that hung up mid-response pushed a *healthy* provider toward an open breaker. Fixed by checking `ctx.Err()` first and classifying as `cancelled`, which by design does not count against health.

**12. A streaming client could not see which provider served it.** The streaming path set none of the `X-Llmgw-*` attempt headers, because they must be written before the response is committed — which happens mid-attempt, before the router returns. Fixed with a small `ProgressSink` interface: the router publishes each target before attempting it, so the sink can stamp the provider, model and attempt count at the moment it commits.

**13. Picodollar arithmetic overflowed at realistic dollar amounts — silently, and negative.** The unit that makes cost exact is also the unit that makes int64 small: a picodollar is 1e-12 USD, so int64 tops out near $9.2M and any *intermediate* product overflows far sooner. Three places did the obvious multiply and wrapped:

  - `Limit.SoftThreshold()` computed `amount * basis_points / 10000`, which overflows above ~$922. A **$5,000 budget's 80% soft threshold came out as $310.65**, so budget degradation fired at 6% of the intended spend. At $100,000 it was $679.
  - `Totals.EstimatedShareBasisPoints()` overflowed above ~$922 of estimated cost and returned a **negative** share — a ledger with half its cost estimated reported −4223 basis points, which an operator reads as "no estimates".
  - `Totals.add` accumulated cost with an unchecked `+=`, so a rollup past ~$9.2M wrapped negative. A negative lifetime cost does not look broken; it looks like a refund.

Fixed with `money.ScaleByBasisPoints` — a quotient/remainder split that is exact for every input and cannot overflow — plus a ratio-of-quotients form for the share, and saturating addition for the rollups (clamping is wrong in an *obvious* direction; wrapping is wrong in a plausible one). The lesson is the general one about fixed-point money: choosing a unit fine enough to be exact makes the overflow boundary much closer than it looks, and every multiply needs to be written for it.

**14. `FormatUSD(math.MinInt64)` produced unparseable output.** `v = -v` does not make `MinInt64` positive, so the whole and fractional parts both rendered with their own minus signs: `-$-9223372.-36854775808`. That string reaches users through `Entry.CostUSD`. Fixed in both formatters. (`ParseUSD` still refuses that magnitude by design — its overflow guard is a safety check worth keeping, and loosening it to round-trip an unreachable value would be the wrong trade.)

## Controls and honesty checks

- **The reconciliation has no tolerance parameter, anywhere.** A one-picodollar or one-token disagreement is a mismatch. Every reconciliation that quietly passes in production does so because someone added an epsilon; after that it tests nothing.
- **Estimated cost is a first-class, separately-counted fact.** A total that mixes measured and estimated numbers is not a bill. The ledger records the source per row, the metrics split measured vs estimated cost into two counters, and the reconciliation refuses to count an estimated row as an exact match.
- **The SSE codec is fuzzed.** The round-trip property (encode→decode is lossless, and a payload can never forge a frame boundary) is the one where a bug silently corrupts every streamed response. `make fuzz` runs 30 s (~5M executions on this machine) and CI runs 20 s on every push; longer ad-hoc runs have also found nothing. No crasher has ever been produced, so there is no committed seed corpus.
- **Two must-not-happen properties are asserted, not hoped for:** a `bad_request` flood does not trip the breaker (multi-tenant isolation), and a mid-stream failure never fails over (the second provider is asserted un-called).

## Honest limitations

1. **Measured against a mock, on loopback.** The mock provider is deterministic and fast by design — it is the *instrument*, chosen so the numbers are reproducible and the reconciliation has an exact ground truth. Real-provider latency, real network jitter, and real tokenizer disagreement are not exercised here. The gateway's *overhead* is independent of provider latency, so that number transfers; the absolute end-to-end latencies do not.
2. **The token estimator is not a real tokenizer.** It is a documented approximation with a stated error characteristic, used only for budget admission (where over-estimating is the safe direction) — never for billing, which uses the provider's reported counts. Shipping a real BPE per model would be megabytes of vocabulary and still wrong for the models whose tokenizers are unpublished.
3. **Per-provider breaker tuning is coarse.** The breaker registry shares one base config across instances; the per-provider `breaker` block in the config is parsed and validated but not yet applied per instance. The wiring point is marked in `cmd/gateway`.
4. **Streaming responses are not written into the cache.** A streamed answer is cached via its non-streaming twin (the cache key excludes the stream flag), so a later non-streaming request populates the cache and a later streaming request replays it — but a purely-streaming workload never populates it. Documented in the code rather than silently doing nothing.
5. **The real OpenAI/Anthropic adapters are unit-tested against `httptest`, not live.** The Anthropic translation (system hoisting, `max_tokens` default, event-typed stream, split-usage accumulation) is covered by a table of event sequences → expected chunks, but no test hits a real endpoint — by design, so the suite runs offline.
6. **All enforcement state is per-process; this is a single-replica gateway.** Budget reservations, the response cache, the circuit breakers and the completion-length priors all live in memory in one process. Run two replicas behind a load balancer and each enforces its own budget, so a tenant's effective limit multiplies by the replica count; the cache hit rate divides by it; and each replica learns provider health independently. Making this horizontally scalable means moving reservations and breaker state to something shared (Redis is the conventional answer, and the reservation protocol here — reserve, commit, expire — was designed to port to it), which is real work that is **not** done. Every headline number above was measured on one replica, where the numbers are honest.

## Tech stack

| Component | Choice |
|---|---|
| Language | Go 1.26, **zero external dependencies** (stdlib only) |
| Internal wire format | OpenAI `chat.completions` schema (also the public API) |
| Providers | OpenAI adapter (fronts the mock too), Anthropic adapter with real bidirectional translation |
| Cost | integer picodollars; inexact prices rejected at load time |
| Breaker | 3-state, ratio-over-window with min-sample, jittered exponential backoff |
| Cache | byte-bounded LRU, per-tenant scoping, deterministic-only, lazy + swept expiry |
| Metrics | hand-rolled Prometheus text exposition; exact-sample recorder for headline numbers |
| Config | JSON with exhaustive load-time validation, every error naming the offending path |
| Load test | `k6`; overhead/failover/reconcile measured by `cmd/gwbench` |
| CI | GitHub Actions: gofmt, vet, `go test -race`, SSE fuzz, end-to-end reconciliation self-check |

## Quickstart

No credentials, no network — everything runs against local mock providers.

```bash
make build          # builds gateway, mockprovider, gwbench (stdlib only)
make test           # all tests
make race           # the whole suite under -race (the important one)
make selfcheck      # end-to-end: real traffic through the gateway, reconciled exactly
```

Reproduce every number in this README:

```bash
make bench          # brings up the stack, runs all measurements, tears it down
```

Run the stack and poke it by hand:

```bash
make stack-up
curl -s localhost:8080/v1/chat/completions \
  -H 'Authorization: Bearer bench-key' -H 'Content-Type: application/json' \
  -d '{"model":"gw-chat","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'

# watch failover: kill the primary and keep sending traffic
curl -XPOST 'localhost:9091/admin/fault?down=true'
make stack-down
```

Point it at real providers by editing `configs/gateway.example.json` (or `bin/gateway -print-example`) — set a provider's `vendor` to `openai`/`anthropic`, its `base_url`, and its `api_key_env` (the *name* of an environment variable; the key itself is never written in the config). Then `export OPENAI_API_KEY=…` and run.

## Repository layout

```
llmgw/
├── cmd/
│   ├── gateway/         # the server: composition root wiring every dependency
│   ├── mockprovider/    # programmable OpenAI-compatible mock upstream (the instrument)
│   └── gwbench/         # overhead / failover / cache / stream / reconcile measurements
├── internal/
│   ├── apiv1/           # OpenAI wire types; union-typed content & stop fields
│   ├── money/           # integer picodollar arithmetic; inexact prices rejected
│   ├── pricing/         # exact cost model; the two "already-included" traps
│   ├── tokens/          # pre-flight estimator + deterministic reference tokenizer
│   ├── sse/             # SSE framing both directions; fuzzed; truncation ≠ clean end
│   ├── breaker/         # 3-state circuit breaker + active prober
│   ├── budget/          # estimate-then-true-up; proven no concurrent over-admission
│   ├── cache/           # byte-bounded LRU; tenant-isolated by default
│   ├── ledger/          # append-only cost rows; exact reconciliation, no tolerance
│   ├── provider/        # Provider contract; OpenAI + Anthropic adapters; failure classes
│   ├── router/          # failover chain; the streaming honesty boundary
│   ├── config/          # JSON config with exhaustive, path-naming validation
│   ├── metrics/         # dependency-free Prometheus; interpolated vs exact quantiles
│   ├── mockprov/        # the mock provider's engine (independent cost accounting)
│   └── server/          # the HTTP request pipeline; ties it all together
├── configs/             # gateway + mock provider example configs
├── scripts/             # stack.sh, bench-all.sh, selfcheck.sh, loadtest.js (k6)
└── results/             # committed benchmark JSON — every number above reproduces from here
```

## The claim, stated narrowly

On one laptop against local mock providers: the gateway adds under 2 ms p99 of its own overhead (under 3 ms observed from outside) at ~19.5k req/s; it absorbs a provider outage under 120k requests of load with zero client-visible errors; its cost ledger reconciles against independently-computed provider logs with zero mismatches across 120,901 rows; and it enforces per-tenant budgets and tenant-isolated caching with the correctness properties above pinned by tests. It does **not** claim real-provider latencies, a production-grade tokenizer, or per-instance breaker tuning — those limits are listed above. The bugs in "Bugs found" are there because the reconciliation, the mid-stream tests and an adversarial audit caught them, and I would rather that class of error keep being caught by the harness than by a reader.
