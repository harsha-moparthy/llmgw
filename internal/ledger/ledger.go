// Package ledger is the gateway's append-only cost record, and it is designed
// around one claim: the numbers it holds must reconcile *exactly* against the
// provider's own logs, field for field, with no tolerance window.
//
// Three commitments follow from that claim and shape everything in this file.
//
// First, one row per upstream attempt, not one row per client request. A request
// that fails over across three providers cost money at up to three of them and
// produced one client-visible response, so it gets up to three cost rows exactly
// one of which is marked ServedClient. Collapsing failover into a single row is
// the single most common way a gateway under-reports spend, and it under-reports
// precisely when the fleet is unhealthy and the operator most needs the number.
//
// Second, estimated usage is a first-class, queryable property of a row, never a
// silent substitution. A total that mixes provider-measured tokens with the
// gateway's own guesses is not a bill; it is an opinion. Report.Exact refuses to
// certify a reconciliation that includes even one estimated row, however well
// the arithmetic happens to line up (see reconcile.go).
//
// Third, the durable artifact is line-delimited JSON, one self-describing entry
// per line, holding the rates that were applied and not just the resulting
// total. That makes the file diffable against a provider export by an offline
// tool — and that diff *is* the reconciliation — and it makes a row from a year
// ago re-derivable after the pricing table has moved on.
package ledger

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// UsageSource records where an entry's token counts came from. It is a string
// in the persisted form so a human reading the JSONL, or a script grepping it,
// cannot mistake an estimate for a measurement.
type UsageSource string

// Usage sources. There are three and not two on purpose: "no tokens were
// consumed" is a measurement, not an estimate, and lumping a connection-refused
// attempt in with a guessed one would make every "does this period contain
// estimates?" query answer yes.
const (
	// SourceReported means the provider told us these counts. Only rows with
	// this source may participate in a reconciliation that claims exactness.
	SourceReported UsageSource = "reported"
	// SourceEstimated means the gateway computed the counts itself, because the
	// provider did not report usage (a stream without stream_options.include_usage,
	// or an attempt that died mid-generation). The estimator over-estimates by
	// construction, so a row with this source is an upper bound, not a fact.
	SourceEstimated UsageSource = "estimated"
	// SourceNone means no tokens were consumed at all: the attempt never reached
	// a model, or the response came from the gateway's own cache.
	SourceNone UsageSource = "none"
)

// Valid reports whether the source is one of the three known values.
func (s UsageSource) Valid() bool {
	switch s {
	case SourceReported, SourceEstimated, SourceNone:
		return true
	}
	return false
}

// Status is the terminal state of one upstream attempt.
type Status string

// Attempt statuses.
const (
	// StatusSucceeded: the attempt produced a complete, usable response.
	StatusSucceeded Status = "succeeded"
	// StatusFailedOver: the attempt failed and the request was retried on
	// another provider. The client never saw this attempt, but we may still have
	// been billed for it.
	StatusFailedOver Status = "failed_over"
	// StatusFailed: the attempt failed and the request failed with it, so the
	// client saw an error. Distinguished from StatusFailedOver because "how much
	// did we spend on requests that ultimately failed" and "how much did we
	// spend on retries that eventually worked" are different questions with
	// different owners.
	StatusFailed Status = "failed"
	// StatusCancelled: the client disconnected. Real providers bill for
	// generation already performed, so these rows are frequently non-zero.
	StatusCancelled Status = "cancelled"
	// StatusCacheHit: served from the gateway's own response cache. No upstream
	// attempt happened; the row exists so that cache savings are auditable
	// against what the request would otherwise have cost.
	StatusCacheHit Status = "cache_hit"
	// StatusRejected: refused before any upstream call (budget, validation).
	// Zero cost by definition, recorded so that a rejection is not invisible.
	StatusRejected Status = "rejected"
)

// Valid reports whether the status is a known value.
func (s Status) Valid() bool {
	switch s {
	case StatusSucceeded, StatusFailedOver, StatusFailed,
		StatusCancelled, StatusCacheHit, StatusRejected:
		return true
	}
	return false
}

// TokenCounts is the token accounting for one attempt.
//
// The containment rules matter and are enforced by Entry.Validate: Cached is a
// subset of Prompt (providers report prompt_tokens inclusive of cache reads) and
// Reasoning is a subset of Completion (reasoning tokens are billed as output and
// are already inside completion_tokens). Storing them as subsets rather than as
// additional buckets is what stops the cost model from double-charging, and a
// provider log that violates the rule is reporting something we should refuse to
// bill rather than quietly accept.
type TokenCounts struct {
	Prompt     int `json:"prompt"`
	Cached     int `json:"cached"`
	Completion int `json:"completion"`
	Reasoning  int `json:"reasoning"`
}

// Total is prompt plus completion. Cached and Reasoning are deliberately not
// added: they are already counted inside Prompt and Completion respectively.
func (t TokenCounts) Total() int { return t.Prompt + t.Completion }

// BillablePrompt is the prompt tokens charged at the full input rate, i.e. the
// ones that were not served from the provider's prefix cache.
func (t TokenCounts) BillablePrompt() int { return t.Prompt - t.Cached }

// IsZero reports whether no tokens at all were counted.
func (t TokenCounts) IsZero() bool {
	return t.Prompt == 0 && t.Cached == 0 && t.Completion == 0 && t.Reasoning == 0
}

func (t *TokenCounts) add(o TokenCounts) {
	t.Prompt += o.Prompt
	t.Cached += o.Cached
	t.Completion += o.Completion
	t.Reasoning += o.Reasoning
}

// FromUsage converts a provider-reported apiv1.Usage into TokenCounts.
func FromUsage(u *apiv1.Usage) TokenCounts {
	if u == nil {
		return TokenCounts{}
	}
	return TokenCounts{
		Prompt:     u.PromptTokens,
		Cached:     u.CachedPromptTokens(),
		Completion: u.CompletionTokens,
		Reasoning:  u.ReasoningTokens(),
	}
}

// Rates records the per-token prices that were applied, in picodollars per
// token. Persisting the rates alongside the total is what makes a historical row
// auditable: a price change six months from now must not silently invalidate
// last quarter's invoice, and "recompute it from today's pricing table" is not
// an answer a finance team accepts.
type Rates struct {
	Prompt     money.Pico `json:"prompt"`
	Cached     money.Pico `json:"cached"`
	Completion money.Pico `json:"completion"`
	Reasoning  money.Pico `json:"reasoning"`
	// Version identifies the pricing table revision these rates came from.
	Version string `json:"version,omitempty"`
}

// CostBreakdown decomposes an entry's cost into the components that produced it.
//
// This type lives in ledger rather than being an alias for pricing's own
// breakdown on purpose: it is the *persisted schema*, and the persisted schema
// must not be a hostage to a refactor of the pricing package's internals. The
// pricing package computes; this struct is the shape that computation is
// recorded in, and Append enforces that the components sum to the total so the
// two can never drift.
//
// Reasoning is the one component that needs a rule. Reasoning tokens are
// already inside the completion count, so ReasoningPico is non-zero only when
// the pricing table charges reasoning at a distinct rate — in which case
// CompletionPico covers Completion-minus-Reasoning tokens. When reasoning bills
// at the plain output rate, ReasoningPico is zero and those tokens are already
// paid for inside CompletionPico.
type CostBreakdown struct {
	// PromptPico is the charge for uncached prompt tokens.
	PromptPico money.Pico `json:"prompt_pico"`
	// CachedPico is the charge for prompt tokens the provider served from its
	// own prefix cache, usually at a steep discount.
	CachedPico money.Pico `json:"cached_pico"`
	// CompletionPico is the charge for completion tokens.
	CompletionPico money.Pico `json:"completion_pico"`
	// ReasoningPico is the separately-rated charge for reasoning tokens; see the
	// type comment.
	ReasoningPico money.Pico `json:"reasoning_pico"`
	// Rates are the prices applied.
	Rates Rates `json:"rates"`
}

// Total sums the components.
func (b CostBreakdown) Total() money.Pico {
	return b.PromptPico + b.CachedPico + b.CompletionPico + b.ReasoningPico
}

// Entry is one durable cost row: exactly one upstream attempt, or one
// gateway-side terminal event (cache hit, budget rejection) that consumed no
// upstream tokens.
type Entry struct {
	// Seq is a monotonic per-ledger sequence number assigned by Append. It gives
	// the JSONL a total order that survives a reader that sorts by timestamp and
	// finds ties at clock resolution.
	Seq int64 `json:"seq"`

	RequestID string `json:"request_id"`
	Tenant    string `json:"tenant"`

	// RequestedModel is the alias the client asked for ("gpt-4o-mini"), which is
	// what the tenant is billed against and what appears in their invoice.
	RequestedModel string `json:"requested_model"`
	// Provider is the provider *instance* name ("openai-primary"), not the
	// vendor: a deployment fronts the same vendor several times with different
	// keys and quota pools, and cost attribution has to follow the key.
	Provider string `json:"provider,omitempty"`
	// UpstreamModel is the model name actually sent upstream, which may differ
	// from RequestedModel after a routing decision or a degraded-budget
	// downgrade. Keeping both is how "why is this row cheaper than the alias
	// implies" is answerable.
	UpstreamModel string `json:"upstream_model,omitempty"`

	// Attempt is 1-based within the request. Attempt 2 existing means attempt 1
	// failed, and both may have cost money.
	Attempt int `json:"attempt"`
	// ServedClient marks the one attempt whose bytes the client actually
	// received. At most one entry per request may set it.
	ServedClient bool `json:"served_client"`
	Streaming    bool `json:"streaming,omitempty"`

	Tokens      TokenCounts `json:"tokens"`
	UsageSource UsageSource `json:"usage_source"`

	// CostPico is the authoritative cost of this attempt. Breakdown must sum to
	// it exactly; Append rejects an entry where it does not.
	CostPico  money.Pico    `json:"cost_pico"`
	Breakdown CostBreakdown `json:"breakdown"`
	// CostUSD is a human-readable rendering of CostPico, recomputed by Append so
	// it can never disagree with it. It exists because these numbers are
	// routinely below a cent and nobody can read picodollars at a glance.
	CostUSD string `json:"cost_usd"`

	// Billable records whether the provider plausibly charged us for this
	// attempt, from provider.Failure.MayHaveBilled. Only billable rows are
	// expected to appear in a provider-side log, so reconciliation uses this to
	// avoid reporting every connection failure as a discrepancy.
	Billable bool `json:"billable"`

	// CacheHit means the gateway's own response cache served this request, so no
	// upstream call happened. It is not the same thing as Tokens.Cached, which
	// is the *provider's* prefix cache inside a real upstream call. Conflating
	// the two makes cache-savings numbers meaningless.
	CacheHit bool `json:"cache_hit,omitempty"`

	Status       Status `json:"status"`
	FailureClass string `json:"failure_class,omitempty"`
	StatusCode   int    `json:"status_code,omitempty"`
	ErrorMessage string `json:"error_message,omitempty"`

	StartedAt time.Time `json:"started_at"`
	EndedAt   time.Time `json:"ended_at"`
	LatencyMS int64     `json:"latency_ms"`
	// RecordedAt is stamped by Append from the ledger's clock, so the file has a
	// write-time ordering independent of whatever the caller put in StartedAt.
	RecordedAt time.Time `json:"recorded_at"`
}

// Key identifies an attempt for reconciliation purposes.
type Key struct {
	RequestID string `json:"request_id"`
	Attempt   int    `json:"attempt"`
}

func (k Key) String() string { return fmt.Sprintf("%s#%d", k.RequestID, k.Attempt) }

// Key returns the entry's reconciliation key.
func (e *Entry) Key() Key { return Key{RequestID: e.RequestID, Attempt: e.Attempt} }

// Estimated reports whether this row's numbers are a guess.
func (e *Entry) Estimated() bool { return e.UsageSource == SourceEstimated }

// ReachedProvider reports whether this row describes a real upstream attempt, as
// opposed to a gateway-side terminal event. Reconciliation only expects rows
// that reached a provider to appear in that provider's log.
func (e *Entry) ReachedProvider() bool {
	return e.Status != StatusCacheHit && e.Status != StatusRejected
}

// Validate checks the invariants that make a row billable. It is applied on
// every Append: a ledger that accepts an internally inconsistent row has
// already lost the argument with the finance team.
func (e *Entry) Validate() error {
	if e.RequestID == "" {
		return errors.New("ledger: entry has no request id")
	}
	if e.Tenant == "" {
		return errors.New("ledger: entry has no tenant")
	}
	if e.RequestedModel == "" {
		return errors.New("ledger: entry has no requested model")
	}
	if e.Attempt < 1 {
		return fmt.Errorf("ledger: attempt must be >= 1, got %d", e.Attempt)
	}
	if !e.Status.Valid() {
		return fmt.Errorf("ledger: unknown status %q", e.Status)
	}
	if !e.UsageSource.Valid() {
		return fmt.Errorf("ledger: unknown usage source %q", e.UsageSource)
	}
	t := e.Tokens
	if t.Prompt < 0 || t.Cached < 0 || t.Completion < 0 || t.Reasoning < 0 {
		return fmt.Errorf("ledger: negative token count %+v", t)
	}
	if t.Cached > t.Prompt {
		return fmt.Errorf("ledger: cached tokens (%d) exceed prompt tokens (%d); "+
			"cached is a subset of prompt, not an addition to it", t.Cached, t.Prompt)
	}
	if t.Reasoning > t.Completion {
		return fmt.Errorf("ledger: reasoning tokens (%d) exceed completion tokens (%d); "+
			"reasoning is billed inside the completion count", t.Reasoning, t.Completion)
	}
	if e.CostPico < 0 {
		return fmt.Errorf("ledger: negative cost %d pico", int64(e.CostPico))
	}
	if got := e.Breakdown.Total(); got != e.CostPico {
		return fmt.Errorf("ledger: breakdown sums to %d pico but cost is %d pico "+
			"(delta %d); a total that does not equal its parts cannot be reconciled",
			int64(got), int64(e.CostPico), int64(got-e.CostPico))
	}
	if e.UsageSource == SourceNone && !t.IsZero() {
		return fmt.Errorf("ledger: usage source %q with non-zero tokens %+v", SourceNone, t)
	}
	if e.CostPico > 0 && e.UsageSource == SourceNone {
		return fmt.Errorf("ledger: cost %d pico attributed to zero tokens", int64(e.CostPico))
	}
	if e.CacheHit {
		if e.CostPico != 0 {
			return fmt.Errorf("ledger: cache hit with non-zero upstream cost %d pico; "+
				"a gateway cache hit consumed no upstream tokens (billing a tenant for "+
				"cache hits is a pricing policy, not a provider cost)", int64(e.CostPico))
		}
		if e.Billable {
			return errors.New("ledger: cache hit marked billable")
		}
	}
	if !e.Billable && e.CostPico != 0 {
		return fmt.Errorf("ledger: non-billable attempt with cost %d pico", int64(e.CostPico))
	}
	return nil
}

// AttemptRecord is the ledger's policy answer to "how should this failed attempt
// appear?". It is computed by ClassifyFailure and consumed by the request
// handler, which owns pricing and therefore owns turning tokens into money.
type AttemptRecord struct {
	// Status is the terminal status to record. Callers that know the request
	// went on to succeed elsewhere should keep StatusFailedOver; the classifier
	// picks the pessimistic StatusFailed when it cannot know.
	Status Status
	// Billable is provider.Failure.MayHaveBilled: whether the upstream
	// plausibly charged us.
	Billable bool
	// Tokens are the counts known at failure time, zero if none were reported.
	Tokens TokenCounts
	// Source is the usage source the row must carry.
	Source UsageSource
	// NeedsEstimate is true when the attempt was billable but the provider told
	// us nothing about usage, so the caller MUST substitute an estimate. This is
	// the case that quietly turns into an under-count if it is ignored: a
	// timeout at 4000 completion tokens is billed by the provider and would
	// otherwise be recorded as free.
	NeedsEstimate bool
}

// ClassifyFailure derives the ledger-facing shape of a failed attempt from a
// provider failure.
//
// It is a pure function so the policy is testable on its own, without a file or
// a pricing table, and so that the "does a timeout get a row?" question has
// exactly one answer in the codebase.
func ClassifyFailure(f *provider.Failure, failedOver bool) AttemptRecord {
	rec := AttemptRecord{Status: StatusFailed, Source: SourceNone}
	if failedOver {
		rec.Status = StatusFailedOver
	}
	if f == nil {
		return rec
	}
	if f.Class == provider.ClassCancelled && !failedOver {
		rec.Status = StatusCancelled
	}
	rec.Billable = f.MayHaveBilled()
	switch {
	case f.UsageAtFailure != nil:
		rec.Tokens = FromUsage(f.UsageAtFailure)
		rec.Source = SourceReported
		if rec.Tokens.IsZero() {
			// The provider explicitly told us zero. That is a measurement, and
			// it means there is nothing to bill even though the class said the
			// attempt might have been billed.
			rec.Source = SourceNone
			rec.Billable = false
		}
	case rec.Billable:
		// Billable but usage unknown: the caller has to estimate, and the row
		// must advertise that it did.
		rec.Source = SourceEstimated
		rec.NeedsEstimate = true
	}
	return rec
}

// Totals is a rollup of cost and tokens. Every counter here is maintained
// incrementally on Append; nothing in this package rescans history to answer a
// metrics scrape, because the metrics endpoint is polled every fifteen seconds
// forever and an O(history) scrape is a self-inflicted outage in week three.
type Totals struct {
	// Entries is the number of rows, i.e. attempts, not requests.
	Entries int64 `json:"entries"`
	// Requests is the number of rows that were served to a client, which is the
	// count a tenant recognises as "my requests".
	Requests int64 `json:"requests"`
	// Retries is the number of rows that lost a failover race.
	Retries int64 `json:"retries"`
	// Failures is the number of rows whose request ultimately failed.
	Failures  int64 `json:"failures"`
	CacheHits int64 `json:"cache_hits"`
	Rejected  int64 `json:"rejected"`

	Tokens TokenCounts `json:"tokens"`

	Cost money.Pico `json:"cost_pico"`
	// ReportedCost and EstimatedCost partition Cost. They are tracked
	// separately, and not derived at read time, because "how much of this
	// invoice did we measure?" must be answerable in constant time — it is the
	// number that decides whether a reconciliation result means anything.
	ReportedCost  money.Pico `json:"reported_cost_pico"`
	EstimatedCost money.Pico `json:"estimated_cost_pico"`
	// EstimatedEntries is the row count behind EstimatedCost. A large cost from
	// one estimated row and a small cost from a thousand are different problems.
	EstimatedEntries int64 `json:"estimated_entries"`

	FirstAt time.Time `json:"first_at"`
	LastAt  time.Time `json:"last_at"`
}

// EstimatedShareBasisPoints is the fraction of Cost that came from estimated
// rows, in basis points (10000 = all of it). Basis points rather than a float64
// so that a monitoring threshold on this value compares exactly.
func (t Totals) EstimatedShareBasisPoints() int {
	if t.Cost <= 0 {
		return 0
	}
	return int(int64(t.EstimatedCost) * 10000 / int64(t.Cost))
}

func (t *Totals) add(e *Entry) {
	t.Entries++
	if e.ServedClient {
		t.Requests++
	}
	switch e.Status {
	case StatusFailedOver:
		t.Retries++
	case StatusFailed, StatusCancelled:
		t.Failures++
	case StatusCacheHit:
		t.CacheHits++
	case StatusRejected:
		t.Rejected++
	}
	t.Tokens.add(e.Tokens)
	t.Cost += e.CostPico
	if e.UsageSource == SourceEstimated {
		t.EstimatedCost += e.CostPico
		t.EstimatedEntries++
	} else {
		t.ReportedCost += e.CostPico
	}
	if t.FirstAt.IsZero() || e.RecordedAt.Before(t.FirstAt) {
		t.FirstAt = e.RecordedAt
	}
	if e.RecordedAt.After(t.LastAt) {
		t.LastAt = e.RecordedAt
	}
}

// TenantModel keys the per-tenant-per-model rollup.
type TenantModel struct {
	Tenant string `json:"tenant"`
	Model  string `json:"model"`
}

// Snapshot is a point-in-time copy of every rollup, safe to hand to a metrics
// encoder that will iterate it without holding the ledger's lock.
type Snapshot struct {
	Global        Totals                 `json:"global"`
	ByTenant      map[string]Totals      `json:"by_tenant"`
	ByTenantModel map[TenantModel]Totals `json:"by_tenant_model"`
	ByProvider    map[string]Totals      `json:"by_provider"`
	// Dropped counts entries that were valid but could not be persisted. It is
	// surfaced in the snapshot, and therefore in the metrics endpoint, because a
	// ledger silently failing to write is the one failure mode that makes every
	// other number here a lie.
	Dropped int64 `json:"dropped"`
	// Anomalies counts consistency violations detected at Append time (two rows
	// claiming to have served the same request, a duplicated attempt number).
	Anomalies int64 `json:"anomalies"`
}

// Options configures a Ledger.
type Options struct {
	// Now is the clock. Defaults to time.Now; injected so tests assert exact
	// timestamps instead of sleeping.
	Now func() time.Time

	// Buffered defers the write(2) syscall until Flush or Close.
	//
	// The durability ladder is worth being explicit about, because "flushed" is
	// used loosely and means three different things:
	//   - buffered:            lost if the process dies.
	//   - written (default):   survives process death, lost if the machine dies.
	//   - written + Fsync:     survives machine death, costs a disk round trip.
	// The default is the middle rung: one write(2) per entry into the page
	// cache, which is a microsecond and means a panic in the handler cannot lose
	// a cost row. Buffered mode is for the benchmark harness, which writes
	// hundreds of thousands of rows and does not care about a crash.
	Buffered bool
	// BufferBytes sizes the write buffer. Defaults to 64 KiB.
	BufferBytes int
	// Fsync forces durable-to-disk on every Append. Off by default: a gateway
	// that fsyncs on the request path has traded its p99 for a guarantee that
	// almost nobody's cost accounting actually requires.
	Fsync bool

	// Retain keeps the last N entries in memory for a debug endpoint. Zero (the
	// default) keeps none: an unbounded in-memory copy of the ledger is a leak
	// with a cost-accounting excuse.
	Retain int

	// GuardWindow is how many recent request IDs the Append-time consistency
	// check remembers. Defaults to 4096, and zero disables the check. The check
	// is necessarily windowed — remembering every request ID forever is the same
	// leak — so it catches the bug it is designed to catch (a handler marking
	// two attempts as served, or repeating an attempt number) while the request
	// is still recent. AuditRequests over the full JSONL is the unbounded
	// version of the same check.
	GuardWindow int
}

func (o Options) now() func() time.Time {
	if o.Now != nil {
		return o.Now
	}
	return time.Now
}

// syncer is the subset of *os.File the Fsync option needs. Declared here rather
// than taking an *os.File so tests can inject a failing sink.
type syncer interface{ Sync() error }

// ConsistencyError reports a violated cross-entry invariant.
//
// An Append that returns a *ConsistencyError HAS still written the entry
// durably. That is deliberate: the entry describes money that was really spent,
// and dropping a real cost row to punish the caller's bookkeeping mistake would
// make the ledger less accurate, not more. The error exists so the mistake is
// loud instead of invisible.
type ConsistencyError struct {
	Key    Key
	Detail string
}

func (e *ConsistencyError) Error() string {
	return fmt.Sprintf("ledger: consistency violation at %s: %s", e.Key, e.Detail)
}

// Ledger is a concurrency-safe append-only cost record with running aggregates.
type Ledger struct {
	now func() time.Time
	opt Options

	mu   sync.Mutex
	seq  int64
	w    *bufio.Writer
	sink io.Writer
	// closer is the underlying file, if this ledger owns one.
	closer io.Closer
	// scratch is reused across Appends to keep the hot path from allocating a
	// fresh encode buffer per request.
	scratch []byte
	closed  bool
	dropped int64
	anomaly int64

	global  Totals
	tenant  map[string]*Totals
	tenmod  map[TenantModel]*Totals
	byprov  map[string]*Totals
	guard   *requestGuard
	retain  []Entry
	retainN int
}

// New builds a Ledger writing JSONL to w. The caller keeps ownership of w; use
// Open when the ledger should own a file.
func New(w io.Writer, opt Options) *Ledger {
	if w == nil {
		w = io.Discard
	}
	bufSize := opt.BufferBytes
	if bufSize <= 0 {
		bufSize = 64 << 10
	}
	if opt.GuardWindow == 0 {
		opt.GuardWindow = 4096
	}
	l := &Ledger{
		now:     opt.now(),
		opt:     opt,
		w:       bufio.NewWriterSize(w, bufSize),
		sink:    w,
		scratch: make([]byte, 0, 1024),
		tenant:  make(map[string]*Totals),
		tenmod:  make(map[TenantModel]*Totals),
		byprov:  make(map[string]*Totals),
	}
	if opt.GuardWindow > 0 {
		l.guard = newRequestGuard(opt.GuardWindow)
	}
	if opt.Retain > 0 {
		l.retain = make([]Entry, 0, opt.Retain)
	}
	if c, ok := w.(io.Closer); ok {
		_ = c // ownership stays with the caller; see Open.
	}
	return l
}

// Open appends to the JSONL file at path, creating it if needed, and takes
// ownership of it: Close closes the file.
//
// O_APPEND is not decoration. It is what makes concurrent writers — a restarted
// process racing its predecessor's shutdown, or a sidecar — append whole lines
// atomically rather than interleaving halves of two rows, provided each write is
// a single line under the pipe-buffer size. Combined with the mutex below, a
// line in this file is never torn.
func Open(path string, opt Options) (*Ledger, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("ledger: open %s: %w", path, err)
	}
	l := New(f, opt)
	l.closer = f
	return l, nil
}

// Append validates, persists and aggregates one entry.
//
// It assigns Seq, RecordedAt, LatencyMS and CostUSD, so those fields are
// authoritative regardless of what the caller supplied. Ordering is FIFO: the
// encode and the buffered write happen under one mutex, so the sequence numbers
// in the file are strictly increasing and the buffer cannot reorder rows.
//
// A validation error means nothing was written. A *ConsistencyError means the
// row WAS written; see that type's documentation.
func (l *Ledger) Append(e Entry) error {
	if err := e.Validate(); err != nil {
		return err
	}
	now := l.now()

	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return errors.New("ledger: append after close")
	}
	l.seq++
	e.Seq = l.seq
	e.RecordedAt = now
	if e.StartedAt.IsZero() {
		e.StartedAt = now
	}
	if e.EndedAt.IsZero() {
		e.EndedAt = now
	}
	if e.LatencyMS == 0 && e.EndedAt.After(e.StartedAt) {
		e.LatencyMS = e.EndedAt.Sub(e.StartedAt).Milliseconds()
	}
	e.CostUSD = money.FormatUSD(e.CostPico)

	line, encErr := json.Marshal(&e)
	if encErr != nil {
		// Nothing sensible to persist and nothing was consumed upstream by this
		// failure, so refuse the row rather than corrupt the file.
		l.seq--
		l.mu.Unlock()
		return fmt.Errorf("ledger: encode entry %s: %w", e.Key(), encErr)
	}
	l.scratch = append(l.scratch[:0], line...)
	l.scratch = append(l.scratch, '\n')

	var writeErr error
	if _, err := l.w.Write(l.scratch); err != nil {
		writeErr = err
	} else if !l.opt.Buffered {
		if err := l.w.Flush(); err != nil {
			writeErr = err
		} else if l.opt.Fsync {
			if s, ok := l.sink.(syncer); ok {
				if err := s.Sync(); err != nil {
					writeErr = err
				}
			}
		}
	}
	if writeErr != nil {
		// The aggregates must describe what is in the file. Counting a row we
		// failed to persist would make the metrics endpoint disagree with the
		// artifact that gets reconciled, which is a worse failure than an
		// undercount we can see: Dropped is exported for exactly this.
		l.dropped++
		l.mu.Unlock()
		return fmt.Errorf("ledger: write entry %s: %w", e.Key(), writeErr)
	}

	l.aggregateLocked(&e)
	var cerr error
	if l.guard != nil {
		if detail := l.guard.observe(&e); detail != "" {
			l.anomaly++
			cerr = &ConsistencyError{Key: e.Key(), Detail: detail}
		}
	}
	if l.opt.Retain > 0 {
		if len(l.retain) < l.opt.Retain {
			l.retain = append(l.retain, e)
		} else {
			l.retain[l.retainN%l.opt.Retain] = e
		}
		l.retainN++
	}
	l.mu.Unlock()
	return cerr
}

func (l *Ledger) aggregateLocked(e *Entry) {
	l.global.add(e)
	bump(l.tenant, e.Tenant, e)
	bump(l.tenmod, TenantModel{Tenant: e.Tenant, Model: e.RequestedModel}, e)
	if e.Provider != "" {
		bump(l.byprov, e.Provider, e)
	}
}

func bump[K comparable](m map[K]*Totals, k K, e *Entry) {
	t, ok := m[k]
	if !ok {
		t = &Totals{}
		m[k] = t
	}
	t.add(e)
}

// Flush pushes buffered bytes to the underlying writer.
func (l *Ledger) Flush() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Flush()
}

// Sync flushes and, if the sink supports it, forces the bytes to stable storage.
func (l *Ledger) Sync() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.w.Flush(); err != nil {
		return err
	}
	if s, ok := l.sink.(syncer); ok {
		return s.Sync()
	}
	return nil
}

// Close flushes, syncs and closes an owned file. It is the durability contract:
// after Close returns nil, every entry Append accepted is on disk. Safe to call
// twice; the second call is a no-op.
func (l *Ledger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	err := l.w.Flush()
	if s, ok := l.sink.(syncer); ok {
		if serr := s.Sync(); err == nil {
			err = serr
		}
	}
	if l.closer != nil {
		if cerr := l.closer.Close(); err == nil {
			err = cerr
		}
	}
	return err
}

// Global returns the whole-ledger rollup.
func (l *Ledger) Global() Totals {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.global
}

// TenantTotals returns a tenant's rollup and whether the tenant has any rows.
func (l *Ledger) TenantTotals(tenant string) (Totals, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	t, ok := l.tenant[tenant]
	if !ok {
		return Totals{}, false
	}
	return *t, true
}

// TenantModelTotals returns a tenant's rollup for one requested model alias.
func (l *Ledger) TenantModelTotals(tenant, model string) (Totals, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	t, ok := l.tenmod[TenantModel{Tenant: tenant, Model: model}]
	if !ok {
		return Totals{}, false
	}
	return *t, true
}

// ProviderTotals returns a provider instance's rollup.
func (l *Ledger) ProviderTotals(name string) (Totals, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	t, ok := l.byprov[name]
	if !ok {
		return Totals{}, false
	}
	return *t, true
}

// Snapshot copies every rollup. The copy is why the metrics encoder can take its
// time formatting without holding up the request path.
func (l *Ledger) Snapshot() Snapshot {
	l.mu.Lock()
	defer l.mu.Unlock()
	s := Snapshot{
		Global:        l.global,
		ByTenant:      make(map[string]Totals, len(l.tenant)),
		ByTenantModel: make(map[TenantModel]Totals, len(l.tenmod)),
		ByProvider:    make(map[string]Totals, len(l.byprov)),
		Dropped:       l.dropped,
		Anomalies:     l.anomaly,
	}
	for k, v := range l.tenant {
		s.ByTenant[k] = *v
	}
	for k, v := range l.tenmod {
		s.ByTenantModel[k] = *v
	}
	for k, v := range l.byprov {
		s.ByProvider[k] = *v
	}
	return s
}

// Retained returns the retained recent entries in append order, or nil when the
// Retain option is off.
func (l *Ledger) Retained() []Entry {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := len(l.retain)
	if n == 0 {
		return nil
	}
	out := make([]Entry, 0, n)
	if l.retainN <= l.opt.Retain {
		return append(out, l.retain...)
	}
	start := l.retainN % l.opt.Retain
	out = append(out, l.retain[start:]...)
	return append(out, l.retain[:start]...)
}

// requestGuard is the windowed Append-time consistency check. See
// Options.GuardWindow for why it is windowed.
type requestGuard struct {
	limit int
	seen  map[string]*guardState
	ring  []string
	next  int
}

type guardState struct {
	served   bool
	attempts uint32
}

func newRequestGuard(limit int) *requestGuard {
	return &requestGuard{
		limit: limit,
		seen:  make(map[string]*guardState, limit),
		ring:  make([]string, 0, limit),
	}
}

// observe records an entry and returns a non-empty detail string when it
// violates a cross-entry invariant.
func (g *requestGuard) observe(e *Entry) string {
	st, ok := g.seen[e.RequestID]
	if !ok {
		st = &guardState{}
		if len(g.ring) < g.limit {
			g.ring = append(g.ring, e.RequestID)
		} else {
			delete(g.seen, g.ring[g.next%g.limit])
			g.ring[g.next%g.limit] = e.RequestID
		}
		g.next++
		g.seen[e.RequestID] = st
	}
	var detail string
	if e.Attempt >= 1 && e.Attempt <= 32 {
		bit := uint32(1) << uint(e.Attempt-1)
		if st.attempts&bit != 0 {
			detail = fmt.Sprintf("attempt %d already recorded for this request", e.Attempt)
		}
		st.attempts |= bit
	}
	if e.ServedClient {
		if st.served && detail == "" {
			detail = "a second attempt claims to have served the client; " +
				"a request has exactly one client-visible response"
		}
		st.served = true
	}
	return detail
}
