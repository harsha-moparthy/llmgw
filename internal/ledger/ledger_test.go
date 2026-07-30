package ledger

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/provider"
)

// fixedClock is a deterministic, monotonically advancing clock.
func fixedClock(start time.Time, step time.Duration) func() time.Time {
	var n int64
	return func() time.Time {
		i := atomic.AddInt64(&n, 1) - 1
		return start.Add(time.Duration(i) * step)
	}
}

var epoch = time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)

// costed builds a consistent entry whose breakdown sums to its cost.
func costed(reqID, tenant string, attempt int, prompt, completion int, perTok money.Pico) Entry {
	promptCost := perTok * money.Pico(prompt)
	compCost := perTok * 2 * money.Pico(completion)
	return Entry{
		RequestID:      reqID,
		Tenant:         tenant,
		RequestedModel: "gpt-4o-mini",
		UpstreamModel:  "gpt-4o-mini-2024-07-18",
		Provider:       "openai-primary",
		Attempt:        attempt,
		ServedClient:   true,
		Tokens:         TokenCounts{Prompt: prompt, Completion: completion},
		UsageSource:    SourceReported,
		CostPico:       promptCost + compCost,
		Breakdown: CostBreakdown{
			PromptPico:     promptCost,
			CompletionPico: compCost,
			Rates:          Rates{Prompt: perTok, Completion: perTok * 2, Version: "v1"},
		},
		Billable: true,
		Status:   StatusSucceeded,
	}
}

func TestEntryValidate(t *testing.T) {
	base := func() Entry { return costed("req-1", "acme", 1, 100, 50, 1000) }

	tests := []struct {
		name    string
		mutate  func(*Entry)
		wantErr string
	}{
		{"valid", func(*Entry) {}, ""},
		{"no request id", func(e *Entry) { e.RequestID = "" }, "no request id"},
		{"no tenant", func(e *Entry) { e.Tenant = "" }, "no tenant"},
		{"no model", func(e *Entry) { e.RequestedModel = "" }, "no requested model"},
		{"attempt zero", func(e *Entry) { e.Attempt = 0 }, "attempt must be >= 1"},
		{"attempt negative", func(e *Entry) { e.Attempt = -3 }, "attempt must be >= 1"},
		{"bad status", func(e *Entry) { e.Status = "weird" }, `unknown status "weird"`},
		{"bad source", func(e *Entry) { e.UsageSource = "guessed" }, `unknown usage source "guessed"`},
		{"negative tokens", func(e *Entry) { e.Tokens.Prompt = -1 }, "negative token count"},
		{
			// The containment rule: a provider log claiming more cached than
			// prompt tokens is describing something we must not bill.
			"cached exceeds prompt",
			func(e *Entry) { e.Tokens.Cached = e.Tokens.Prompt + 1 },
			"cached tokens (101) exceed prompt tokens (100)",
		},
		{
			"reasoning exceeds completion",
			func(e *Entry) { e.Tokens.Reasoning = e.Tokens.Completion + 1 },
			"reasoning tokens (51) exceed completion tokens (50)",
		},
		{"negative cost", func(e *Entry) { e.CostPico = -1; e.Breakdown.PromptPico = -1 - e.Breakdown.CompletionPico }, "negative cost"},
		{
			// The invariant that stops a total drifting from its parts.
			"breakdown does not sum",
			func(e *Entry) { e.Breakdown.PromptPico += 1 },
			"breakdown sums to 200001 pico but cost is 200000 pico (delta 1)",
		},
		{
			"source none with tokens",
			func(e *Entry) { e.UsageSource = SourceNone },
			`usage source "none" with non-zero tokens`,
		},
		{
			"cost with no tokens",
			func(e *Entry) { e.Tokens = TokenCounts{}; e.UsageSource = SourceNone },
			"cost 200000 pico attributed to zero tokens",
		},
		{
			"cache hit with cost",
			func(e *Entry) { e.CacheHit = true; e.Status = StatusCacheHit },
			"cache hit with non-zero upstream cost",
		},
		{
			"non-billable with cost",
			func(e *Entry) { e.Billable = false },
			"non-billable attempt with cost 200000 pico",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := base()
			tc.mutate(&e)
			err := e.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Validate() = nil, want error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Validate() = %q, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestValidateRejectionAndCacheHitRowsAreLegal(t *testing.T) {
	// Zero-cost, zero-token rows are the shape budget rejections and cache hits
	// take, and they must not trip the "cost with no tokens" check.
	for _, e := range []Entry{
		{
			RequestID: "r", Tenant: "t", RequestedModel: "m", Attempt: 1,
			Status: StatusRejected, UsageSource: SourceNone,
		},
		{
			RequestID: "r2", Tenant: "t", RequestedModel: "m", Attempt: 1,
			Status: StatusCacheHit, UsageSource: SourceNone, CacheHit: true,
			ServedClient: true,
		},
	} {
		if err := e.Validate(); err != nil {
			t.Fatalf("status %s: Validate() = %v, want nil", e.Status, err)
		}
	}
}

func TestAppendStampsDerivedFieldsAndPersistsJSONL(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, Options{Now: fixedClock(epoch, time.Second)})

	e := costed("req-1", "acme", 1, 100, 50, 1000)
	e.StartedAt = epoch.Add(-250 * time.Millisecond)
	e.EndedAt = epoch
	// Deliberately wrong: Append must overwrite, not trust.
	e.Seq = 999
	e.CostUSD = "$1000000"
	if err := l.Append(e); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := l.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	got, err := Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("decoded %d entries, want 1", len(got))
	}
	g := got[0]
	if g.Seq != 1 {
		t.Errorf("Seq = %d, want 1 (Append assigns it, the caller does not)", g.Seq)
	}
	if !g.RecordedAt.Equal(epoch) {
		t.Errorf("RecordedAt = %v, want %v from the injected clock", g.RecordedAt, epoch)
	}
	if g.LatencyMS != 250 {
		t.Errorf("LatencyMS = %d, want 250", g.LatencyMS)
	}
	// 100*1000 + 50*2000 = 200000 pico = $0.0000002
	if want := money.FormatUSD(200000); g.CostUSD != want {
		t.Errorf("CostUSD = %q, want %q recomputed from CostPico", g.CostUSD, want)
	}
	if g.CostPico != 200000 {
		t.Errorf("CostPico = %d, want 200000", g.CostPico)
	}
	if lines := bytes.Count(buf.Bytes(), []byte("\n")); lines != 1 {
		t.Errorf("wrote %d newlines, want exactly 1 per entry", lines)
	}
}

func TestAppendRejectsInvalidWithoutWriting(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, Options{Now: fixedClock(epoch, time.Second)})
	bad := costed("req-1", "acme", 1, 100, 50, 1000)
	bad.Tenant = ""
	if err := l.Append(bad); err == nil {
		t.Fatal("Append accepted an entry with no tenant")
	}
	if err := l.Flush(); err != nil {
		t.Fatal(err)
	}
	if buf.Len() != 0 {
		t.Fatalf("a rejected entry wrote %d bytes; nothing must be written", buf.Len())
	}
	if g := l.Global(); g.Entries != 0 {
		t.Fatalf("rejected entry counted in aggregates: %+v", g)
	}
}

// failingWriter fails after n successful writes.
type failingWriter struct {
	ok  int
	err error
	n   int
}

func (w *failingWriter) Write(p []byte) (int, error) {
	if w.n >= w.ok {
		return 0, w.err
	}
	w.n++
	return len(p), nil
}

func TestAppendWriteFailureIsCountedNotSilentlyAggregated(t *testing.T) {
	// A ledger whose writes fail must not report healthy aggregates: the
	// aggregates describe the artifact that gets reconciled, so a row that never
	// reached the file must not appear in them.
	sentinel := errors.New("disk full")
	fw := &failingWriter{ok: 1, err: sentinel}
	l := New(fw, Options{Now: fixedClock(epoch, time.Second), BufferBytes: 1})

	if err := l.Append(costed("req-1", "acme", 1, 10, 5, 1000)); err != nil {
		t.Fatalf("first Append: %v", err)
	}
	err := l.Append(costed("req-2", "acme", 1, 10, 5, 1000))
	if !errors.Is(err, sentinel) {
		t.Fatalf("Append error = %v, want it to wrap %v", err, sentinel)
	}
	snap := l.Snapshot()
	if snap.Global.Entries != 1 {
		t.Errorf("Entries = %d, want 1: the failed write must not aggregate", snap.Global.Entries)
	}
	if snap.Dropped != 1 {
		t.Errorf("Dropped = %d, want 1: a silently dropped row makes every other number a lie", snap.Dropped)
	}
}

func TestAppendAfterCloseFails(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, Options{Now: fixedClock(epoch, time.Second)})
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	if err := l.Close(); err != nil {
		t.Fatalf("second Close = %v, want nil (idempotent)", err)
	}
	if err := l.Append(costed("req-1", "acme", 1, 1, 1, 1000)); err == nil {
		t.Fatal("Append after Close succeeded")
	}
}

func TestOpenCloseIsDurable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cost.jsonl")

	// Buffered mode is the weakest rung of the durability ladder; Close must
	// still guarantee everything is on disk.
	l, err := Open(path, Options{Now: fixedClock(epoch, time.Second), Buffered: true, Fsync: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 50; i++ {
		if err := l.Append(costed(fmt.Sprintf("req-%d", i), "acme", 1, i, i, 1000)); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}
	// Nothing is guaranteed on disk yet in buffered mode, and that is the point
	// of the option; we only assert the post-Close guarantee.
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	entries, err := DecodeFile(path)
	if err != nil {
		t.Fatalf("DecodeFile: %v", err)
	}
	if len(entries) != 50 {
		t.Fatalf("read %d entries after Close, want 50", len(entries))
	}
	// Buffering must not reorder.
	for i, e := range entries {
		if e.Seq != int64(i+1) {
			t.Fatalf("entry %d has Seq %d; the buffer reordered rows", i, e.Seq)
		}
	}

	// Reopening appends rather than truncating: the record is append-only.
	l2, err := Open(path, Options{Now: fixedClock(epoch.Add(time.Hour), time.Second)})
	if err != nil {
		t.Fatal(err)
	}
	if err := l2.Append(costed("req-51", "acme", 1, 1, 1, 1000)); err != nil {
		t.Fatal(err)
	}
	if err := l2.Close(); err != nil {
		t.Fatal(err)
	}
	entries, err = DecodeFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 51 {
		t.Fatalf("after reopen+append the file has %d entries, want 51 (O_APPEND, not O_TRUNC)", len(entries))
	}
}

func TestUnbufferedAppendIsVisibleWithoutFlush(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cost.jsonl")
	l, err := Open(path, Options{Now: fixedClock(epoch, time.Second)})
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if err := l.Append(costed("req-1", "acme", 1, 3, 4, 1000)); err != nil {
		t.Fatal(err)
	}
	// The default mode's promise: a panic in the handler cannot lose this row.
	entries, err := DecodeFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("unbuffered Append left %d entries on disk before Flush, want 1", len(entries))
	}
}

func TestAggregatesMatchAFullRescan(t *testing.T) {
	// The whole point of the running totals is that they are not a rescan; this
	// test is the check that they nonetheless agree with one. If someone adds a
	// counter to Totals.add and forgets a case, this fails.
	var buf bytes.Buffer
	l := New(&buf, Options{Now: fixedClock(epoch, time.Second)})

	var appended []Entry
	add := func(e Entry) {
		if err := l.Append(e); err != nil {
			var ce *ConsistencyError
			if !errors.As(err, &ce) {
				t.Fatalf("Append: %v", err)
			}
		}
		appended = append(appended, e)
	}

	// acme: a success, a failover pair, a cache hit, a rejection.
	add(costed("r1", "acme", 1, 100, 50, 1000))

	failed := costed("r2", "acme", 1, 80, 10, 1000)
	failed.ServedClient = false
	failed.Status = StatusFailedOver
	failed.Provider = "openai-secondary"
	failed.UsageSource = SourceEstimated
	add(failed)
	retry := costed("r2", "acme", 2, 80, 40, 1000)
	add(retry)

	hit := Entry{
		RequestID: "r3", Tenant: "acme", RequestedModel: "gpt-4o-mini", Attempt: 1,
		Status: StatusCacheHit, UsageSource: SourceNone, CacheHit: true, ServedClient: true,
	}
	add(hit)

	rej := Entry{
		RequestID: "r4", Tenant: "acme", RequestedModel: "gpt-4o-mini", Attempt: 1,
		Status: StatusRejected, UsageSource: SourceNone,
	}
	add(rej)

	// A second tenant on a second model, one hard failure.
	other := costed("r5", "globex", 1, 500, 200, 2000)
	other.RequestedModel = "claude-3-5-sonnet"
	other.Provider = "anthropic-primary"
	add(other)
	hard := costed("r6", "globex", 1, 12, 0, 2000)
	hard.Tokens = TokenCounts{Prompt: 12}
	hard.CostPico = 12 * 2000
	hard.Breakdown = CostBreakdown{PromptPico: 12 * 2000, Rates: Rates{Prompt: 2000}}
	hard.Status = StatusFailed
	hard.ServedClient = false
	add(hard)

	// Rescan.
	want := Totals{}
	wantTenant := map[string]*Totals{}
	wantTM := map[TenantModel]*Totals{}
	wantProv := map[string]*Totals{}
	if err := l.Flush(); err != nil {
		t.Fatal(err)
	}
	decoded, err := Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != len(appended) {
		t.Fatalf("persisted %d rows, appended %d", len(decoded), len(appended))
	}
	for i := range decoded {
		e := &decoded[i]
		want.add(e)
		bump(wantTenant, e.Tenant, e)
		bump(wantTM, TenantModel{Tenant: e.Tenant, Model: e.RequestedModel}, e)
		if e.Provider != "" {
			bump(wantProv, e.Provider, e)
		}
	}

	snap := l.Snapshot()
	if snap.Global != want {
		t.Errorf("Global = %+v\nrescan  = %+v", snap.Global, want)
	}
	for k, v := range wantTenant {
		if got := snap.ByTenant[k]; got != *v {
			t.Errorf("tenant %s: got %+v want %+v", k, got, *v)
		}
	}
	for k, v := range wantTM {
		if got := snap.ByTenantModel[k]; got != *v {
			t.Errorf("tenant/model %v: got %+v want %+v", k, got, *v)
		}
	}
	for k, v := range wantProv {
		if got := snap.ByProvider[k]; got != *v {
			t.Errorf("provider %s: got %+v want %+v", k, got, *v)
		}
	}

	// Spot-check the classification counters against hand arithmetic, so a bug
	// mirrored in both add() paths cannot hide.
	acme, ok := l.TenantTotals("acme")
	if !ok {
		t.Fatal("no totals for acme")
	}
	// Five rows for four requests: r2 failed over, so it has two.
	if acme.Entries != 5 {
		t.Errorf("acme Entries = %d, want 5 (rows are attempts, not requests)", acme.Entries)
	}
	if acme.Requests != 3 {
		t.Errorf("acme Requests = %d, want 3 client-visible responses", acme.Requests)
	}
	if acme.Retries != 1 {
		t.Errorf("acme Retries = %d, want 1", acme.Retries)
	}
	if acme.CacheHits != 1 || acme.Rejected != 1 {
		t.Errorf("acme CacheHits=%d Rejected=%d, want 1 and 1", acme.CacheHits, acme.Rejected)
	}
	// r1 200000 + r2a1 (80*1000+10*2000=100000) + r2a2 (80*1000+40*2000=160000)
	if wantCost := money.Pico(200000 + 100000 + 160000); acme.Cost != wantCost {
		t.Errorf("acme Cost = %d, want %d", acme.Cost, wantCost)
	}
	if acme.EstimatedCost != 100000 || acme.EstimatedEntries != 1 {
		t.Errorf("acme estimated = %d pico over %d rows, want 100000 over 1",
			acme.EstimatedCost, acme.EstimatedEntries)
	}
	if acme.ReportedCost+acme.EstimatedCost != acme.Cost {
		t.Errorf("reported+estimated (%d) != cost (%d): the partition leaks",
			acme.ReportedCost+acme.EstimatedCost, acme.Cost)
	}
	// Tokens.Total deliberately excludes cached/reasoning; check the subsets are
	// not double counted.
	if acme.Tokens.Prompt != 100+80+80 {
		t.Errorf("acme prompt tokens = %d, want 260", acme.Tokens.Prompt)
	}

	if _, ok := l.TenantTotals("nobody"); ok {
		t.Error("TenantTotals reported a tenant with no rows")
	}
	if _, ok := l.ProviderTotals("openai-primary"); !ok {
		t.Error("no provider totals for openai-primary")
	}
	if _, ok := l.ProviderTotals("ghost"); ok {
		t.Error("ProviderTotals invented a provider")
	}
}

func TestEstimatedShareBasisPoints(t *testing.T) {
	tests := []struct {
		name string
		tot  Totals
		want int
	}{
		{"no cost", Totals{}, 0},
		{"all reported", Totals{Cost: 1000, ReportedCost: 1000}, 0},
		{"all estimated", Totals{Cost: 1000, EstimatedCost: 1000}, 10000},
		{"half", Totals{Cost: 1000, EstimatedCost: 500, ReportedCost: 500}, 5000},
		{"one part in ten thousand", Totals{Cost: 10000, EstimatedCost: 1}, 1},
		{"below basis point resolution truncates", Totals{Cost: 100000, EstimatedCost: 1}, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.tot.EstimatedShareBasisPoints(); got != tc.want {
				t.Fatalf("got %d, want %d", got, tc.want)
			}
		})
	}
}

func TestConsistencyGuardCatchesDoubleServeAndRepeatedAttempt(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, Options{Now: fixedClock(epoch, time.Second), GuardWindow: 8})

	if err := l.Append(costed("r1", "acme", 1, 10, 5, 1000)); err != nil {
		t.Fatalf("first Append: %v", err)
	}
	second := costed("r1", "acme", 2, 10, 5, 1000) // also ServedClient
	err := l.Append(second)
	var ce *ConsistencyError
	if !errors.As(err, &ce) {
		t.Fatalf("Append error = %v, want *ConsistencyError for a second served row", err)
	}
	if !strings.Contains(ce.Detail, "served the client") {
		t.Errorf("detail = %q, want it to mention serving the client", ce.Detail)
	}

	// Critically: the row was still persisted. It describes money really spent,
	// and dropping it to punish the caller would make the ledger less accurate.
	if err := l.Flush(); err != nil {
		t.Fatal(err)
	}
	entries, err := Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("persisted %d rows, want 2: a ConsistencyError must not discard the row", len(entries))
	}
	if snap := l.Snapshot(); snap.Anomalies != 1 {
		t.Errorf("Anomalies = %d, want 1", snap.Anomalies)
	}

	// A repeated attempt number on a fresh request.
	if err := l.Append(costed("r2", "acme", 1, 10, 5, 1000)); err != nil {
		t.Fatal(err)
	}
	dupAttempt := costed("r2", "acme", 1, 10, 5, 1000)
	dupAttempt.ServedClient = false
	err = l.Append(dupAttempt)
	if !errors.As(err, &ce) {
		t.Fatalf("Append error = %v, want *ConsistencyError for a repeated attempt", err)
	}
	if !strings.Contains(ce.Detail, "attempt 1 already recorded") {
		t.Errorf("detail = %q, want it to mention the repeated attempt", ce.Detail)
	}
}

func TestConsistencyGuardWindowEvictsAndDoesNotGrow(t *testing.T) {
	var buf bytes.Buffer
	const window = 4
	l := New(&buf, Options{Now: fixedClock(epoch, time.Millisecond), GuardWindow: window})

	for i := 0; i < 100; i++ {
		if err := l.Append(costed(fmt.Sprintf("r%d", i), "acme", 1, 1, 1, 1000)); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}
	l.mu.Lock()
	seen, ring := len(l.guard.seen), len(l.guard.ring)
	l.mu.Unlock()
	if seen > window || ring > window {
		t.Fatalf("guard holds %d ids / ring %d after 100 appends; window is %d "+
			"(an unbounded guard is a leak with a bookkeeping excuse)", seen, ring, window)
	}

	// A duplicate that has fallen out of the window is not reported: that is the
	// documented, deliberate limitation, and AuditRequests is the unbounded check.
	dup := costed("r0", "acme", 1, 1, 1, 1000)
	if err := l.Append(dup); err != nil {
		t.Fatalf("Append of an evicted duplicate = %v, want nil (out of window)", err)
	}
}

func TestGuardWindowDisabled(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, Options{Now: fixedClock(epoch, time.Second), GuardWindow: -1})
	if err := l.Append(costed("r1", "acme", 1, 1, 1, 1000)); err != nil {
		t.Fatal(err)
	}
	if err := l.Append(costed("r1", "acme", 1, 1, 1, 1000)); err != nil {
		t.Fatalf("GuardWindow<0 should disable the check, got %v", err)
	}
}

func TestRetainRingIsBoundedAndOrdered(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, Options{Now: fixedClock(epoch, time.Second), Retain: 3, GuardWindow: -1})

	if got := l.Retained(); got != nil {
		t.Fatalf("Retained() on an empty ledger = %v, want nil", got)
	}
	for i := 1; i <= 7; i++ {
		if err := l.Append(costed(fmt.Sprintf("r%d", i), "acme", 1, i, i, 1000)); err != nil {
			t.Fatal(err)
		}
		got := l.Retained()
		wantLen := i
		if wantLen > 3 {
			wantLen = 3
		}
		if len(got) != wantLen {
			t.Fatalf("after %d appends Retained() has %d, want %d", i, len(got), wantLen)
		}
		// Must be the most recent, in append order.
		for j, e := range got {
			wantSeq := int64(i - wantLen + j + 1)
			if e.Seq != wantSeq {
				t.Fatalf("after %d appends Retained()[%d].Seq = %d, want %d", i, j, e.Seq, wantSeq)
			}
		}
	}
}

func TestRetainOffKeepsNothing(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, Options{Now: fixedClock(epoch, time.Second), GuardWindow: -1})
	for i := 0; i < 10; i++ {
		if err := l.Append(costed(fmt.Sprintf("r%d", i), "acme", 1, 1, 1, 1000)); err != nil {
			t.Fatal(err)
		}
	}
	if got := l.Retained(); got != nil {
		t.Fatalf("Retained() = %d entries with Retain unset; the default must keep none", len(got))
	}
}

// syncCountingBuffer records Sync calls so the Fsync option is observably real.
type syncCountingBuffer struct {
	mu    sync.Mutex
	buf   bytes.Buffer
	syncs int
}

func (s *syncCountingBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.Write(p)
}

func (s *syncCountingBuffer) Sync() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.syncs++
	return nil
}

func TestFsyncOptionActuallySyncs(t *testing.T) {
	// Without this test the Fsync option could be a no-op forever and nothing
	// would notice: the data still lands, just less durably.
	s := &syncCountingBuffer{}
	l := New(s, Options{Now: fixedClock(epoch, time.Second), Fsync: true, GuardWindow: -1})
	for i := 0; i < 3; i++ {
		if err := l.Append(costed(fmt.Sprintf("r%d", i), "acme", 1, 1, 1, 1000)); err != nil {
			t.Fatal(err)
		}
	}
	s.mu.Lock()
	got := s.syncs
	s.mu.Unlock()
	if got != 3 {
		t.Fatalf("Sync called %d times for 3 appends with Fsync on, want 3", got)
	}

	// And with Fsync off it must not: an fsync per request is a p99 disaster.
	s2 := &syncCountingBuffer{}
	l2 := New(s2, Options{Now: fixedClock(epoch, time.Second), GuardWindow: -1})
	if err := l2.Append(costed("r", "acme", 1, 1, 1, 1000)); err != nil {
		t.Fatal(err)
	}
	s2.mu.Lock()
	got2 := s2.syncs
	s2.mu.Unlock()
	if got2 != 0 {
		t.Fatalf("Sync called %d times with Fsync off, want 0", got2)
	}
	// Explicit Sync() still works.
	if err := l2.Sync(); err != nil {
		t.Fatal(err)
	}
	s2.mu.Lock()
	got2 = s2.syncs
	s2.mu.Unlock()
	if got2 != 1 {
		t.Fatalf("after explicit Sync(), syncs = %d, want 1", got2)
	}
}

func TestAppendConcurrentLosesNothingAndDoesNotTearLines(t *testing.T) {
	// Run with -race. The claims: every entry lands, sequence numbers are a
	// permutation of 1..N with no gaps or repeats, no line is torn, and the
	// aggregates equal the file.
	const writers = 32
	const perWriter = 100
	const total = writers * perWriter

	dir := t.TempDir()
	path := filepath.Join(dir, "cost.jsonl")
	l, err := Open(path, Options{Now: fixedClock(epoch, time.Microsecond), GuardWindow: 512})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	var appendErrs atomic.Int64
	// A concurrent reader hammers Snapshot to prove the read path is safe too.
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = l.Snapshot()
				_, _ = l.TenantTotals("t0")
				_ = l.Global()
			}
		}
	}()
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				e := costed(fmt.Sprintf("r-%d-%d", w, i), fmt.Sprintf("t%d", w%4), 1, 10, 5, 1000)
				if err := l.Append(e); err != nil {
					appendErrs.Add(1)
				}
			}
		}(w)
	}
	// Wait for writers before stopping the reader.
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	time.Sleep(time.Millisecond)
	close(stop)
	<-done

	if n := appendErrs.Load(); n != 0 {
		t.Fatalf("%d Appends failed under concurrency", n)
	}
	snapshot := l.Snapshot()
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}

	entries, err := DecodeFile(path)
	if err != nil {
		t.Fatalf("DecodeFile (a torn line decodes as an error): %v", err)
	}
	if len(entries) != total {
		t.Fatalf("persisted %d entries, want %d: Append lost rows", len(entries), total)
	}
	seen := make([]bool, total+1)
	for _, e := range entries {
		if e.Seq < 1 || e.Seq > total {
			t.Fatalf("Seq %d out of range", e.Seq)
		}
		if seen[e.Seq] {
			t.Fatalf("Seq %d assigned twice", e.Seq)
		}
		seen[e.Seq] = true
	}
	// Buffered=false plus a mutex means the file order is the sequence order.
	for i, e := range entries {
		if e.Seq != int64(i+1) {
			t.Fatalf("file line %d has Seq %d; writes were reordered", i+1, e.Seq)
		}
	}
	if snapshot.Global.Entries != total {
		t.Fatalf("aggregate Entries = %d, want %d", snapshot.Global.Entries, total)
	}
	// Each row is 10 prompt tokens at 1000 pico and 5 completion at 2000 pico.
	if want := money.Pico(total) * (10*1000 + 5*2000); snapshot.Global.Cost != want {
		t.Fatalf("aggregate Cost = %d, want %d", snapshot.Global.Cost, want)
	}
}

func TestClassifyFailure(t *testing.T) {
	usage := func(p, c int) *apiv1.Usage {
		return &apiv1.Usage{PromptTokens: p, CompletionTokens: c, TotalTokens: p + c}
	}
	tests := []struct {
		name       string
		failure    *provider.Failure
		failedOver bool
		want       AttemptRecord
	}{
		{
			// A connection that never established cannot have been billed, so no
			// cost row is expected and the reconciliation must not look for one.
			name:    "connect refused is free",
			failure: &provider.Failure{Class: provider.ClassConnect},
			want:    AttemptRecord{Status: StatusFailed, Source: SourceNone},
		},
		{
			name:    "rate limit is free",
			failure: &provider.Failure{Class: provider.ClassRateLimit, StatusCode: 429},
			want:    AttemptRecord{Status: StatusFailed, Source: SourceNone},
		},
		{
			name:    "bad request is free",
			failure: &provider.Failure{Class: provider.ClassBadRequest, StatusCode: 400},
			want:    AttemptRecord{Status: StatusFailed, Source: SourceNone},
		},
		{
			// The row that would otherwise be recorded as free: a timeout after
			// generation began is billed by real providers, so the caller MUST
			// estimate. NeedsEstimate is the flag that forces it.
			name:    "timeout with no usage needs an estimate",
			failure: &provider.Failure{Class: provider.ClassTimeout},
			want: AttemptRecord{
				Status: StatusFailed, Billable: true,
				Source: SourceEstimated, NeedsEstimate: true,
			},
		},
		{
			name:    "5xx with no usage needs an estimate",
			failure: &provider.Failure{Class: provider.ClassUpstream5xx, StatusCode: 503},
			want: AttemptRecord{
				Status: StatusFailed, Billable: true,
				Source: SourceEstimated, NeedsEstimate: true,
			},
		},
		{
			// Reported usage beats every heuristic: it is a measurement.
			name:    "mid-stream failure with reported usage",
			failure: &provider.Failure{Class: provider.ClassTimeout, UsageAtFailure: usage(120, 400)},
			want: AttemptRecord{
				Status: StatusFailed, Billable: true, Source: SourceReported,
				Tokens: TokenCounts{Prompt: 120, Completion: 400},
			},
		},
		{
			// An explicit zero from the provider is a measurement of nothing, not
			// a reason to estimate.
			name:    "reported zero usage is not an estimate",
			failure: &provider.Failure{Class: provider.ClassTimeout, UsageAtFailure: usage(0, 0)},
			want:    AttemptRecord{Status: StatusFailed, Source: SourceNone},
		},
		{
			name:    "cancelled client is billed",
			failure: &provider.Failure{Class: provider.ClassCancelled},
			want: AttemptRecord{
				Status: StatusCancelled, Billable: true,
				Source: SourceEstimated, NeedsEstimate: true,
			},
		},
		{
			// The failover flag wins over the cancelled special case: the status
			// records what happened to *this attempt* in the request's story.
			name:       "cancelled but failed over",
			failure:    &provider.Failure{Class: provider.ClassCancelled},
			failedOver: true,
			want: AttemptRecord{
				Status: StatusFailedOver, Billable: true,
				Source: SourceEstimated, NeedsEstimate: true,
			},
		},
		{
			name:       "failover keeps the retry status",
			failure:    &provider.Failure{Class: provider.ClassUpstream5xx, StatusCode: 500},
			failedOver: true,
			want: AttemptRecord{
				Status: StatusFailedOver, Billable: true,
				Source: SourceEstimated, NeedsEstimate: true,
			},
		},
		{
			name:    "nil failure",
			failure: nil,
			want:    AttemptRecord{Status: StatusFailed, Source: SourceNone},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyFailure(tc.failure, tc.failedOver)
			if got != tc.want {
				t.Fatalf("got %+v\nwant %+v", got, tc.want)
			}
			// Cross-check against the frozen contract rather than restating it.
			if tc.failure != nil && tc.failure.UsageAtFailure == nil {
				if got.Billable != tc.failure.MayHaveBilled() {
					t.Errorf("Billable = %t but MayHaveBilled() = %t; the ledger must "+
						"not second-guess the provider contract",
						got.Billable, tc.failure.MayHaveBilled())
				}
			}
			// NeedsEstimate implies the row will claim SourceEstimated.
			if got.NeedsEstimate && got.Source != SourceEstimated {
				t.Errorf("NeedsEstimate with source %q", got.Source)
			}
		})
	}
}

func TestFromUsage(t *testing.T) {
	tests := []struct {
		name string
		u    *apiv1.Usage
		want TokenCounts
	}{
		{"nil", nil, TokenCounts{}},
		{"plain", &apiv1.Usage{PromptTokens: 10, CompletionTokens: 3}, TokenCounts{Prompt: 10, Completion: 3}},
		{
			"with details",
			&apiv1.Usage{
				PromptTokens:            100,
				CompletionTokens:        50,
				PromptTokensDetails:     &apiv1.PromptTokensDetails{CachedTokens: 64},
				CompletionTokensDetails: &apiv1.CompletionTokensDetails{ReasoningTokens: 30},
			},
			TokenCounts{Prompt: 100, Cached: 64, Completion: 50, Reasoning: 30},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := FromUsage(tc.u); got != tc.want {
				t.Fatalf("got %+v want %+v", got, tc.want)
			}
		})
	}
}

func TestTokenCountsArithmetic(t *testing.T) {
	tc := TokenCounts{Prompt: 100, Cached: 60, Completion: 40, Reasoning: 25}
	// Total must NOT add cached or reasoning: they are subsets already counted.
	if got := tc.Total(); got != 140 {
		t.Errorf("Total() = %d, want 140 (cached and reasoning are subsets)", got)
	}
	if got := tc.BillablePrompt(); got != 40 {
		t.Errorf("BillablePrompt() = %d, want 40", got)
	}
	if tc.IsZero() {
		t.Error("IsZero() true for a populated count")
	}
	if !(TokenCounts{}).IsZero() {
		t.Error("IsZero() false for the zero value")
	}
}

func TestUsageSourceAndStatusValidity(t *testing.T) {
	for _, s := range []UsageSource{SourceReported, SourceEstimated, SourceNone} {
		if !s.Valid() {
			t.Errorf("%q reported invalid", s)
		}
	}
	for _, s := range []UsageSource{"", "guess", "REPORTED"} {
		if s.Valid() {
			t.Errorf("%q reported valid", s)
		}
	}
	for _, s := range []Status{
		StatusSucceeded, StatusFailedOver, StatusFailed,
		StatusCancelled, StatusCacheHit, StatusRejected,
	} {
		if !s.Valid() {
			t.Errorf("%q reported invalid", s)
		}
	}
	for _, s := range []Status{"", "ok", "SUCCEEDED"} {
		if s.Valid() {
			t.Errorf("%q reported valid", s)
		}
	}
}

func TestEntryJSONRoundTripPreservesRatesAndSource(t *testing.T) {
	// The persisted row must carry the rates that were applied, so a row remains
	// auditable after the pricing table moves on.
	e := costed("r1", "acme", 1, 100, 50, 1500)
	e.Tokens.Cached = 40
	e.UsageSource = SourceEstimated
	e.Breakdown.Rates.Version = "2026-03-01"
	e.Breakdown.CachedPico = 0

	b, err := json.Marshal(&e)
	if err != nil {
		t.Fatal(err)
	}
	var back Entry
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.Breakdown.Rates != e.Breakdown.Rates {
		t.Errorf("rates lost in the round trip: %+v vs %+v", back.Breakdown.Rates, e.Breakdown.Rates)
	}
	if back.UsageSource != SourceEstimated {
		t.Errorf("usage source = %q, want %q", back.UsageSource, SourceEstimated)
	}
	// The estimated marker must be legible to a human grepping the file, which is
	// why UsageSource is a string and not an int.
	if !bytes.Contains(b, []byte(`"usage_source":"estimated"`)) {
		t.Errorf("serialised row does not spell out the estimate: %s", b)
	}
}

func TestDecodeIsStrictAboutMalformedLines(t *testing.T) {
	// Skipping unparseable lines would let a reconciliation report a clean match
	// over whichever subset happened to decode.
	good := `{"seq":1,"request_id":"r1","tenant":"t","requested_model":"m","attempt":1,"status":"succeeded","usage_source":"reported"}`
	tests := []struct {
		name    string
		input   string
		wantN   int
		wantErr string
	}{
		{"empty", "", 0, ""},
		{"one line", good + "\n", 1, ""},
		{"no trailing newline", good, 1, ""},
		{"blank lines skipped", good + "\n\n   \n" + good + "\n", 2, ""},
		{"truncated line", good + "\n{\"seq\":2,\"reque", 1, "decode line 2"},
		{"garbage", "not json\n", 0, "decode line 1"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Decode(strings.NewReader(tc.input))
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("Decode = %v, want nil", err)
				}
			} else {
				if err == nil {
					t.Fatalf("Decode = nil, want error containing %q", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("Decode = %q, want it to contain %q", err, tc.wantErr)
				}
			}
			if len(got) != tc.wantN {
				t.Fatalf("decoded %d entries, want %d", len(got), tc.wantN)
			}
		})
	}
}

func TestDecodeFileMissing(t *testing.T) {
	if _, err := DecodeFile(filepath.Join(t.TempDir(), "nope.jsonl")); err == nil {
		t.Fatal("DecodeFile on a missing path returned nil error")
	} else if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("error = %v, want it to wrap os.ErrNotExist", err)
	}
}

func TestNewWithNilWriterDoesNotPanic(t *testing.T) {
	l := New(nil, Options{Now: fixedClock(epoch, time.Second)})
	if err := l.Append(costed("r1", "acme", 1, 1, 1, 1000)); err != nil {
		t.Fatal(err)
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	if l.Global().Entries != 1 {
		t.Fatal("aggregates should still work against a discard sink")
	}
}

func BenchmarkAppend(b *testing.B) {
	l := New(io.Discard, Options{Buffered: true, GuardWindow: -1})
	e := costed("r", "acme", 1, 500, 250, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := l.Append(e); err != nil {
			b.Fatal(err)
		}
	}
}
