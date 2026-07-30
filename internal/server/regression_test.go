package server

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/config"
	"github.com/harsha-moparthy/llmgw/internal/ledger"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/pricing"
	"github.com/harsha-moparthy/llmgw/internal/provider"
	"github.com/harsha-moparthy/llmgw/internal/router"
)

// This file pins the two accounting bugs that the LIVE failover reconciliation
// found and that no unit test had covered. Both were invisible to the suite: the
// first only manifests when a provider actually fails over under load, and the
// second only when a request is cancelled after the upstream began generating.
// Fixes without regression tests silently rot, and these two are the ones that
// move money.

func testPrices(t *testing.T) *pricing.Table {
	t.Helper()
	tbl, err := (&pricing.Sheet{Models: []pricing.ModelPrice{
		{Model: "mock-fast", InputPerMTok: "0.15", CachedInputPerMTok: "0.075", OutputPerMTok: "0.60"},
	}}).Table()
	if err != nil {
		t.Fatal(err)
	}
	return tbl
}

// TestBillableCancelledAttemptIsEstimatedNotFree is the regression test for the
// under-billing bug.
//
// A request cancelled after the provider began generating IS charged by the
// provider. Recording zero tokens for it makes the gateway's total drift below
// the invoice exactly when things go wrong — the most expensive direction to be
// wrong in. The ledger's ClassifyFailure flags this case via NeedsEstimate; the
// server has to honour it.
func TestBillableCancelledAttemptIsEstimatedNotFree(t *testing.T) {
	s := &Server{
		deps: Deps{Prices: testPrices(t)},
		now:  time.Now,
	}

	// A cancelled attempt with NO usage reported: the provider told us nothing,
	// but ClassCancelled means generation may well have happened and been billed.
	outcome := router.AttemptOutcome{
		Tenant:        "acme",
		Alias:         "gw-chat",
		Provider:      "mock-primary",
		UpstreamModel: "mock-fast",
		Attempt:       1,
		Failure:       &provider.Failure{Class: provider.ClassCancelled},
		Usage:         nil,
	}

	// Confirm the premise: the ledger's own policy says this needs an estimate.
	// If that contract ever changes, this test should fail loudly rather than
	// silently prove nothing.
	rec := ledger.ClassifyFailure(outcome.Failure, false)
	if !rec.NeedsEstimate {
		t.Fatalf("precondition: ClassifyFailure no longer flags a cancelled attempt as NeedsEstimate (%+v)", rec)
	}

	// Confirm ledgerFromOutcome alone produces a zero-cost row: the estimate is
	// applied on top, and that layering is what the wiring test below pins.
	bare := ledgerFromOutcome(outcome, "gw-1", "gw-chat", s.deps.Prices, s.now())
	if bare.CostPico != 0 {
		t.Fatalf("precondition: expected ledgerFromOutcome to leave cost at 0 for a usage-less failure, got %d", bare.CostPico)
	}

	// Drive the REAL path — flushLedger — rather than calling the helper
	// directly. A test that only exercises applyEstimateIfBilled passes even if
	// nothing ever calls it, which is exactly how a fix silently regresses.
	// Verified: removing the applyEstimateIfBilled call from flushLedger makes
	// this test fail.
	buf := &syncBuffer{}
	lg := ledger.New(buf, ledger.Options{})
	s.deps.Ledger = lg
	collector := &attemptCollector{}
	collector.RecordAttempt(outcome)
	s.flushLedger(collector, "gw-1", "gw-chat", tokenEstimate{prompt: 100, max: 50}, s.now())
	if err := lg.Flush(); err != nil {
		t.Fatal(err)
	}
	written := buf.entries(t)
	if len(written) != 1 {
		t.Fatalf("expected exactly 1 ledger row, got %d", len(written))
	}
	entry := written[0]

	if entry.CostPico == 0 {
		t.Error("a billable cancelled attempt was recorded as FREE; the provider charged for those tokens")
	}
	if entry.UsageSource != ledger.SourceEstimated {
		t.Errorf("usage_source = %q, want %q: a bill must never present an estimate as a measurement",
			entry.UsageSource, ledger.SourceEstimated)
	}
	if entry.Tokens.Prompt != 100 || entry.Tokens.Completion != 50 {
		t.Errorf("tokens = %+v, want prompt=100 completion=50 from the estimate", entry.Tokens)
	}
	if !entry.Billable {
		t.Error("entry with a non-zero estimated cost must be marked billable")
	}
	// The ledger enforces that the breakdown sums to the total; an entry that
	// violates it is rejected on Append, which would silently drop a cost row.
	if got := entry.Breakdown.Total(); got != entry.CostPico {
		t.Errorf("breakdown sums to %d but cost is %d; Append would reject this row", got, entry.CostPico)
	}
	if err := entry.Validate(); err != nil {
		t.Errorf("estimated row does not pass ledger validation: %v", err)
	}
}

// TestNonBillableFailureStaysFree is the other half of the control. Without it,
// the test above could be satisfied by an implementation that slaps an estimate
// on EVERY failure — over-billing for requests the provider never even started.
func TestNonBillableFailureStaysFree(t *testing.T) {
	s := &Server{deps: Deps{Prices: testPrices(t)}, now: time.Now}

	for _, class := range []provider.Class{
		provider.ClassConnect,   // never established: nothing was generated
		provider.ClassRateLimit, // rejected before generation
		provider.ClassAuth,      // rejected before generation
		provider.ClassBadRequest,
	} {
		t.Run(class.String(), func(t *testing.T) {
			outcome := router.AttemptOutcome{
				Tenant: "acme", Alias: "gw-chat", Provider: "p", UpstreamModel: "mock-fast",
				Attempt: 1, Failure: &provider.Failure{Class: class},
			}
			entry := ledgerFromOutcome(outcome, "gw-2", "gw-chat", s.deps.Prices, s.now())
			s.applyEstimateIfBilled(&entry, outcome, tokenEstimate{prompt: 100, max: 50})

			if entry.CostPico != 0 {
				t.Errorf("%s attempt charged %d pico; the provider never generated anything",
					class, entry.CostPico)
			}
			if entry.UsageSource != ledger.SourceNone {
				t.Errorf("%s attempt usage_source = %q, want %q", class, entry.UsageSource, ledger.SourceNone)
			}
			if entry.Billable {
				t.Errorf("%s attempt marked billable", class)
			}
		})
	}
}

// TestReportedUsageIsNeverOverwrittenByAnEstimate: when the provider DID report
// usage at failure time, that measurement must win. Replacing it with an
// estimate would throw away a real number in favour of a guess.
func TestReportedUsageIsNeverOverwrittenByAnEstimate(t *testing.T) {
	s := &Server{deps: Deps{Prices: testPrices(t)}, now: time.Now}
	realUsage := &apiv1.Usage{PromptTokens: 37, CompletionTokens: 48, TotalTokens: 85}
	outcome := router.AttemptOutcome{
		Tenant: "acme", Alias: "gw-chat", Provider: "p", UpstreamModel: "mock-fast",
		Attempt: 1,
		Failure: &provider.Failure{Class: provider.ClassTimeout, UsageAtFailure: realUsage},
		Usage:   realUsage,
	}
	entry := ledgerFromOutcome(outcome, "gw-3", "gw-chat", s.deps.Prices, s.now())
	before := entry.CostPico
	s.applyEstimateIfBilled(&entry, outcome, tokenEstimate{prompt: 9999, max: 9999})

	if entry.CostPico != before {
		t.Errorf("reported usage was overwritten by an estimate: cost %d -> %d", before, entry.CostPico)
	}
	if entry.UsageSource != ledger.SourceReported {
		t.Errorf("usage_source = %q, want %q", entry.UsageSource, ledger.SourceReported)
	}
	if entry.Tokens.Prompt != 37 || entry.Tokens.Completion != 48 {
		t.Errorf("tokens = %+v, want the provider-reported 37/48", entry.Tokens)
	}
}

// TestBilledFailureChargesTheBudget pins a revenue leak found by audit.
//
// The failure path used to call Budget.Release(), freeing the entire hold. But a
// failure is not automatically free: a 5xx after generation began is billed by
// the provider, and the ledger records that cost. Releasing the hold meant the
// ledger showed spend the budget never saw — so a tenant could burn unlimited
// budget through a failing upstream and never be rejected.
//
// Verified: swapping the Commit(billedCost) back to Release() makes this fail.
func TestBilledFailureChargesTheBudget(t *testing.T) {
	tenant := config.Tenant{
		ID: "acme", APIKeyHash: config.HashKey("acme-key"),
		BudgetLimit: "1.00", BudgetPeriod: "hour",
		AllowedModels: []string{"gw-chat"},
	}
	h := newHarness(t, 1, []config.Tenant{tenant})

	// A 500 is health-counting and MayHaveBilled()==true: the provider began
	// generating and charged us, but reports no usage, so the row is estimated.
	h.mocks[0].Faults().Set("status_code", "500")

	const n = 25
	for i := 0; i < n; i++ {
		resp := h.post("acme-key", chatBody("gw-chat", fmt.Sprintf("billed failure %d", i), false))
		resp.Body.Close()
	}
	if err := h.ledger.Flush(); err != nil {
		t.Fatal(err)
	}

	var ledgered money.Pico
	for _, e := range h.ledgerBuf.entries(t) {
		ledgered += e.CostPico
	}
	spent := h.server.deps.Budget.Status("acme").Spent

	if ledgered == 0 {
		t.Fatal("precondition: the ledger recorded no billable cost, so this test proves nothing")
	}
	if spent == 0 {
		t.Fatalf("the ledger recorded %s of billable spend but the budget shows %s: "+
			"a tenant can burn unlimited budget through a failing upstream",
			money.FormatUSD(ledgered), money.FormatUSD(spent))
	}
	// The two must agree: the budget is charged exactly what the ledger attributed.
	if spent != ledgered {
		t.Errorf("budget spent %s but the ledger recorded %s; the two accounts disagree",
			money.FormatUSD(spent), money.FormatUSD(ledgered))
	}
}

// TestUnbilledFailureDoesNotChargeTheBudget is the control: a failure the
// provider could not have billed (connection refused) must leave the budget
// untouched, or the fix above would over-charge every transient blip.
func TestUnbilledFailureDoesNotChargeTheBudget(t *testing.T) {
	tenant := config.Tenant{
		ID: "acme", APIKeyHash: config.HashKey("acme-key"),
		BudgetLimit: "1.00", BudgetPeriod: "hour",
		AllowedModels: []string{"gw-chat"},
	}
	h := newHarness(t, 1, []config.Tenant{tenant})
	// A rate limit is rejected before any generation: nothing was billed.
	h.mocks[0].Faults().Set("status_code", "429")

	for i := 0; i < 10; i++ {
		resp := h.post("acme-key", chatBody("gw-chat", fmt.Sprintf("unbilled %d", i), false))
		resp.Body.Close()
	}
	if err := h.ledger.Flush(); err != nil {
		t.Fatal(err)
	}
	if spent := h.server.deps.Budget.Status("acme").Spent; spent != 0 {
		t.Errorf("budget charged %s for rate-limited requests the provider never generated",
			money.FormatUSD(spent))
	}
}

// TestToolCallStreamReachesClient pins a critical bug found by audit: the sink
// gated "is this content?" on delta text alone, so a tool-call stream — whose
// entire payload is delta.tool_calls with NO text — never started the response
// and the client received only "data: [DONE]". An empty-completion stream lost
// its finish_reason and usage frames the same way.
func TestChunkForwardableCoversNonTextPayloads(t *testing.T) {
	stop := apiv1.FinishToolCalls
	tests := []struct {
		name  string
		chunk *apiv1.ChatChunk
		want  bool
	}{
		{"nil chunk", nil, false},
		{
			// The role-only opening frame must NOT count: excluding it is what
			// holds the transparent-failover window open.
			name:  "role only",
			chunk: &apiv1.ChatChunk{Choices: []apiv1.Choice{{Delta: &apiv1.Message{Role: apiv1.RoleAssistant}}}},
			want:  false,
		},
		{
			name:  "delta text",
			chunk: &apiv1.ChatChunk{Choices: []apiv1.Choice{{Delta: &apiv1.Message{Content: apiv1.NewTextContent("hi")}}}},
			want:  true,
		},
		{
			// The bug: a tool-call chunk carries no text at all.
			name: "tool calls with no text",
			chunk: &apiv1.ChatChunk{Choices: []apiv1.Choice{{
				Delta: &apiv1.Message{ToolCalls: []byte(`[{"index":0,"function":{"name":"f"}}]`)},
			}}},
			want: true,
		},
		{
			name:  "finish reason only",
			chunk: &apiv1.ChatChunk{Choices: []apiv1.Choice{{Delta: &apiv1.Message{}, FinishReason: &stop}}},
			want:  true,
		},
		{
			name:  "usage only",
			chunk: &apiv1.ChatChunk{Usage: &apiv1.Usage{TotalTokens: 5}},
			want:  true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := chunkIsForwardable(tc.chunk); got != tc.want {
				t.Errorf("chunkIsForwardable = %v, want %v; a dropped chunk means the "+
					"client silently loses this part of the response", got, tc.want)
			}
		})
	}
}

// TestUnknownModelDoesNotLeakMetricSeries pins a denial-of-service found by
// audit. The `model` field comes from the request body and was used verbatim as a
// metric label, and a label is a permanently retained series — so any
// authenticated tenant could exhaust memory by sending a fresh random model name
// per request, with a 404 as the only visible symptom.
func TestUnknownModelDoesNotLeakMetricSeries(t *testing.T) {
	h := newHarness(t, 1, []config.Tenant{benchTenant()})

	const attempts = 200
	for i := 0; i < attempts; i++ {
		body := chatBody(fmt.Sprintf("evil-model-%d", i), "leak a series", false)
		resp := h.post("bench-key", body)
		if resp.StatusCode != 404 {
			t.Fatalf("expected 404 for an unconfigured model, got %d", resp.StatusCode)
		}
		resp.Body.Close()
	}

	// Every one of those requests must collapse into ONE series, not 200.
	out := h.server.deps.Metrics.Registry().String()
	leaked := strings.Count(out, "evil-model-")
	if leaked > 0 {
		t.Errorf("%d client-supplied model names reached metric labels; each is a "+
			"permanently retained series, so this is an unbounded memory leak "+
			"reachable from a request body", leaked)
	}
	if !strings.Contains(out, `model="unknown"`) {
		t.Errorf("unconfigured models should collapse to model=\"unknown\"; got:\n%s", out)
	}

	// A configured route must still be labelled by its real name, or the fix
	// would have destroyed the metric's usefulness.
	r := h.post("bench-key", chatBody("gw-chat", "legitimate", false))
	r.Body.Close()
	if !strings.Contains(h.server.deps.Metrics.Registry().String(), `model="gw-chat"`) {
		t.Error("a configured route lost its model label")
	}
}
