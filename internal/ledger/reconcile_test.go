package ledger

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/money"
)

// jsonMarshalLine encodes one JSONL line, mimicking what a provider-side logger
// writes.
func jsonMarshalLine(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// recordFor builds the provider-side record that agrees exactly with an entry.
func recordFor(e Entry) ProviderRecord {
	return ProviderRecord{
		RequestID: e.RequestID,
		Attempt:   e.Attempt,
		Provider:  e.Provider,
		Model:     e.UpstreamModel,
		Tokens:    e.Tokens,
		CostPico:  e.CostPico,
	}
}

func TestReconcileExactMatch(t *testing.T) {
	entries := []Entry{
		costed("r1", "acme", 1, 100, 50, 1000),
		costed("r2", "acme", 1, 7, 3, 1000),
		costed("r3", "globex", 1, 1, 1, 1500),
	}
	var records []ProviderRecord
	for _, e := range entries {
		records = append(records, recordFor(e))
	}

	rep := Reconcile(entries, records)
	if !rep.Balanced() {
		t.Fatalf("not balanced:\n%s", rep.Summary())
	}
	if !rep.Exact() {
		t.Fatalf("an all-reported, all-agreeing reconciliation must be Exact:\n%s", rep.Summary())
	}
	if len(rep.Matched) != 3 {
		t.Errorf("matched %d, want 3", len(rep.Matched))
	}
	if rep.TotalDeltaPico() != 0 {
		t.Errorf("delta = %d, want 0", rep.TotalDeltaPico())
	}
	if rep.LedgerCost != rep.UpstreamCost {
		t.Errorf("cost sums disagree: %d vs %d", rep.LedgerCost, rep.UpstreamCost)
	}
	if !strings.Contains(rep.Summary(), "EXACT") {
		t.Errorf("summary does not say EXACT:\n%s", rep.Summary())
	}
}

// TestReconcileReportsOneTokenDiscrepancy is the headline test for this file.
// A fuzz factor is the thing that turns a reconciliation into theatre, so a
// single token and a single picodollar must both be reported.
func TestReconcileReportsOneTokenDiscrepancy(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*ProviderRecord)
		wantField Field
		wantDiff  int64
	}{
		{
			name:      "one prompt token more upstream",
			mutate:    func(r *ProviderRecord) { r.Tokens.Prompt++ },
			wantField: FieldPromptTokens,
			wantDiff:  -1,
		},
		{
			name:      "one prompt token fewer upstream",
			mutate:    func(r *ProviderRecord) { r.Tokens.Prompt-- },
			wantField: FieldPromptTokens,
			wantDiff:  +1,
		},
		{
			name:      "one completion token",
			mutate:    func(r *ProviderRecord) { r.Tokens.Completion++ },
			wantField: FieldCompletionTokens,
			wantDiff:  -1,
		},
		{
			name:      "one cached token",
			mutate:    func(r *ProviderRecord) { r.Tokens.Cached++ },
			wantField: FieldCachedTokens,
			wantDiff:  -1,
		},
		{
			name:      "one reasoning token",
			mutate:    func(r *ProviderRecord) { r.Tokens.Reasoning++ },
			wantField: FieldReasoningTokens,
			wantDiff:  -1,
		},
		{
			// One picodollar is a trillionth of a dollar. It is still a mismatch:
			// exact arithmetic means the two sides either agree or they do not.
			name:      "one picodollar",
			mutate:    func(r *ProviderRecord) { r.CostPico++ },
			wantField: FieldCost,
			wantDiff:  -1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := costed("r1", "acme", 1, 100, 50, 1000)
			rec := recordFor(e)
			tc.mutate(&rec)

			rep := Reconcile([]Entry{e}, []ProviderRecord{rec})
			if rep.Balanced() {
				t.Fatalf("a %s discrepancy was swallowed:\n%s", tc.wantField, rep.Summary())
			}
			if rep.Exact() {
				t.Fatal("Exact() true despite a mismatch")
			}
			if len(rep.Matched) != 0 {
				t.Errorf("matched %d rows despite a discrepancy", len(rep.Matched))
			}
			if len(rep.Mismatched) != 1 {
				t.Fatalf("mismatched %d, want 1", len(rep.Mismatched))
			}
			m := rep.Mismatched[0]
			if m.Key != e.Key() {
				t.Errorf("mismatch key = %v, want %v", m.Key, e.Key())
			}
			if len(m.Deltas) != 1 {
				t.Fatalf("deltas = %+v, want exactly one on %s", m.Deltas, tc.wantField)
			}
			d := m.Deltas[0]
			if d.Field != tc.wantField {
				t.Errorf("delta field = %s, want %s", d.Field, tc.wantField)
			}
			if d.Diff() != tc.wantDiff {
				t.Errorf("delta diff = %+d, want %+d", d.Diff(), tc.wantDiff)
			}
			// The report must be legible enough for a human to act on.
			sum := rep.Summary()
			if !strings.Contains(sum, "MISMATCH") || !strings.Contains(sum, string(tc.wantField)) {
				t.Errorf("summary does not name the mismatched field:\n%s", sum)
			}
		})
	}
}

func TestReconcileReportsEveryDisagreeingFieldAtOnce(t *testing.T) {
	e := costed("r1", "acme", 1, 100, 50, 1000)
	e.Tokens.Cached = 20
	e.Tokens.Reasoning = 10
	rec := recordFor(e)
	rec.Tokens = TokenCounts{Prompt: 99, Cached: 21, Completion: 51, Reasoning: 9}
	rec.CostPico = e.CostPico - 5
	rec.Provider = "openai-secondary"
	rec.Model = "gpt-4o-2024-08-06"

	rep := Reconcile([]Entry{e}, []ProviderRecord{rec})
	if len(rep.Mismatched) != 1 {
		t.Fatalf("mismatched %d, want 1", len(rep.Mismatched))
	}
	got := map[Field]int64{}
	for _, d := range rep.Mismatched[0].Deltas {
		got[d.Field] = d.Diff()
	}
	want := map[Field]int64{
		FieldPromptTokens:     +1,
		FieldCachedTokens:     -1,
		FieldCompletionTokens: -1,
		FieldReasoningTokens:  +1,
		FieldCost:             +5,
		FieldProvider:         0,
		FieldModel:            0,
	}
	if len(got) != len(want) {
		t.Fatalf("reported %d deltas, want %d: %+v", len(got), len(want), rep.Mismatched[0].Deltas)
	}
	for f, wd := range want {
		gd, ok := got[f]
		if !ok {
			t.Errorf("field %s not reported", f)
			continue
		}
		if gd != wd {
			t.Errorf("field %s diff = %+d, want %+d", f, gd, wd)
		}
	}
	// The identity deltas must carry the text, not a meaningless zero.
	for _, d := range rep.Mismatched[0].Deltas {
		if d.Field == FieldProvider {
			if d.LedgerText != "openai-primary" || d.UpstreamText != "openai-secondary" {
				t.Errorf("provider delta = %+v, want the two names", d)
			}
			if !strings.Contains(d.String(), "openai-secondary") {
				t.Errorf("provider delta String() = %q", d.String())
			}
		}
	}
}

func TestReconcileEstimatedMatchIsBalancedButNotExact(t *testing.T) {
	// The dangerous state: the arithmetic agrees while our number was a guess.
	// A reconciliation that reported this as verified would be lying.
	e := costed("r1", "acme", 1, 100, 50, 1000)
	e.UsageSource = SourceEstimated
	rep := Reconcile([]Entry{e}, []ProviderRecord{recordFor(e)})

	if !rep.Balanced() {
		t.Fatalf("should balance:\n%s", rep.Summary())
	}
	if rep.Exact() {
		t.Fatal("Exact() true over an estimated row; an estimator that guesses " +
			"right does not turn a guess into a measurement")
	}
	if rep.EstimatedMatched != 1 {
		t.Errorf("EstimatedMatched = %d, want 1", rep.EstimatedMatched)
	}
	sum := rep.Summary()
	if !strings.Contains(sum, "not verified") {
		t.Errorf("summary must flag the estimate:\n%s", sum)
	}
	if !strings.Contains(sum, "is not evidence") {
		t.Errorf("summary must explain why:\n%s", sum)
	}
}

func TestReconcileMismatchFlagsEstimatedProvenance(t *testing.T) {
	e := costed("r1", "acme", 1, 100, 50, 1000)
	e.UsageSource = SourceEstimated
	rec := recordFor(e)
	rec.Tokens.Completion = 40 // our over-estimate was 10 tokens high

	rep := Reconcile([]Entry{e}, []ProviderRecord{rec})
	if len(rep.Mismatched) != 1 {
		t.Fatalf("want 1 mismatch, got %d", len(rep.Mismatched))
	}
	if !rep.Mismatched[0].LedgerEstimated {
		t.Error("LedgerEstimated false; the operator needs to know to fix the " +
			"estimator rather than hunt a double-charge")
	}
	if d := rep.Mismatched[0].Deltas[0]; d.Diff() != 10 {
		t.Errorf("diff = %+d, want +10 (the estimator over-counted)", d.Diff())
	}
}

func TestReconcileMissingOnEitherSide(t *testing.T) {
	billable := costed("r1", "acme", 1, 100, 50, 1000)

	// Ledger has a billable row the provider log does not.
	rep := Reconcile([]Entry{billable}, nil)
	if rep.Balanced() {
		t.Fatal("a billable ledger row with no provider record must not balance")
	}
	if len(rep.MissingUpstream) != 1 {
		t.Fatalf("MissingUpstream = %d, want 1", len(rep.MissingUpstream))
	}
	if rep.MissingUpstream[0].CostPico != billable.CostPico {
		t.Errorf("orphan cost = %d, want %d", rep.MissingUpstream[0].CostPico, billable.CostPico)
	}
	if !strings.Contains(rep.MissingUpstream[0].Reason, "over-billing") {
		t.Errorf("reason = %q; it should name the risk", rep.MissingUpstream[0].Reason)
	}
	if rep.LedgerCost != billable.CostPico {
		t.Errorf("LedgerCost = %d, want the orphan's cost %d", rep.LedgerCost, billable.CostPico)
	}

	// Provider log has a record the ledger does not: the expensive direction.
	rep = Reconcile(nil, []ProviderRecord{recordFor(billable)})
	if rep.Balanced() {
		t.Fatal("a provider record with no ledger row must not balance")
	}
	if len(rep.MissingLocal) != 1 {
		t.Fatalf("MissingLocal = %d, want 1", len(rep.MissingLocal))
	}
	if !strings.Contains(rep.Summary(), "MISSING LOCALLY") {
		t.Errorf("summary must lead with the expensive direction:\n%s", rep.Summary())
	}
	if rep.UpstreamCost != billable.CostPico {
		t.Errorf("UpstreamCost = %d, want %d", rep.UpstreamCost, billable.CostPico)
	}
}

func TestReconcileNonBillableRowsAreNotDiscrepancies(t *testing.T) {
	// A connect failure cost nothing and appears in no provider log. Reporting
	// it as missing would drown the real findings in noise.
	free := Entry{
		RequestID: "r1", Tenant: "acme", RequestedModel: "m", Attempt: 1,
		Provider: "openai-primary", Status: StatusFailed, UsageSource: SourceNone,
	}
	rep := Reconcile([]Entry{free}, nil)
	if !rep.Balanced() {
		t.Fatalf("a non-billable row with no provider record must balance:\n%s", rep.Summary())
	}
	if rep.NotBilled != 1 {
		t.Errorf("NotBilled = %d, want 1", rep.NotBilled)
	}
	if !rep.Exact() {
		t.Errorf("should still be Exact:\n%s", rep.Summary())
	}
}

func TestReconcileGatewaySideRowsAreExcluded(t *testing.T) {
	rows := []Entry{
		{
			RequestID: "r1", Tenant: "acme", RequestedModel: "m", Attempt: 1,
			Status: StatusCacheHit, UsageSource: SourceNone, CacheHit: true, ServedClient: true,
		},
		{
			RequestID: "r2", Tenant: "acme", RequestedModel: "m", Attempt: 1,
			Status: StatusRejected, UsageSource: SourceNone,
		},
	}
	rep := Reconcile(rows, nil)
	if rep.GatewaySide != 2 {
		t.Fatalf("GatewaySide = %d, want 2", rep.GatewaySide)
	}
	if !rep.Balanced() || !rep.Exact() {
		t.Fatalf("gateway-side rows must not create discrepancies:\n%s", rep.Summary())
	}
	if rep.NotBilled != 0 {
		t.Errorf("NotBilled = %d; gateway-side rows are counted separately", rep.NotBilled)
	}
}

func TestReconcileFailoverThreeProvidersOneResponse(t *testing.T) {
	// The modelled case from the spec: three cost rows, one client-visible
	// response, and all three must reconcile.
	a1 := costed("r1", "acme", 1, 100, 20, 1000)
	a1.ServedClient = false
	a1.Status = StatusFailedOver
	a1.Provider = "openai-primary"

	a2 := costed("r1", "acme", 2, 100, 5, 1000)
	a2.ServedClient = false
	a2.Status = StatusFailedOver
	a2.Provider = "openai-secondary"

	a3 := costed("r1", "acme", 3, 100, 60, 1000)
	a3.Provider = "anthropic-primary"

	entries := []Entry{a1, a2, a3}
	records := []ProviderRecord{recordFor(a1), recordFor(a2), recordFor(a3)}

	rep := Reconcile(entries, records)
	if !rep.Exact() {
		t.Fatalf("failover chain must reconcile exactly:\n%s", rep.Summary())
	}
	if len(rep.Matched) != 3 {
		t.Fatalf("matched %d of 3 attempts", len(rep.Matched))
	}
	wantCost := a1.CostPico + a2.CostPico + a3.CostPico
	if rep.LedgerCost != wantCost {
		t.Errorf("LedgerCost = %d, want %d: the failover attempts must be counted",
			rep.LedgerCost, wantCost)
	}

	// The failure mode this guards: collapsing the chain into one row loses the
	// two failed attempts' cost, and the provider log notices.
	collapsed := Reconcile([]Entry{a3}, records)
	if collapsed.Balanced() {
		t.Fatal("collapsing a failover chain to one row must be reported, not tolerated")
	}
	if len(collapsed.MissingLocal) != 2 {
		t.Fatalf("MissingLocal = %d, want the 2 dropped attempts", len(collapsed.MissingLocal))
	}
	if lost := collapsed.UpstreamCost - collapsed.LedgerCost; lost != a1.CostPico+a2.CostPico {
		t.Errorf("under-count = %d, want %d", lost, a1.CostPico+a2.CostPico)
	}
}

func TestReconcileAttemptNumberDisambiguatesSameProviderRetry(t *testing.T) {
	// Two attempts against the SAME provider must not collide. If the provider
	// log omits the attempt number, the collision is reported rather than
	// silently matching one and dropping the other.
	a1 := costed("r1", "acme", 1, 50, 10, 1000)
	a1.ServedClient = false
	a1.Status = StatusFailedOver
	a2 := costed("r1", "acme", 2, 50, 30, 1000)

	withAttempts := []ProviderRecord{recordFor(a1), recordFor(a2)}
	if rep := Reconcile([]Entry{a1, a2}, withAttempts); !rep.Exact() {
		t.Fatalf("attempt-numbered records must match:\n%s", rep.Summary())
	}

	// Same records, attempt number stripped: both normalise to attempt 1.
	stripped := []ProviderRecord{recordFor(a1), recordFor(a2)}
	stripped[0].Attempt = 0
	stripped[1].Attempt = 0
	rep := Reconcile([]Entry{a1, a2}, stripped)
	if rep.Balanced() {
		t.Fatalf("colliding provider records must be reported:\n%s", rep.Summary())
	}
	if len(rep.Duplicates) != 1 {
		t.Fatalf("Duplicates = %d, want 1", len(rep.Duplicates))
	}
	if !strings.Contains(rep.Duplicates[0].Reason, "attempt number") {
		t.Errorf("reason = %q; it should tell the operator how to fix the log",
			rep.Duplicates[0].Reason)
	}
}

func TestReconcileDuplicateLedgerRow(t *testing.T) {
	e := costed("r1", "acme", 1, 10, 5, 1000)
	rep := Reconcile([]Entry{e, e}, []ProviderRecord{recordFor(e)})
	if rep.Balanced() {
		t.Fatal("a duplicated ledger attempt must be reported")
	}
	if len(rep.Duplicates) != 1 {
		t.Fatalf("Duplicates = %d, want 1", len(rep.Duplicates))
	}
	if !strings.Contains(rep.Duplicates[0].Reason, "unique within a request") {
		t.Errorf("reason = %q", rep.Duplicates[0].Reason)
	}
	// The non-duplicate copy still matched, so the real cost is not lost.
	if len(rep.Matched) != 1 {
		t.Errorf("Matched = %d, want 1", len(rep.Matched))
	}
}

func TestReconcileTokenOnlyProviderLog(t *testing.T) {
	// Real provider logs often report tokens and not cost. Comparing against a
	// missing cost's zero would manufacture a mismatch on every row.
	e := costed("r1", "acme", 1, 100, 50, 1000)
	rec := recordFor(e)
	rec.CostPico = 0
	rec.CostNotReported = true
	rec.Provider = ""
	rec.Model = ""

	rep := Reconcile([]Entry{e}, []ProviderRecord{rec})
	if !rep.Balanced() {
		t.Fatalf("a token-only log must still reconcile on tokens:\n%s", rep.Summary())
	}
	if rep.Exact() {
		t.Fatal("Exact() must be false when the provider did not state a cost: " +
			"we verified the tokens, not the money")
	}
	if rep.CostNotReported != 1 {
		t.Errorf("CostNotReported = %d, want 1", rep.CostNotReported)
	}
	if rep.UpstreamCost != 0 {
		t.Errorf("UpstreamCost = %d; an unstated cost must not be summed as zero", rep.UpstreamCost)
	}

	// And a token-only log DOES still catch a token discrepancy.
	rec.Tokens.Completion++
	if rep := Reconcile([]Entry{e}, []ProviderRecord{rec}); rep.Balanced() {
		t.Fatal("a token-only log must still report a token mismatch")
	}
}

func TestReconcileIsDeterministic(t *testing.T) {
	var entries []Entry
	var records []ProviderRecord
	for i := 0; i < 40; i++ {
		e := costed(fmt.Sprintf("r%02d", i), fmt.Sprintf("t%d", i%3), 1, 10+i, 5+i, 1000)
		entries = append(entries, e)
		rec := recordFor(e)
		if i%7 == 0 {
			rec.Tokens.Completion += int(i%3) + 1
		}
		records = append(records, rec)
	}
	first := Reconcile(entries, records).Summary()
	for i := 0; i < 20; i++ {
		if got := Reconcile(entries, records).Summary(); got != first {
			t.Fatalf("run %d differs; the report must be diffable in CI:\n%s\n---\n%s", i, first, got)
		}
	}
	// Ledger input order drives the output order, so Matched is sorted by input.
	rep := Reconcile(entries, records)
	for i := 1; i < len(rep.Matched); i++ {
		if rep.Matched[i-1].RequestID >= rep.Matched[i].RequestID {
			t.Fatalf("Matched is not in ledger order: %v then %v",
				rep.Matched[i-1], rep.Matched[i])
		}
	}
}

func TestReconcileEndToEndThroughJSONL(t *testing.T) {
	// The reconciliation is meant to run offline over the persisted file, so it
	// must survive the round trip through JSONL. Anything the encoder drops is a
	// field the reconciliation silently stops checking.
	var buf bytes.Buffer
	l := New(&buf, Options{Now: fixedClock(epoch, time.Second), GuardWindow: -1})

	var provLog bytes.Buffer
	writeRec := func(r ProviderRecord) {
		b, err := jsonMarshalLine(r)
		if err != nil {
			t.Fatal(err)
		}
		provLog.Write(b)
	}

	for i := 0; i < 5; i++ {
		e := costed(fmt.Sprintf("r%d", i), "acme", 1, 100+i, 50+i, 1000)
		e.Tokens.Cached = i
		e.Tokens.Reasoning = i * 2
		if err := l.Append(e); err != nil {
			t.Fatal(err)
		}
		// The provider log carries no seq/status, only the reconciled facts.
		writeRec(ProviderRecord{
			RequestID: e.RequestID, Attempt: 1, Provider: e.Provider,
			Model: e.UpstreamModel, Tokens: e.Tokens, CostPico: e.CostPico,
		})
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}

	entries, err := Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	records, err := DecodeProviderLog(bytes.NewReader(provLog.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 5 || len(records) != 5 {
		t.Fatalf("decoded %d entries and %d records, want 5 each", len(entries), len(records))
	}
	rep := Reconcile(entries, records)
	if !rep.Exact() {
		t.Fatalf("the persisted file must reconcile exactly:\n%s", rep.Summary())
	}

	// Corrupt one token in the provider log and prove the file-based path still
	// notices. This is the check that the JSONL round trip did not quietly drop
	// the fields being compared.
	corrupt := strings.Replace(provLog.String(), `"completion":54`, `"completion":55`, 1)
	if corrupt == provLog.String() {
		t.Fatal("test setup: nothing was corrupted")
	}
	records, err = DecodeProviderLog(strings.NewReader(corrupt))
	if err != nil {
		t.Fatal(err)
	}
	rep = Reconcile(entries, records)
	if rep.Balanced() {
		t.Fatalf("a one-token corruption survived the file round trip:\n%s", rep.Summary())
	}
	if len(rep.Mismatched) != 1 || rep.Mismatched[0].Deltas[0].Field != FieldCompletionTokens {
		t.Fatalf("wrong finding:\n%s", rep.Summary())
	}
}

func TestDecodeProviderLogIsStrict(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		wantN   int
		wantErr string
	}{
		{"ok", `{"request_id":"r1","attempt":1,"tokens":{"prompt":1}}` + "\n", 1, ""},
		{"blank lines", "\n  \n" + `{"request_id":"r1"}` + "\n", 1, ""},
		{"garbage", "oops\n", 0, "provider log line 1"},
		{"truncated", `{"request_id":"r1"}` + "\n{\"req", 1, "provider log line 2"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DecodeProviderLog(strings.NewReader(tc.in))
			if tc.wantErr == "" && err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want it to contain %q", err, tc.wantErr)
				}
			}
			if len(got) != tc.wantN {
				t.Fatalf("got %d records, want %d", len(got), tc.wantN)
			}
		})
	}
}

func TestProviderRecordKeyNormalisesAttempt(t *testing.T) {
	tests := []struct {
		in   int
		want int
	}{{0, 1}, {-5, 1}, {1, 1}, {3, 3}}
	for _, tc := range tests {
		got := ProviderRecord{RequestID: "r", Attempt: tc.in}.Key()
		if got.Attempt != tc.want {
			t.Errorf("Attempt %d normalised to %d, want %d", tc.in, got.Attempt, tc.want)
		}
	}
	if s := (Key{RequestID: "r1", Attempt: 2}).String(); s != "r1#2" {
		t.Errorf("Key.String() = %q, want %q", s, "r1#2")
	}
}

func TestDeltaStringFormatsCostAsMoney(t *testing.T) {
	d := Delta{Field: FieldCost, Ledger: int64(money.Cents(5)), Upstream: int64(money.Cents(4))}
	s := d.String()
	// Picodollars are unreadable; the cost delta must render as dollars.
	for _, want := range []string{"$0.05", "$0.04", "$0.01"} {
		if !strings.Contains(s, want) {
			t.Errorf("Delta.String() = %q, want it to contain %q", s, want)
		}
	}
	plain := Delta{Field: FieldPromptTokens, Ledger: 10, Upstream: 12}.String()
	if !strings.Contains(plain, "delta=-2") {
		t.Errorf("token delta = %q, want a signed diff", plain)
	}
}

func TestAuditRequests(t *testing.T) {
	served := func(e Entry) Entry { e.ServedClient = true; return e }
	unserved := func(e Entry) Entry { e.ServedClient = false; e.Status = StatusFailedOver; return e }

	tests := []struct {
		name         string
		entries      []Entry
		wantProblems []string
	}{
		{
			name: "clean failover chain",
			entries: []Entry{
				unserved(costed("r1", "acme", 1, 10, 1, 1000)),
				unserved(costed("r1", "acme", 2, 10, 1, 1000)),
				served(costed("r1", "acme", 3, 10, 5, 1000)),
			},
		},
		{
			name:    "clean single attempt",
			entries: []Entry{served(costed("r1", "acme", 1, 10, 5, 1000))},
		},
		{
			name: "two rows served the client",
			entries: []Entry{
				served(costed("r1", "acme", 1, 10, 5, 1000)),
				served(costed("r1", "acme", 2, 10, 5, 1000)),
			},
			wantProblems: []string{"2 rows claim to have served the client"},
		},
		{
			name: "no row served the client",
			entries: []Entry{
				unserved(costed("r1", "acme", 1, 10, 5, 1000)),
			},
			wantProblems: []string{"no row served the client"},
		},
		{
			// The check the windowed Append guard cannot do: a gap in the attempt
			// numbers means a row was lost between attempt 1 and 3.
			name: "gap in attempt numbers",
			entries: []Entry{
				unserved(costed("r1", "acme", 1, 10, 5, 1000)),
				served(costed("r1", "acme", 3, 10, 5, 1000)),
			},
			wantProblems: []string{"attempt 2 is missing"},
		},
		{
			name: "duplicate attempt",
			entries: []Entry{
				unserved(costed("r1", "acme", 1, 10, 5, 1000)),
				served(costed("r1", "acme", 1, 10, 5, 1000)),
			},
			wantProblems: []string{"attempt 1 recorded 2 times"},
		},
		{
			// A request does not keep spending after it has answered the client.
			name: "spending continued after the client was served",
			entries: []Entry{
				served(costed("r1", "acme", 1, 10, 5, 1000)),
				unserved(costed("r1", "acme", 2, 10, 5, 1000)),
			},
			wantProblems: []string{"attempt 1 served the client but attempt 2 was also recorded"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := AuditRequests(tc.entries)
			if len(tc.wantProblems) == 0 {
				if len(got) != 0 {
					t.Fatalf("clean input flagged: %+v", got)
				}
				return
			}
			if len(got) != 1 {
				t.Fatalf("audits = %+v, want 1", got)
			}
			joined := strings.Join(got[0].Problems, " | ")
			for _, want := range tc.wantProblems {
				if !strings.Contains(joined, want) {
					t.Errorf("problems = %q, want it to contain %q", joined, want)
				}
			}
		})
	}
}

func TestAuditRequestsIgnoresGatewaySideRowsForUpstreamChecks(t *testing.T) {
	// A cache hit is a complete, correct request with one row and no upstream
	// attempt; it must not be flagged.
	hit := Entry{
		RequestID: "r1", Tenant: "acme", RequestedModel: "m", Attempt: 1,
		Status: StatusCacheHit, UsageSource: SourceNone, CacheHit: true, ServedClient: true,
	}
	if got := AuditRequests([]Entry{hit}); len(got) != 0 {
		t.Fatalf("cache hit flagged: %+v", got)
	}
	// A budget rejection never served the client, and that IS worth flagging:
	// the handler owes the client a response either way, so a rejection row with
	// no served row means the request vanished.
	rej := Entry{
		RequestID: "r2", Tenant: "acme", RequestedModel: "m", Attempt: 1,
		Status: StatusRejected, UsageSource: SourceNone,
	}
	if got := AuditRequests([]Entry{rej}); len(got) != 1 {
		t.Fatalf("rejection with no served row not flagged: %+v", got)
	}
}

func TestReportSummaryAccountsForEveryRow(t *testing.T) {
	// A report that reads N rows and says nothing about some of them is how a
	// reconciliation hides a gap.
	entries := []Entry{
		costed("m1", "acme", 1, 10, 5, 1000),
		func() Entry { e := costed("mm", "acme", 1, 10, 5, 1000); return e }(),
		{RequestID: "free", Tenant: "acme", RequestedModel: "m", Attempt: 1,
			Provider: "p", Status: StatusFailed, UsageSource: SourceNone},
		{RequestID: "hit", Tenant: "acme", RequestedModel: "m", Attempt: 1,
			Status: StatusCacheHit, UsageSource: SourceNone, CacheHit: true, ServedClient: true},
		costed("orphan", "acme", 1, 10, 5, 1000),
	}
	badRec := recordFor(entries[1])
	badRec.Tokens.Prompt++
	records := []ProviderRecord{
		recordFor(entries[0]),
		badRec,
		{RequestID: "ghost", Attempt: 1, Tokens: TokenCounts{Prompt: 1}, CostPico: 42},
	}

	rep := Reconcile(entries, records)
	accounted := len(rep.Matched) + len(rep.Mismatched) + len(rep.MissingUpstream) +
		rep.NotBilled + rep.GatewaySide
	if accounted != len(entries) {
		t.Fatalf("report accounts for %d of %d ledger rows:\n%s", accounted, len(entries), rep.Summary())
	}
	upstreamAccounted := len(rep.Matched) + len(rep.Mismatched) + len(rep.MissingLocal)
	if upstreamAccounted != len(records) {
		t.Fatalf("report accounts for %d of %d upstream rows:\n%s",
			upstreamAccounted, len(records), rep.Summary())
	}
	if rep.LedgerRows != len(entries) || rep.UpstreamRows != len(records) {
		t.Errorf("input sizes not recorded: %d/%d", rep.LedgerRows, rep.UpstreamRows)
	}
	sum := rep.Summary()
	for _, want := range []string{"MISMATCH", "MISSING LOCALLY", "MISSING UPSTREAM", "excluded:"} {
		if !strings.Contains(sum, want) {
			t.Errorf("summary lacks %q:\n%s", want, sum)
		}
	}
}

func TestReconcileEmptyInputs(t *testing.T) {
	rep := Reconcile(nil, nil)
	if !rep.Balanced() || !rep.Exact() {
		t.Fatalf("nothing versus nothing must reconcile:\n%s", rep.Summary())
	}
	if rep.LedgerRows != 0 || rep.UpstreamRows != 0 {
		t.Errorf("row counts = %d/%d, want 0/0", rep.LedgerRows, rep.UpstreamRows)
	}
}
