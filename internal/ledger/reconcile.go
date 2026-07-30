package ledger

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/harsha-moparthy/llmgw/internal/money"
)

// ProviderRecord is one row of a provider-side log: what the upstream says it
// served and what it says it charged.
//
// This is the other half of the reconciliation. The mock provider in this
// project emits exactly these fields, which is what lets the end-to-end test
// assert that the gateway's ledger and the provider's own accounting agree to
// the picodollar rather than merely "look about right".
type ProviderRecord struct {
	RequestID string      `json:"request_id"`
	Attempt   int         `json:"attempt"`
	Provider  string      `json:"provider,omitempty"`
	Model     string      `json:"model,omitempty"`
	Tokens    TokenCounts `json:"tokens"`
	// CostPico is the provider's own cost figure. A provider that does not
	// report cost should leave CostNotReported set instead of sending zero,
	// because zero is a claim and "I did not say" is not.
	CostPico        money.Pico `json:"cost_pico"`
	CostNotReported bool       `json:"cost_not_reported,omitempty"`
}

// Key returns the record's reconciliation key.
//
// An attempt of zero is normalised to 1. Provider logs in the wild have no
// concept of the gateway's failover attempts, so a log that omits the field is
// describing the first (and, for that provider, only) attempt. If the gateway
// really did hit the same provider twice for one request and the log cannot
// distinguish them, the two records collide on this key and Reconcile reports a
// duplicate rather than quietly matching one and dropping the other.
func (r ProviderRecord) Key() Key {
	a := r.Attempt
	if a <= 0 {
		a = 1
	}
	return Key{RequestID: r.RequestID, Attempt: a}
}

// Field names the quantity that disagreed.
type Field string

// Reconcilable fields.
const (
	FieldPromptTokens     Field = "prompt_tokens"
	FieldCachedTokens     Field = "cached_tokens"
	FieldCompletionTokens Field = "completion_tokens"
	FieldReasoningTokens  Field = "reasoning_tokens"
	FieldCost             Field = "cost_pico"
	FieldProvider         Field = "provider"
	FieldModel            Field = "model"
)

// Delta is a single disagreement on a single field.
type Delta struct {
	Field Field `json:"field"`
	// Ledger and Upstream are the two numeric claims. For the string-valued
	// fields (provider, model) they are zero and LedgerText/UpstreamText carry
	// the values instead.
	Ledger   int64 `json:"ledger"`
	Upstream int64 `json:"upstream"`

	LedgerText   string `json:"ledger_text,omitempty"`
	UpstreamText string `json:"upstream_text,omitempty"`
}

// Diff is ledger minus upstream: positive means the gateway over-counted.
func (d Delta) Diff() int64 { return d.Ledger - d.Upstream }

// String renders the delta for a report line.
func (d Delta) String() string {
	if d.LedgerText != "" || d.UpstreamText != "" {
		return fmt.Sprintf("%s: ledger=%q upstream=%q", d.Field, d.LedgerText, d.UpstreamText)
	}
	if d.Field == FieldCost {
		return fmt.Sprintf("%s: ledger=%s upstream=%s delta=%+d pico (%s)",
			d.Field, money.FormatUSD(money.Pico(d.Ledger)),
			money.FormatUSD(money.Pico(d.Upstream)), d.Diff(),
			money.FormatUSD(money.Pico(d.Diff())))
	}
	return fmt.Sprintf("%s: ledger=%d upstream=%d delta=%+d",
		d.Field, d.Ledger, d.Upstream, d.Diff())
}

// Mismatch is one attempt present on both sides whose numbers disagree.
type Mismatch struct {
	Key    Key     `json:"key"`
	Deltas []Delta `json:"deltas"`
	// LedgerEstimated flags the case where the disagreement is explained by the
	// gateway having guessed. It is not an excuse — the row is still a mismatch —
	// but it tells the operator to fix the estimator rather than hunt a
	// double-charge.
	LedgerEstimated bool `json:"ledger_estimated"`
}

// Orphan is an attempt that exists on one side only.
type Orphan struct {
	Key      Key        `json:"key"`
	Provider string     `json:"provider,omitempty"`
	Tokens   int        `json:"tokens"`
	CostPico money.Pico `json:"cost_pico"`
	// Reason explains why the row is unmatched, so the report distinguishes "we
	// lost a write" from "we charged for something the provider has no record
	// of".
	Reason string `json:"reason"`
}

// Report is the outcome of a reconciliation.
//
// The design commitment: there is no tolerance parameter anywhere in this type
// or the function that builds it. A one-picodollar or one-token disagreement is
// a mismatch, full stop. Every "reconciliation" that quietly passes in
// production does so because someone added an epsilon to stop a nightly job
// paging them, and after that the job no longer tests anything.
type Report struct {
	// Matched are attempts whose every reconciled field agreed exactly.
	Matched []Key `json:"matched"`
	// Mismatched are attempts present on both sides that disagreed.
	Mismatched []Mismatch `json:"mismatched"`
	// MissingUpstream are billable ledger rows with no provider record: the
	// gateway believes it was charged and the provider has no matching line.
	MissingUpstream []Orphan `json:"missing_upstream"`
	// MissingLocal are provider records with no ledger row: the provider charged
	// us for something the gateway never recorded. This is the expensive
	// direction and it is why the reconciliation runs in both.
	MissingLocal []Orphan `json:"missing_local"`
	// Duplicates are keys that appeared more than once on one side.
	Duplicates []Orphan `json:"duplicates"`

	// NotBilled counts ledger rows that reached a provider, were judged
	// non-billable, and indeed have no provider record. Expected, not a problem,
	// but counted so the report accounts for every row it read.
	NotBilled int `json:"not_billed"`
	// GatewaySide counts rows that never reached a provider (cache hits, budget
	// rejections) and so were excluded from matching.
	GatewaySide int `json:"gateway_side"`
	// EstimatedMatched counts matched rows whose ledger numbers were estimated.
	// These are the rows that make a "balanced" report untrustworthy: an
	// estimator that happens to guess right does not turn a guess into a
	// measurement.
	EstimatedMatched int `json:"estimated_matched"`

	// LedgerRows and UpstreamRows are the input sizes, so a report can be
	// audited for having looked at everything.
	LedgerRows   int `json:"ledger_rows"`
	UpstreamRows int `json:"upstream_rows"`

	// LedgerCost and UpstreamCost are the totals over every row the report
	// attributes to a side: matched, mismatched, AND orphaned. Including the
	// orphans is what makes TotalDeltaPico the true net over/under-count rather
	// than merely the drift among rows that happened to pair up — the money lost
	// by dropping a whole row is exactly the money an operator most wants named.
	// They are reported even when the per-row comparison passes, because a pair
	// of compensating per-row errors is conceivable and the sums are a cheap
	// independent check.
	LedgerCost   money.Pico `json:"ledger_cost_pico"`
	UpstreamCost money.Pico `json:"upstream_cost_pico"`
	// CostNotReported counts provider records that declined to state a cost, so
	// their cost field was not compared and is excluded from UpstreamCost.
	CostNotReported int `json:"cost_not_reported"`
}

// Balanced reports whether every attempt lines up: nothing mismatched, nothing
// missing on either side, no duplicates.
func (r *Report) Balanced() bool {
	return len(r.Mismatched) == 0 && len(r.MissingUpstream) == 0 &&
		len(r.MissingLocal) == 0 && len(r.Duplicates) == 0
}

// Exact reports whether the reconciliation is a proof rather than a coincidence:
// balanced AND built entirely from provider-reported numbers.
//
// Balanced-but-not-Exact is the dangerous state and it deserves its own name.
// It means the arithmetic agrees while at least one row on our side was a guess
// the estimator got right — possibly because the estimator is good, possibly
// because the two sides are both derived from the same wrong assumption. Only
// Exact justifies telling a customer the invoice is verified.
func (r *Report) Exact() bool {
	return r.Balanced() && r.EstimatedMatched == 0 && r.CostNotReported == 0
}

// TotalDeltaPico is the signed cost difference over rows that were compared.
// Positive means the gateway billed more than the provider did.
func (r *Report) TotalDeltaPico() money.Pico { return r.LedgerCost - r.UpstreamCost }

// Summary renders a human-readable multi-line report. Written by hand rather
// than reflected out of the struct so the ordering puts the expensive findings
// first: an operator reading a truncated log should see MissingLocal before
// they see the matched count.
func (r *Report) Summary() string {
	var sb strings.Builder
	verdict := "MISMATCH"
	switch {
	case r.Exact():
		verdict = "EXACT"
	case r.Balanced():
		verdict = "BALANCED (contains estimates; not verified)"
	}
	fmt.Fprintf(&sb, "reconciliation: %s\n", verdict)
	fmt.Fprintf(&sb, "  ledger rows=%d upstream rows=%d matched=%d mismatched=%d\n",
		r.LedgerRows, r.UpstreamRows, len(r.Matched), len(r.Mismatched))
	fmt.Fprintf(&sb, "  ledger cost=%s upstream cost=%s delta=%s\n",
		money.FormatUSD(r.LedgerCost), money.FormatUSD(r.UpstreamCost),
		money.FormatUSD(r.TotalDeltaPico()))
	for _, o := range r.MissingLocal {
		fmt.Fprintf(&sb, "  MISSING LOCALLY %s provider=%s cost=%s: %s\n",
			o.Key, o.Provider, money.FormatUSD(o.CostPico), o.Reason)
	}
	for _, o := range r.MissingUpstream {
		fmt.Fprintf(&sb, "  MISSING UPSTREAM %s provider=%s cost=%s: %s\n",
			o.Key, o.Provider, money.FormatUSD(o.CostPico), o.Reason)
	}
	for _, o := range r.Duplicates {
		fmt.Fprintf(&sb, "  DUPLICATE %s: %s\n", o.Key, o.Reason)
	}
	for _, m := range r.Mismatched {
		fmt.Fprintf(&sb, "  MISMATCH %s estimated=%t\n", m.Key, m.LedgerEstimated)
		for _, d := range m.Deltas {
			fmt.Fprintf(&sb, "      %s\n", d)
		}
	}
	if r.EstimatedMatched > 0 {
		fmt.Fprintf(&sb, "  note: %d matched row(s) used ESTIMATED usage; "+
			"agreement there is not evidence\n", r.EstimatedMatched)
	}
	if r.NotBilled > 0 || r.GatewaySide > 0 || r.CostNotReported > 0 {
		fmt.Fprintf(&sb, "  excluded: not_billed=%d gateway_side=%d cost_not_reported=%d\n",
			r.NotBilled, r.GatewaySide, r.CostNotReported)
	}
	return sb.String()
}

// Reconcile compares gateway ledger entries against provider-side records.
//
// Both sides are indexed by (request id, attempt) and every field either side
// claims is compared with ==. Rows are examined in a deterministic order
// (ledger sequence, then upstream input order) so two runs over the same inputs
// produce byte-identical reports, which is what makes the report diffable in CI.
func Reconcile(entries []Entry, records []ProviderRecord) *Report {
	rep := &Report{LedgerRows: len(entries), UpstreamRows: len(records)}

	type ledgerSlot struct {
		entry *Entry
		order int
	}
	byKey := make(map[Key]*ledgerSlot, len(entries))
	// order preserves the input order of keys eligible for matching.
	order := make([]Key, 0, len(entries))
	for i := range entries {
		e := &entries[i]
		if !e.ReachedProvider() {
			rep.GatewaySide++
			continue
		}
		k := e.Key()
		if prev, dup := byKey[k]; dup {
			rep.Duplicates = append(rep.Duplicates, Orphan{
				Key:      k,
				Provider: e.Provider,
				Tokens:   e.Tokens.Total(),
				CostPico: e.CostPico,
				Reason: fmt.Sprintf(
					"ledger has two rows for this attempt (seq %d and %d); "+
						"an attempt number must be unique within a request",
					prev.entry.Seq, e.Seq),
			})
			continue
		}
		byKey[k] = &ledgerSlot{entry: e, order: len(order)}
		order = append(order, k)
	}

	matchedUpstream := make(map[Key]bool, len(records))
	// Collect mismatches keyed so they can be emitted in ledger order.
	mismatchByKey := make(map[Key]Mismatch, 8)
	matchedByKey := make(map[Key]bool, len(records))

	for i := range records {
		r := records[i]
		k := r.Key()
		if matchedUpstream[k] {
			rep.Duplicates = append(rep.Duplicates, Orphan{
				Key:      k,
				Provider: r.Provider,
				Tokens:   r.Tokens.Total(),
				CostPico: r.CostPico,
				Reason: "provider log has two records for this attempt; " +
					"if the gateway really retried the same provider, the log must " +
					"carry the attempt number to disambiguate",
			})
			continue
		}
		matchedUpstream[k] = true

		slot, ok := byKey[k]
		if !ok {
			rep.MissingLocal = append(rep.MissingLocal, Orphan{
				Key:      k,
				Provider: r.Provider,
				Tokens:   r.Tokens.Total(),
				CostPico: r.CostPico,
				Reason: "provider charged for an attempt the gateway has no row for " +
					"(a lost ledger write, or a request that died before it could be recorded)",
			})
			if !r.CostNotReported {
				rep.UpstreamCost += r.CostPico
			} else {
				rep.CostNotReported++
			}
			continue
		}
		e := slot.entry
		deltas := compare(e, &r)
		if !r.CostNotReported {
			rep.UpstreamCost += r.CostPico
		} else {
			rep.CostNotReported++
		}
		rep.LedgerCost += e.CostPico
		if len(deltas) == 0 {
			matchedByKey[k] = true
			if e.Estimated() {
				rep.EstimatedMatched++
			}
			continue
		}
		mismatchByKey[k] = Mismatch{Key: k, Deltas: deltas, LedgerEstimated: e.Estimated()}
	}

	// Walk the ledger in input order so Matched/Mismatched are deterministic and
	// so unmatched ledger rows are classified.
	for _, k := range order {
		mm, isMismatch := mismatchByKey[k]
		switch {
		case matchedByKey[k]:
			rep.Matched = append(rep.Matched, k)
		case isMismatch:
			rep.Mismatched = append(rep.Mismatched, mm)
		default:
			e := byKey[k].entry
			if !e.Billable {
				// We said the provider would not have charged us and it did not.
				// That is the judgment working, not a discrepancy.
				rep.NotBilled++
				continue
			}
			rep.MissingUpstream = append(rep.MissingUpstream, Orphan{
				Key:      k,
				Provider: e.Provider,
				Tokens:   e.Tokens.Total(),
				CostPico: e.CostPico,
				Reason: "gateway recorded a billable attempt the provider log does not " +
					"contain (over-billing the tenant, or a provider log gap)",
			})
			rep.LedgerCost += e.CostPico
		}
	}
	return rep
}

// compare produces the deltas between one ledger row and one provider record.
// Every comparison is exact equality; there is deliberately no epsilon.
func compare(e *Entry, r *ProviderRecord) []Delta {
	var d []Delta
	if e.Tokens.Prompt != r.Tokens.Prompt {
		d = append(d, Delta{Field: FieldPromptTokens, Ledger: int64(e.Tokens.Prompt), Upstream: int64(r.Tokens.Prompt)})
	}
	if e.Tokens.Cached != r.Tokens.Cached {
		d = append(d, Delta{Field: FieldCachedTokens, Ledger: int64(e.Tokens.Cached), Upstream: int64(r.Tokens.Cached)})
	}
	if e.Tokens.Completion != r.Tokens.Completion {
		d = append(d, Delta{Field: FieldCompletionTokens, Ledger: int64(e.Tokens.Completion), Upstream: int64(r.Tokens.Completion)})
	}
	if e.Tokens.Reasoning != r.Tokens.Reasoning {
		d = append(d, Delta{Field: FieldReasoningTokens, Ledger: int64(e.Tokens.Reasoning), Upstream: int64(r.Tokens.Reasoning)})
	}
	// A provider that did not state a cost is not disagreeing about it. Comparing
	// against its zero would manufacture a mismatch on every row of a log that
	// only reports tokens.
	if !r.CostNotReported && e.CostPico != r.CostPico {
		d = append(d, Delta{Field: FieldCost, Ledger: int64(e.CostPico), Upstream: int64(r.CostPico)})
	}
	// The identity fields are only compared when the provider log states them,
	// so a token-only log is reconcilable.
	if r.Provider != "" && e.Provider != r.Provider {
		d = append(d, Delta{Field: FieldProvider, LedgerText: e.Provider, UpstreamText: r.Provider})
	}
	if r.Model != "" && e.UpstreamModel != "" && e.UpstreamModel != r.Model {
		d = append(d, Delta{Field: FieldModel, LedgerText: e.UpstreamModel, UpstreamText: r.Model})
	}
	return d
}

// Decode reads a JSONL ledger stream.
//
// It is strict: a malformed line is an error rather than a skip. A reconciliation
// that silently ignored the lines it could not parse would report a clean match
// over whichever subset happened to decode, which is the failure mode this whole
// package exists to prevent. The reported error carries the line number so the
// operator can look at the byte.
func Decode(r io.Reader) ([]Entry, error) {
	sc := bufio.NewScanner(r)
	// Ledger lines carry prompts only as token counts, never as text, so they are
	// small; 1 MiB is generous headroom against the 64 KiB default.
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
	var out []Entry
	line := 0
	for sc.Scan() {
		line++
		b := sc.Bytes()
		if len(strings.TrimSpace(string(b))) == 0 {
			continue
		}
		var e Entry
		if err := json.Unmarshal(b, &e); err != nil {
			return out, fmt.Errorf("ledger: decode line %d: %w", line, err)
		}
		out = append(out, e)
	}
	if err := sc.Err(); err != nil {
		return out, fmt.Errorf("ledger: read line %d: %w", line+1, err)
	}
	return out, nil
}

// DecodeFile reads a JSONL ledger file from disk.
func DecodeFile(path string) ([]Entry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("ledger: open %s: %w", path, err)
	}
	defer f.Close()
	return Decode(f)
}

// DecodeProviderLog reads a JSONL provider-side log.
func DecodeProviderLog(r io.Reader) ([]ProviderRecord, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
	var out []ProviderRecord
	line := 0
	for sc.Scan() {
		line++
		b := sc.Bytes()
		if len(strings.TrimSpace(string(b))) == 0 {
			continue
		}
		var rec ProviderRecord
		if err := json.Unmarshal(b, &rec); err != nil {
			return out, fmt.Errorf("ledger: decode provider log line %d: %w", line, err)
		}
		out = append(out, rec)
	}
	if err := sc.Err(); err != nil {
		return out, fmt.Errorf("ledger: read provider log line %d: %w", line+1, err)
	}
	return out, nil
}

// RequestAudit is the per-request consistency finding produced by AuditRequests.
type RequestAudit struct {
	RequestID string   `json:"request_id"`
	Attempts  int      `json:"attempts"`
	Problems  []string `json:"problems"`
}

// AuditRequests is the unbounded version of the Append-time consistency guard:
// it walks a whole decoded ledger and reports per-request structural problems.
//
// Append's guard is windowed because remembering every request id forever in a
// long-lived process is a leak. This runs offline over the file, where memory is
// bounded by the file rather than by uptime, and so it can check what the guard
// cannot: that attempt numbers are contiguous from 1, and that a request with a
// served row has no *later* attempts after it (a request does not keep spending
// money after it has answered the client).
func AuditRequests(entries []Entry) []RequestAudit {
	type state struct {
		attempts   map[int]int
		served     []int
		maxAttempt int
		upstream   int
	}
	byReq := make(map[string]*state)
	order := make([]string, 0, 16)
	for i := range entries {
		e := &entries[i]
		st, ok := byReq[e.RequestID]
		if !ok {
			st = &state{attempts: make(map[int]int, 2)}
			byReq[e.RequestID] = st
			order = append(order, e.RequestID)
		}
		st.attempts[e.Attempt]++
		if e.Attempt > st.maxAttempt {
			st.maxAttempt = e.Attempt
		}
		if e.ReachedProvider() {
			st.upstream++
		}
		if e.ServedClient {
			st.served = append(st.served, e.Attempt)
		}
	}
	var out []RequestAudit
	for _, id := range order {
		st := byReq[id]
		var problems []string
		if len(st.served) > 1 {
			sort.Ints(st.served)
			problems = append(problems, fmt.Sprintf(
				"%d rows claim to have served the client (attempts %v); a request has one response",
				len(st.served), st.served))
		}
		if len(st.served) == 0 {
			problems = append(problems, "no row served the client; every request either "+
				"answers or records a terminal failure")
		}
		for a := 1; a <= st.maxAttempt; a++ {
			switch n := st.attempts[a]; {
			case n == 0:
				problems = append(problems, fmt.Sprintf(
					"attempt %d is missing though attempt %d exists; attempts must be contiguous from 1",
					a, st.maxAttempt))
			case n > 1:
				problems = append(problems, fmt.Sprintf("attempt %d recorded %d times", a, n))
			}
		}
		if len(st.served) == 1 && st.served[0] < st.maxAttempt {
			problems = append(problems, fmt.Sprintf(
				"attempt %d served the client but attempt %d was also recorded; "+
					"a request stops spending once it has answered",
				st.served[0], st.maxAttempt))
		}
		if len(problems) > 0 {
			out = append(out, RequestAudit{
				RequestID: id, Attempts: len(st.attempts), Problems: problems,
			})
		}
	}
	return out
}
