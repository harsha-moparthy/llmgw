// Package pricing is the gateway's cost model: an immutable, exact price sheet
// and the arithmetic that turns a provider's usage record into an explainable
// charge.
//
// Two commitments shape everything here.
//
// First, no price is ever rounded. List prices arrive quoted per 1,000,000
// tokens ("$0.15") and are converted to a per-token price by money.PerToken,
// which refuses a division that is not exact. A price the gateway cannot
// represent exactly is rejected when the sheet loads, with the offending model
// named, rather than being silently rounded into every invoice that follows.
//
// Second, an unknown model is an error, never a zero. A billing system that
// prices what it does not recognise at $0 is the worst possible failure mode,
// because nothing looks broken: requests succeed, the ledger balances against
// itself, dashboards are green, and the money is simply gone. So Lookup returns
// an *UnpricedError and the caller — not this package — decides whether to fail
// the request closed or serve it and flag the row.
//
// A Table is immutable once built, so the request path reads it with no lock at
// all. Reloading a price sheet means building a new Table and swapping the
// pointer (see internal/config); it never means mutating a live one.
package pricing

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/money"
)

// Rates is the per-token price of one model, already divided down from the
// per-million list price and proven exact at load time.
//
// CachedInput is the rate for prompt tokens the provider served from its own
// prefix cache. It is a separate field rather than a discount percentage
// because vendors quote it as an absolute price, and deriving it from a
// percentage would reintroduce the inexact division this package exists to
// avoid.
type Rates struct {
	Input       money.Pico
	CachedInput money.Pico
	Output      money.Pico
}

// NewRates builds per-token Rates from prices quoted per 1,000,000 tokens.
//
// Every vendor publishes per-million prices, so this is the only constructor
// that should be used with a number taken off a pricing page. It fails on a
// price that is not exactly representable per token, and the error names the
// field so an operator can find the typo.
func NewRates(inputPerMillion, cachedInputPerMillion, outputPerMillion money.Pico) (Rates, error) {
	in, err := money.PerToken(inputPerMillion)
	if err != nil {
		return Rates{}, fmt.Errorf("input price: %w", err)
	}
	cached, err := money.PerToken(cachedInputPerMillion)
	if err != nil {
		return Rates{}, fmt.Errorf("cached input price: %w", err)
	}
	out, err := money.PerToken(outputPerMillion)
	if err != nil {
		return Rates{}, fmt.Errorf("output price: %w", err)
	}
	return Rates{Input: in, CachedInput: cached, Output: out}, nil
}

// Rule records which lookup rule matched a model name. It is carried through to
// the ledger so a bill can say not just what it charged but why it believed
// that was the right price.
type Rule int

const (
	// RuleExact is a literal hit on a model name or a configured alias. This is
	// the only rule that carries no inference.
	RuleExact Rule = iota
	// RuleVendorPrefix matched after stripping a routing prefix, so that
	// "openai/gpt-4o" and "openrouter/openai/gpt-4o" price as "gpt-4o".
	RuleVendorPrefix
	// RuleDatedSnapshot matched after stripping a dated snapshot suffix, so
	// that "gpt-4o-2024-08-06" prices as "gpt-4o".
	RuleDatedSnapshot
	// RuleVendorPrefixAndDatedSnapshot matched after stripping both.
	RuleVendorPrefixAndDatedSnapshot
)

// String renders the rule for logs, metric labels and ledger rows.
func (r Rule) String() string {
	switch r {
	case RuleExact:
		return "exact"
	case RuleVendorPrefix:
		return "vendor_prefix_stripped"
	case RuleDatedSnapshot:
		return "dated_snapshot_stripped"
	case RuleVendorPrefixAndDatedSnapshot:
		return "vendor_prefix_and_dated_snapshot_stripped"
	default:
		return "invalid"
	}
}

// Exact reports whether the rule involved no inference about the model name.
func (r Rule) Exact() bool { return r == RuleExact }

// Match is the result of resolving a model name against a Table.
type Match struct {
	// Requested is the model name the client asked for.
	Requested string
	// PricedAs is the table entry that supplied the rates. Equal to Requested
	// for RuleExact hits on a model name; different for an alias or a
	// normalisation rule.
	PricedAs string
	// Rule is how PricedAs was derived from Requested.
	Rule Rule
	// Rates is the per-token price sheet for the match.
	Rates Rates
}

// ErrUnpriced is the sentinel behind every *UnpricedError, so callers can
// branch on errors.Is without depending on the concrete type.
var ErrUnpriced = errors.New("model is not priced")

// UnpricedError reports that no rule could resolve a model to a price.
//
// It is a typed error rather than a formatted string because the two callers
// need the model name programmatically: the HTTP layer puts it in an
// OpenAI-shaped error body, and the metrics layer uses it as a label so an
// operator can see *which* model is unpriced without grepping logs.
type UnpricedError struct {
	Model string
	// Tried lists the candidate names the fallback rules generated, so the
	// error explains what the gateway looked for and not just what it wanted.
	Tried []string
}

func (e *UnpricedError) Error() string {
	if len(e.Tried) == 0 {
		return fmt.Sprintf("pricing: model %q is not priced", e.Model)
	}
	return fmt.Sprintf("pricing: model %q is not priced (also tried %s)",
		e.Model, strings.Join(quoteAll(e.Tried), ", "))
}

// Unwrap ties the typed error to the sentinel.
func (e *UnpricedError) Unwrap() error { return ErrUnpriced }

func quoteAll(ss []string) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = fmt.Sprintf("%q", s)
	}
	return out
}

// ErrInvalidUsage is the sentinel for a usage record that cannot be billed
// because it is internally contradictory.
var ErrInvalidUsage = errors.New("invalid usage record")

type entry struct {
	canonical string
	rates     Rates
}

// Table maps model names to Rates.
//
// It is read-only after construction, which is what lets the request path use
// it from many goroutines with no synchronisation. Nothing in this package
// takes a lock, and nothing should need to: correctness comes from immutability
// rather than from a mutex whose critical section would sit on the hot path of
// every request.
type Table struct {
	entries map[string]entry
	// models is the sorted list of canonical model names, precomputed because
	// the admin endpoint that lists them should not sort under load.
	models []string
}

// Len returns the number of canonical models priced by the table.
func (t *Table) Len() int {
	if t == nil {
		return 0
	}
	return len(t.models)
}

// Models returns the canonical model names in sorted order. The returned slice
// is shared and must not be modified; it exists so an admin endpoint can render
// the price sheet without walking a map.
func (t *Table) Models() []string {
	if t == nil {
		return nil
	}
	return t.models
}

// Lookup resolves a model name to its rates.
//
// The rules, in order, are:
//
//  1. exact match on a model name or a configured alias;
//  2. exact match after stripping a routing prefix up to the last '/';
//  3. exact match after stripping a trailing dated snapshot suffix
//     ("-2024-08-06" or "-20241022");
//  4. exact match after stripping both.
//
// Anything else is an *UnpricedError. Rules 2-4 exist because model names in
// the wild are decorated versions of the names on the pricing page, and failing
// a request over a date suffix would be pedantic. They are deliberately narrow:
// no case folding, no prefix-of-a-prefix search, no "closest" match. A rule
// that guessed would silently bill a customer at another model's price, which
// is a quieter and worse bug than refusing to price the request. Every non-exact
// hit records its Rule so the ledger can flag it for review.
func (t *Table) Lookup(model string) (Match, error) {
	if t == nil || len(t.entries) == 0 {
		return Match{}, &UnpricedError{Model: model}
	}
	if e, ok := t.entries[model]; ok {
		return Match{Requested: model, PricedAs: e.canonical, Rule: RuleExact, Rates: e.rates}, nil
	}

	// Fixed-size array, not a slice: generating the candidates must not allocate,
	// because a client looping on a decorated-but-valid model name (the common
	// case for rules 2-4) would otherwise produce garbage on every request. Only
	// an outright miss allocates, and only to build the error.
	var cands [3]struct {
		name string
		rule Rule
	}
	n := 0
	noPrefix, hadPrefix := stripRoutingPrefix(model)
	if hadPrefix {
		cands[n].name, cands[n].rule = noPrefix, RuleVendorPrefix
		n++
	}
	if base, ok := stripDatedSuffix(model); ok {
		cands[n].name, cands[n].rule = base, RuleDatedSnapshot
		n++
	}
	if hadPrefix {
		if base, ok := stripDatedSuffix(noPrefix); ok {
			cands[n].name, cands[n].rule = base, RuleVendorPrefixAndDatedSnapshot
			n++
		}
	}
	for i := 0; i < n; i++ {
		if e, ok := t.entries[cands[i].name]; ok {
			return Match{
				Requested: model,
				PricedAs:  e.canonical,
				Rule:      cands[i].rule,
				Rates:     e.rates,
			}, nil
		}
	}
	tried := make([]string, 0, n)
	for i := 0; i < n; i++ {
		tried = append(tried, cands[i].name)
	}
	return Match{}, &UnpricedError{Model: model, Tried: tried}
}

// stripRoutingPrefix removes a routing namespace up to the last '/', so both
// "openai/gpt-4o" and "openrouter/openai/gpt-4o" reduce to "gpt-4o".
func stripRoutingPrefix(model string) (string, bool) {
	i := strings.LastIndexByte(model, '/')
	if i < 0 || i == len(model)-1 {
		return model, false
	}
	return model[i+1:], true
}

// stripDatedSuffix removes a trailing "-YYYY-MM-DD" or "-YYYYMMDD" snapshot
// suffix.
//
// Only 8-digit dates are recognised. OpenAI's older 4-digit forms
// ("gpt-4-0613") are deliberately left alone: four trailing digits are
// indistinguishable from a size or version suffix ("qwen-32b-2507" style
// names), so stripping them would let an unrelated model inherit a price. A
// name that needs that mapping gets an explicit alias in the sheet, where a
// human made the decision.
func stripDatedSuffix(model string) (string, bool) {
	// "-YYYY-MM-DD" is 11 characters.
	if len(model) > 11 {
		s := model[len(model)-11:]
		if s[0] == '-' && s[5] == '-' && s[8] == '-' &&
			isYear(s[1:5]) && allDigits(s[6:8]) && allDigits(s[9:11]) {
			return model[:len(model)-11], true
		}
	}
	// "-YYYYMMDD" is 9 characters.
	if len(model) > 9 {
		s := model[len(model)-9:]
		if s[0] == '-' && isYear(s[1:5]) && allDigits(s[5:9]) {
			return model[:len(model)-9], true
		}
	}
	return model, false
}

// isYear requires a 20xx year, which keeps the suffix rule from firing on an
// arbitrary 4-digit group such as a parameter count.
func isYear(s string) bool { return len(s) == 4 && s[0] == '2' && s[1] == '0' && allDigits(s[2:]) }

func allDigits(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return len(s) > 0
}

// Breakdown is a charge explained line by line: every component, and the token
// count it was derived from.
//
// The point of carrying the counts alongside the money is that a customer
// dispute is answered by arithmetic, not by trust. Given a Breakdown and the
// Rates it was computed from, anyone can recompute the total by hand.
type Breakdown struct {
	// Model is the requested model name; PricedAs and Rule say how it was
	// resolved, and are copied from the Match so a stored Breakdown is
	// self-contained.
	Model    string
	PricedAs string
	Rule     Rule
	Rates    Rates

	// InputTokens is prompt tokens billed at the full input rate, i.e.
	// prompt_tokens MINUS cached_tokens. See Rates.Cost.
	InputTokens int64
	// CachedInputTokens is prompt tokens billed at the cached rate.
	CachedInputTokens int64
	// OutputTokens is completion_tokens as reported, which already includes any
	// reasoning tokens.
	OutputTokens int64
	// ReasoningTokens is reported for visibility only. It is a SUBSET of
	// OutputTokens and contributes no cost component of its own; adding it would
	// double-bill a reasoning model.
	ReasoningTokens int64

	InputCost       money.Pico
	CachedInputCost money.Pico
	OutputCost      money.Pico
	// Total is InputCost + CachedInputCost + OutputCost, and nothing else.
	Total money.Pico
}

// PromptTokens returns the full prompt token count, reconstructed from the two
// billed components. It should equal the provider's prompt_tokens exactly, and
// the reconciliation report asserts that it does.
func (b Breakdown) PromptTokens() int64 { return b.InputTokens + b.CachedInputTokens }

// ComponentsSum returns the sum of the three cost components. Total is expected
// to equal it; the reconciliation check compares them so that a future edit
// which adds a fourth component without adding it to Total is caught.
func (b Breakdown) ComponentsSum() money.Pico {
	return b.InputCost + b.CachedInputCost + b.OutputCost
}

// ValidateUsage rejects a usage record that cannot be billed.
//
// The counts come from an upstream body, which in this gateway means either a
// third-party provider or (on the pass-through path) something a client could
// influence. Neither is trusted to be self-consistent, and every check here
// exists because the alternative is a wrong number rather than an error:
// negative counts would produce a negative charge, cached_tokens exceeding
// prompt_tokens would make the full-rate input component negative and *credit*
// the customer, and reasoning_tokens exceeding completion_tokens means the
// provider does not follow the OpenAI convention this cost model is built on,
// so its output count cannot be billed as-is.
func ValidateUsage(u *apiv1.Usage) error {
	if u == nil {
		return fmt.Errorf("%w: usage is absent", ErrInvalidUsage)
	}
	if u.PromptTokens < 0 || u.CompletionTokens < 0 {
		return fmt.Errorf("%w: negative token counts (prompt=%d completion=%d)",
			ErrInvalidUsage, u.PromptTokens, u.CompletionTokens)
	}
	cached := u.CachedPromptTokens()
	if cached < 0 {
		return fmt.Errorf("%w: negative cached_tokens (%d)", ErrInvalidUsage, cached)
	}
	if cached > u.PromptTokens {
		return fmt.Errorf(
			"%w: cached_tokens (%d) exceeds prompt_tokens (%d); cached tokens are a subset of the prompt",
			ErrInvalidUsage, cached, u.PromptTokens)
	}
	reasoning := u.ReasoningTokens()
	if reasoning < 0 {
		return fmt.Errorf("%w: negative reasoning_tokens (%d)", ErrInvalidUsage, reasoning)
	}
	if reasoning > u.CompletionTokens {
		return fmt.Errorf(
			"%w: reasoning_tokens (%d) exceeds completion_tokens (%d); OpenAI semantics require "+
				"completion_tokens to include reasoning tokens, so this provider's usage cannot be billed",
			ErrInvalidUsage, reasoning, u.CompletionTokens)
	}
	return nil
}

// Cost computes the charge for a usage record at these rates.
//
// Two provider conventions dominate this function, and getting either wrong
// changes the bill by an order of magnitude:
//
//   - prompt_tokens ALREADY INCLUDES prompt_tokens_details.cached_tokens. The
//     tokens billed at the full input rate are therefore
//     (prompt_tokens - cached_tokens), and the cached ones are billed once, at
//     the cached rate. Charging the whole prompt at the input rate and then the
//     cached tokens again at the cached rate over-bills every cache hit — which
//     is precisely the traffic a customer enabled caching to make cheaper.
//
//   - completion_tokens ALREADY INCLUDES
//     completion_tokens_details.reasoning_tokens. Reasoning tokens are recorded
//     in the Breakdown for visibility but are NOT a cost component; adding them
//     on top of completion_tokens roughly doubles the bill for a thinking model
//     and, worse, does so invisibly, since both numbers look plausible.
func (r Rates) Cost(u *apiv1.Usage) (Breakdown, error) {
	if err := ValidateUsage(u); err != nil {
		return Breakdown{}, err
	}
	cached := int64(u.CachedPromptTokens())
	fullRate := int64(u.PromptTokens) - cached
	output := int64(u.CompletionTokens)

	b := Breakdown{
		Rates:             r,
		InputTokens:       fullRate,
		CachedInputTokens: cached,
		OutputTokens:      output,
		ReasoningTokens:   int64(u.ReasoningTokens()),
	}
	var err error
	if b.InputCost, err = money.Cost(r.Input, fullRate); err != nil {
		return Breakdown{}, fmt.Errorf("input cost for %d tokens: %w", fullRate, err)
	}
	if b.CachedInputCost, err = money.Cost(r.CachedInput, cached); err != nil {
		return Breakdown{}, fmt.Errorf("cached input cost for %d tokens: %w", cached, err)
	}
	if b.OutputCost, err = money.Cost(r.Output, output); err != nil {
		return Breakdown{}, fmt.Errorf("output cost for %d tokens: %w", output, err)
	}
	total, err := money.Add(b.InputCost, b.CachedInputCost)
	if err != nil {
		return Breakdown{}, fmt.Errorf("summing prompt cost: %w", err)
	}
	if total, err = money.Add(total, b.OutputCost); err != nil {
		return Breakdown{}, fmt.Errorf("summing total cost: %w", err)
	}
	b.Total = total
	return b, nil
}

// Cost resolves the model and computes its charge in one step. This is the
// entry point the request path uses.
func (t *Table) Cost(model string, u *apiv1.Usage) (Breakdown, error) {
	m, err := t.Lookup(model)
	if err != nil {
		return Breakdown{}, err
	}
	b, err := m.Rates.Cost(u)
	if err != nil {
		return Breakdown{}, fmt.Errorf("model %q: %w", model, err)
	}
	b.Model, b.PricedAs, b.Rule = m.Requested, m.PricedAs, m.Rule
	return b, nil
}

// Estimate returns an upper bound on what a request will cost, for the budget
// pre-flight check that runs before the request is sent upstream.
//
// It is deliberately pessimistic in both directions where it has a choice: the
// whole prompt is priced at the full input rate because a cache hit cannot be
// predicted, and maxOutputTokens is assumed to be fully consumed. An estimator
// that could come in under the real cost would let a request slip past a budget
// that it then blows through, so the guarantee this function makes is
// Estimate >= actual cost whenever the completion respects the cap.
//
// maxOutputTokens must be supplied by the caller; a request with no client-set
// cap has an unbounded worst case, and inventing a default here would hide that
// policy decision inside the cost model.
func (t *Table) Estimate(model string, promptTokens, maxOutputTokens int64) (money.Pico, error) {
	if promptTokens < 0 || maxOutputTokens < 0 {
		return 0, fmt.Errorf("%w: negative estimate inputs (prompt=%d max_output=%d)",
			ErrInvalidUsage, promptTokens, maxOutputTokens)
	}
	m, err := t.Lookup(model)
	if err != nil {
		return 0, err
	}
	in, err := money.Cost(m.Rates.Input, promptTokens)
	if err != nil {
		return 0, fmt.Errorf("model %q: input estimate for %d tokens: %w", model, promptTokens, err)
	}
	out, err := money.Cost(m.Rates.Output, maxOutputTokens)
	if err != nil {
		return 0, fmt.Errorf("model %q: output estimate for %d tokens: %w", model, maxOutputTokens, err)
	}
	total, err := money.Add(in, out)
	if err != nil {
		return 0, fmt.Errorf("model %q: summing estimate: %w", model, err)
	}
	return total, nil
}

// ModelPrice is one row of a JSON price sheet.
//
// Prices are decimal strings, not JSON numbers. A float64 cannot hold "0.15"
// exactly, so a numeric field would mean the sheet's own representation
// introduced the rounding error this package refuses to tolerate — and it would
// do so before any of the validation below could see it. Strings go straight to
// money.ParseUSD, which is exact.
type ModelPrice struct {
	// Model is the canonical name, spelled as the vendor's pricing page does.
	Model string `json:"model"`
	// Aliases are additional names that price identically. They exist so an
	// operator can map a name the Lookup rules would refuse to guess at (a
	// 4-digit snapshot, a deployment name, a vendor-specific route) without
	// duplicating the prices.
	Aliases []string `json:"aliases,omitempty"`

	// InputPerMTok is the price of 1,000,000 uncached prompt tokens, e.g. "2.50".
	InputPerMTok string `json:"input_per_1m_tokens"`
	// CachedInputPerMTok is the price of 1,000,000 cached prompt tokens. When
	// omitted it defaults to InputPerMTok — no discount — because the other
	// plausible default, zero, would make cached prompts free and quietly
	// under-bill every cache hit. A provider that genuinely does not charge for
	// cache reads must say "0" explicitly.
	CachedInputPerMTok string `json:"cached_input_per_1m_tokens,omitempty"`
	// OutputPerMTok is the price of 1,000,000 completion tokens (which include
	// reasoning tokens).
	OutputPerMTok string `json:"output_per_1m_tokens"`

	// Note is free-form provenance: where and when the price was captured.
	Note string `json:"note,omitempty"`
}

// Sheet is a loadable price sheet.
//
// Models is an array and not an object keyed by model name, which looks like
// the more natural encoding until you consider duplicates: encoding/json
// silently keeps the last value for a repeated object key, so a sheet listing
// gpt-4o twice — the classic result of a bad config merge — would load without
// complaint at whichever price happened to come second. An array makes the
// duplicate visible, and Table() rejects it.
type Sheet struct {
	// Currency must be "USD" if present. The money package is USD-only, so a
	// sheet quoted in anything else would be billed as if it were dollars.
	Currency string `json:"currency,omitempty"`
	// Source is free-form provenance for the sheet as a whole.
	Source string       `json:"source,omitempty"`
	Models []ModelPrice `json:"models"`
}

// Load reads and validates a JSON price sheet.
//
// Unknown fields are rejected. A typo in a price key ("cached_input_per_1m")
// would otherwise be dropped by the decoder and silently fall back to the
// default rate, which is exactly the class of config error that shows up as a
// wrong invoice weeks later.
func Load(r io.Reader) (*Table, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var s Sheet
	if err := dec.Decode(&s); err != nil {
		return nil, fmt.Errorf("pricing: decoding price sheet: %w", err)
	}
	// Reject trailing content rather than ignoring it: two concatenated JSON
	// documents means the operator's second sheet is not being applied.
	if err := expectEOF(dec); err != nil {
		return nil, err
	}
	return s.Table()
}

func expectEOF(dec *json.Decoder) error {
	if _, err := dec.Token(); err != io.EOF {
		if err == nil {
			return errors.New("pricing: unexpected trailing content after price sheet")
		}
		return fmt.Errorf("pricing: unexpected trailing content after price sheet: %w", err)
	}
	return nil
}

// LoadFile reads a JSON price sheet from disk.
func LoadFile(path string) (*Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("pricing: %w", err)
	}
	defer f.Close()
	t, err := Load(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return t, nil
}

// Table validates the sheet and builds an immutable lookup table. Every error
// names the offending model, because "invalid price" in a startup log with
// forty models configured is not an actionable message.
func (s *Sheet) Table() (*Table, error) {
	if s.Currency != "" && !strings.EqualFold(s.Currency, "USD") {
		return nil, fmt.Errorf("pricing: currency %q is not supported; this gateway prices in USD only",
			s.Currency)
	}
	if len(s.Models) == 0 {
		return nil, errors.New("pricing: price sheet contains no models")
	}
	t := &Table{entries: make(map[string]entry, len(s.Models)*2)}
	canonical := make(map[string]struct{}, len(s.Models))
	for i := range s.Models {
		mp := &s.Models[i]
		name := strings.TrimSpace(mp.Model)
		if name == "" {
			return nil, fmt.Errorf("pricing: models[%d] has an empty 'model' name", i)
		}
		if _, dup := canonical[name]; dup {
			return nil, fmt.Errorf("pricing: model %q is listed more than once", name)
		}
		if prior, taken := t.entries[name]; taken {
			return nil, fmt.Errorf("pricing: model %q collides with an alias of %q", name, prior.canonical)
		}
		rates, err := mp.rates()
		if err != nil {
			return nil, fmt.Errorf("pricing: model %q: %w", name, err)
		}
		canonical[name] = struct{}{}
		t.entries[name] = entry{canonical: name, rates: rates}
		t.models = append(t.models, name)
		for _, alias := range mp.Aliases {
			a := strings.TrimSpace(alias)
			if a == "" {
				return nil, fmt.Errorf("pricing: model %q has an empty alias", name)
			}
			if prior, taken := t.entries[a]; taken {
				return nil, fmt.Errorf("pricing: model %q: alias %q is already used by %q",
					name, a, prior.canonical)
			}
			t.entries[a] = entry{canonical: name, rates: rates}
		}
	}
	sort.Strings(t.models)
	return t, nil
}

// rates parses and validates one row's prices.
func (mp *ModelPrice) rates() (Rates, error) {
	if strings.TrimSpace(mp.InputPerMTok) == "" {
		return Rates{}, errors.New("'input_per_1m_tokens' is required")
	}
	if strings.TrimSpace(mp.OutputPerMTok) == "" {
		return Rates{}, errors.New("'output_per_1m_tokens' is required")
	}
	in, err := money.ParseUSD(mp.InputPerMTok)
	if err != nil {
		return Rates{}, fmt.Errorf("'input_per_1m_tokens' = %q: %w", mp.InputPerMTok, err)
	}
	out, err := money.ParseUSD(mp.OutputPerMTok)
	if err != nil {
		return Rates{}, fmt.Errorf("'output_per_1m_tokens' = %q: %w", mp.OutputPerMTok, err)
	}
	// Default the cached rate to the full input rate, never to zero.
	cached := in
	cachedRaw := mp.InputPerMTok
	if strings.TrimSpace(mp.CachedInputPerMTok) != "" {
		cachedRaw = mp.CachedInputPerMTok
		if cached, err = money.ParseUSD(mp.CachedInputPerMTok); err != nil {
			return Rates{}, fmt.Errorf("'cached_input_per_1m_tokens' = %q: %w", mp.CachedInputPerMTok, err)
		}
	}
	// Negative prices are checked here, before PerToken, so the message blames
	// the field an operator can find rather than the division.
	for _, f := range []struct {
		field string
		raw   string
		value money.Pico
	}{
		{"input_per_1m_tokens", mp.InputPerMTok, in},
		{"cached_input_per_1m_tokens", cachedRaw, cached},
		{"output_per_1m_tokens", mp.OutputPerMTok, out},
	} {
		if f.value < 0 {
			return Rates{}, fmt.Errorf("'%s' = %q is negative; prices must be >= 0", f.field, f.raw)
		}
	}
	return NewRates(in, cached, out)
}

// defaultSheet is a demonstration price sheet.
//
// LIST PRICES CAPTURED FOR DEMONSTRATION IN 2026. NOT AUTHORITATIVE, NOT
// CONTRACTUAL, AND CERTAIN TO BE STALE: vendors reprice without notice, and
// negotiated and regional rates differ from the public list. A real deployment
// loads its own sheet with LoadFile and treats this as a smoke-test fixture
// only. It is here so the gateway starts up with a plausible cost model instead
// of an empty table that would make every model unpriced.
var defaultSheet = Sheet{
	Currency: "USD",
	Source:   "public list prices, captured 2026 for demonstration only; not authoritative",
	Models: []ModelPrice{
		{
			Model:              "gpt-5",
			InputPerMTok:       "1.25",
			CachedInputPerMTok: "0.125",
			OutputPerMTok:      "10.00",
			Note:               "demo list price",
		},
		{
			Model:              "gpt-5-mini",
			InputPerMTok:       "0.25",
			CachedInputPerMTok: "0.025",
			OutputPerMTok:      "2.00",
			Note:               "demo list price",
		},
		{
			Model:              "gpt-4o",
			InputPerMTok:       "2.50",
			CachedInputPerMTok: "1.25",
			OutputPerMTok:      "10.00",
			Note:               "demo list price",
		},
		{
			Model:              "gpt-4o-mini",
			InputPerMTok:       "0.15",
			CachedInputPerMTok: "0.075",
			OutputPerMTok:      "0.60",
			Note: "demo list price; the cheapest rate here is 75 pico/token, " +
				"which is why money uses picodollars",
		},
		{
			Model:              "claude-sonnet-4-5",
			Aliases:            []string{"claude-sonnet-4-5-20250929"},
			InputPerMTok:       "3.00",
			CachedInputPerMTok: "0.30",
			OutputPerMTok:      "15.00",
			Note:               "demo list price; cached rate is the cache-read rate",
		},
		{
			Model:              "claude-haiku-4-5",
			InputPerMTok:       "1.00",
			CachedInputPerMTok: "0.10",
			OutputPerMTok:      "5.00",
			Note:               "demo list price",
		},
		{
			Model:              "gemini-2.5-flash",
			InputPerMTok:       "0.30",
			CachedInputPerMTok: "0.075",
			OutputPerMTok:      "2.50",
			Note:               "demo list price",
		},
		{
			Model:              "gemini-2.5-pro",
			InputPerMTok:       "1.25",
			CachedInputPerMTok: "0.3125",
			OutputPerMTok:      "10.00",
			Note:               "demo list price; 0.3125 exercises a 4-decimal price that is still exact",
		},
	},
}

// defaultTable is built once, lazily, and shared. Building it cannot fail
// unless defaultSheet above is edited into an invalid state, which is a
// programming error and is caught by TestDefaultTable rather than at runtime.
var defaultTable = sync.OnceValue(func() *Table {
	t, err := defaultSheet.Table()
	if err != nil {
		panic("pricing: built-in demonstration sheet is invalid: " + err.Error())
	}
	return t
})

// DefaultTable returns the shared demonstration price sheet. The returned Table
// is immutable and safe to share across goroutines.
func DefaultTable() *Table { return defaultTable() }

// DefaultSheet returns a deep copy of the demonstration sheet, so an operator
// can dump it as JSON and use it as the starting point for a real one. The copy
// is deep down to the Aliases slices: a shallow copy would let a caller's edit
// of one row's aliases reach back into the package-level sheet, which
// DefaultTable also reads.
func DefaultSheet() Sheet {
	s := defaultSheet
	s.Models = make([]ModelPrice, len(defaultSheet.Models))
	copy(s.Models, defaultSheet.Models)
	for i := range s.Models {
		if a := s.Models[i].Aliases; a != nil {
			s.Models[i].Aliases = append([]string(nil), a...)
		}
	}
	return s
}
