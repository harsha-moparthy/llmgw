package pricing

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/money"
)

// usd parses a decimal dollar string or fails the test. Used so the expected
// values in the tables below are written the way a pricing page writes them,
// which makes a wrong expectation visible on inspection.
func usd(t *testing.T, s string) money.Pico {
	t.Helper()
	p, err := money.ParseUSD(s)
	if err != nil {
		t.Fatalf("ParseUSD(%q): %v", s, err)
	}
	return p
}

// mustRates builds rates from per-million list prices.
func mustRates(t *testing.T, in, cached, out string) Rates {
	t.Helper()
	r, err := NewRates(usd(t, in), usd(t, cached), usd(t, out))
	if err != nil {
		t.Fatalf("NewRates(%s,%s,%s): %v", in, cached, out, err)
	}
	return r
}

func TestNewRates(t *testing.T) {
	tests := []struct {
		name              string
		in, cached, out   string
		wantErr           bool
		wantErrContains   string
		wantIn, wantCache money.Pico
		wantOut           money.Pico
	}{
		{
			name: "typical prices divide exactly",
			in:   "2.50", cached: "1.25", out: "10.00",
			wantIn: 2_500_000, wantCache: 1_250_000, wantOut: 10_000_000,
		},
		{
			name: "cheapest realistic price is still an exact integer",
			in:   "0.15", cached: "0.075", out: "0.60",
			// $0.15 per 1M tokens = 1.5e11 pico / 1e6 = 150000 pico/token.
			wantIn: 150_000, wantCache: 75_000, wantOut: 600_000,
		},
		{
			name: "free is a legitimate price",
			in:   "0", cached: "0", out: "0",
		},
		{
			name: "six decimal places is the exactness boundary",
			in:   "0.000001", cached: "0", out: "1",
			wantIn: 1, wantOut: 1_000_000,
		},
		{
			name: "seven decimal places is rejected rather than rounded",
			in:   "0.0000001", cached: "0", out: "1",
			wantErr: true, wantErrContains: "input price",
		},
		{
			name: "an inexact cached price is rejected and named",
			in:   "1", cached: "0.0000005", out: "1",
			wantErr: true, wantErrContains: "cached input price",
		},
		{
			name: "an inexact output price is rejected and named",
			in:   "1", cached: "1", out: "0.00000025",
			wantErr: true, wantErrContains: "output price",
		},
		{
			name: "a negative price is rejected",
			in:   "-1", cached: "0", out: "1",
			wantErr: true, wantErrContains: "negative",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NewRates(usd(t, tc.in), usd(t, tc.cached), usd(t, tc.out))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected an error, got rates %+v", got)
				}
				if !strings.Contains(err.Error(), tc.wantErrContains) {
					t.Fatalf("error %q does not mention %q", err, tc.wantErrContains)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.Input != tc.wantIn || got.CachedInput != tc.wantCache || got.Output != tc.wantOut {
				t.Fatalf("rates = %+v, want {Input:%d CachedInput:%d Output:%d}",
					got, tc.wantIn, tc.wantCache, tc.wantOut)
			}
		})
	}
}

// testTable is a small, fully explicit sheet. Tests use it rather than
// DefaultTable so that a change to the demonstration prices cannot break the
// arithmetic assertions.
func testTable(t *testing.T) *Table {
	t.Helper()
	s := Sheet{
		Currency: "USD",
		Models: []ModelPrice{
			{
				Model:              "gpt-4o",
				InputPerMTok:       "2.50",
				CachedInputPerMTok: "1.25",
				OutputPerMTok:      "10.00",
			},
			{
				Model:              "thinker-1",
				InputPerMTok:       "1.00",
				CachedInputPerMTok: "0.10",
				OutputPerMTok:      "100.00", // deliberately lopsided: output errors are loud
			},
			{
				Model:         "no-cache-discount",
				InputPerMTok:  "3.00",
				OutputPerMTok: "6.00",
			},
			{
				Model:              "legacy",
				Aliases:            []string{"legacy-0613", "azure-legacy-deployment"},
				InputPerMTok:       "0.50",
				CachedInputPerMTok: "0.50",
				OutputPerMTok:      "1.50",
			},
		},
	}
	tbl, err := s.Table()
	if err != nil {
		t.Fatalf("building test table: %v", err)
	}
	return tbl
}

func TestLookup(t *testing.T) {
	tbl := testTable(t)
	tests := []struct {
		name       string
		model      string
		wantErr    bool
		wantPriced string
		wantRule   Rule
	}{
		{name: "exact", model: "gpt-4o", wantPriced: "gpt-4o", wantRule: RuleExact},
		{name: "alias is an exact hit on the alias name", model: "legacy-0613",
			wantPriced: "legacy", wantRule: RuleExact},
		{name: "alias with a deployment-style name", model: "azure-legacy-deployment",
			wantPriced: "legacy", wantRule: RuleExact},
		{name: "single routing prefix", model: "openai/gpt-4o",
			wantPriced: "gpt-4o", wantRule: RuleVendorPrefix},
		{name: "nested routing prefix strips to the last segment", model: "openrouter/openai/gpt-4o",
			wantPriced: "gpt-4o", wantRule: RuleVendorPrefix},
		{name: "dashed dated snapshot", model: "gpt-4o-2024-08-06",
			wantPriced: "gpt-4o", wantRule: RuleDatedSnapshot},
		{name: "compact dated snapshot", model: "gpt-4o-20240806",
			wantPriced: "gpt-4o", wantRule: RuleDatedSnapshot},
		{name: "prefix and snapshot together", model: "openai/gpt-4o-2024-08-06",
			wantPriced: "gpt-4o", wantRule: RuleVendorPrefixAndDatedSnapshot},
		{name: "unknown model", model: "gpt-9-ultra", wantErr: true},
		{name: "case differences are not guessed at", model: "GPT-4o", wantErr: true},
		{name: "a prefix of a known model is not a match", model: "gpt-4", wantErr: true},
		{name: "a known model as a prefix of the request is not a match", model: "gpt-4o-audio", wantErr: true},
		{name: "trailing slash is not a prefix strip", model: "gpt-4o/", wantErr: true},
		{name: "empty model", model: "", wantErr: true},
		{
			// The 4-digit-suffix case documented on stripDatedSuffix: it must NOT
			// be stripped, or an unrelated size/version suffix could inherit a
			// price. "legacy-0613" only resolves because it is an explicit alias.
			name: "four-digit suffix is not treated as a date", model: "gpt-4o-0613", wantErr: true,
		},
		{
			// A 19xx "year" must not trip the suffix rule.
			name: "non-20xx date-shaped suffix is not stripped", model: "gpt-4o-1999-08-06", wantErr: true,
		},
		{
			name: "date-shaped suffix with letters is not stripped", model: "gpt-4o-2024-08-0x", wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, err := tbl.Lookup(tc.model)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected an unpriced error, got match %+v", m)
				}
				if !errors.Is(err, ErrUnpriced) {
					t.Fatalf("error %v does not match ErrUnpriced", err)
				}
				var ue *UnpricedError
				if !errors.As(err, &ue) {
					t.Fatalf("error %v is not an *UnpricedError", err)
				}
				if ue.Model != tc.model {
					t.Fatalf("UnpricedError.Model = %q, want %q", ue.Model, tc.model)
				}
				// The single worst failure mode: an unknown model must not
				// resolve to a zero-rate match that bills nothing.
				if m.Rates != (Rates{}) || m.PricedAs != "" {
					t.Fatalf("failed lookup returned a usable match: %+v", m)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if m.Requested != tc.model {
				t.Fatalf("Requested = %q, want %q", m.Requested, tc.model)
			}
			if m.PricedAs != tc.wantPriced {
				t.Fatalf("PricedAs = %q, want %q", m.PricedAs, tc.wantPriced)
			}
			if m.Rule != tc.wantRule {
				t.Fatalf("Rule = %v, want %v", m.Rule, tc.wantRule)
			}
			if m.Rates == (Rates{}) {
				t.Fatal("match carries zero rates")
			}
		})
	}
}

// TestUnknownModelCostIsAnErrorNotZero is the anti-regression for the whole
// point of the unpriced path: it asserts Cost refuses, rather than returning a
// zero Breakdown that a caller would happily write to the ledger.
func TestUnknownModelCostIsAnErrorNotZero(t *testing.T) {
	tbl := testTable(t)
	u := &apiv1.Usage{PromptTokens: 1000, CompletionTokens: 500, TotalTokens: 1500}

	b, err := tbl.Cost("some-model-nobody-configured", u)
	if err == nil {
		t.Fatalf("unknown model priced successfully as %+v (total %s)", b, money.FormatUSD(b.Total))
	}
	if !errors.Is(err, ErrUnpriced) {
		t.Fatalf("error %v does not match ErrUnpriced", err)
	}
	if b.Total != 0 || b.OutputTokens != 0 {
		t.Fatalf("expected a zero-value Breakdown alongside the error, got %+v", b)
	}
	if _, err := tbl.Estimate("some-model-nobody-configured", 10, 10); !errors.Is(err, ErrUnpriced) {
		t.Fatalf("Estimate on an unknown model: err = %v, want ErrUnpriced", err)
	}
	// A nil table must behave the same way rather than pricing everything free.
	var nilTable *Table
	if _, err := nilTable.Lookup("gpt-4o"); !errors.Is(err, ErrUnpriced) {
		t.Fatalf("nil table Lookup: err = %v, want ErrUnpriced", err)
	}
	if nilTable.Len() != 0 || nilTable.Models() != nil {
		t.Fatal("nil table should report no models")
	}
}

func TestUnpricedErrorMessageListsCandidates(t *testing.T) {
	tbl := testTable(t)
	_, err := tbl.Lookup("openai/mystery-2024-08-06")
	if err == nil {
		t.Fatal("expected an error")
	}
	var ue *UnpricedError
	if !errors.As(err, &ue) {
		t.Fatalf("not an *UnpricedError: %v", err)
	}
	want := []string{"mystery-2024-08-06", "openai/mystery", "mystery"}
	if len(ue.Tried) != len(want) {
		t.Fatalf("Tried = %q, want %q", ue.Tried, want)
	}
	for i := range want {
		if ue.Tried[i] != want[i] {
			t.Fatalf("Tried = %q, want %q", ue.Tried, want)
		}
	}
	for _, w := range want {
		if !strings.Contains(err.Error(), w) {
			t.Fatalf("message %q omits candidate %q", err, w)
		}
	}
}

// TestCostReasoningTokensAreNotDoubleBilled is trap (a).
//
// OpenAI's completion_tokens ALREADY INCLUDES reasoning tokens. The expected
// output charge is therefore completion_tokens * output rate, full stop. If the
// implementation ever adds ReasoningTokens on top, the "reasoning is a large
// fraction of the completion" case below charges 1800 tokens instead of 1000
// and this test fails.
func TestCostReasoningTokensAreNotDoubleBilled(t *testing.T) {
	tbl := testTable(t)
	rates := mustRates(t, "1.00", "0.10", "100.00")

	tests := []struct {
		name             string
		completion       int
		reasoning        int
		wantOutputTokens int64
	}{
		{name: "no reasoning details", completion: 1000, reasoning: 0, wantOutputTokens: 1000},
		{name: "reasoning is most of the completion", completion: 1000, reasoning: 800, wantOutputTokens: 1000},
		{name: "reasoning is the entire completion", completion: 1000, reasoning: 1000, wantOutputTokens: 1000},
		{name: "reasoning with a tiny visible answer", completion: 4096, reasoning: 4090, wantOutputTokens: 4096},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			u := &apiv1.Usage{
				PromptTokens:     0,
				CompletionTokens: tc.completion,
				CompletionTokensDetails: &apiv1.CompletionTokensDetails{
					ReasoningTokens: tc.reasoning,
				},
			}
			b, err := tbl.Cost("thinker-1", u)
			if err != nil {
				t.Fatalf("Cost: %v", err)
			}
			if b.OutputTokens != tc.wantOutputTokens {
				t.Fatalf("OutputTokens = %d, want %d (completion_tokens already includes reasoning)",
					b.OutputTokens, tc.wantOutputTokens)
			}
			// Reasoning must still be reported, or the breakdown loses the
			// visibility that makes a thinking model's bill explainable.
			if b.ReasoningTokens != int64(tc.reasoning) {
				t.Fatalf("ReasoningTokens = %d, want %d", b.ReasoningTokens, tc.reasoning)
			}
			wantCost, err := money.Mul(rates.Output, tc.wantOutputTokens)
			if err != nil {
				t.Fatal(err)
			}
			if b.OutputCost != wantCost {
				t.Fatalf("OutputCost = %s, want %s", money.FormatUSD(b.OutputCost), money.FormatUSD(wantCost))
			}
			if b.Total != wantCost {
				t.Fatalf("Total = %s, want %s (reasoning must add no component)",
					money.FormatUSD(b.Total), money.FormatUSD(wantCost))
			}
		})
	}
}

// TestCostCachedTokensAreNotDoubleBilled is trap (b).
//
// prompt_tokens ALREADY INCLUDES cached_tokens, so the full-rate component is
// (prompt - cached). The naive implementation charges prompt*input +
// cached*cached_rate; each case below has a distinct expected total under the
// correct rule, and the fully-cached case is the sharpest: it must cost the
// cached rate alone, not the input rate plus the cached rate.
func TestCostCachedTokensAreNotDoubleBilled(t *testing.T) {
	tbl := testTable(t)
	// gpt-4o in the test sheet: input $2.50/M, cached $1.25/M, output $10.00/M.
	const inRate, cachedRate, outRate = 2_500_000, 1_250_000, 10_000_000

	tests := []struct {
		name              string
		prompt, cached    int
		completion        int
		wantFullTokens    int64
		wantCachedTokens  int64
		wantTotal         money.Pico
		wantNaiveOverbill bool // documents that the naive rule differs here
	}{
		{
			name:   "no cache details at all",
			prompt: 1000, cached: 0, completion: 200,
			wantFullTokens: 1000, wantCachedTokens: 0,
			wantTotal: 1000*inRate + 200*outRate,
		},
		{
			name:   "partial cache hit",
			prompt: 1000, cached: 800, completion: 200,
			wantFullTokens: 200, wantCachedTokens: 800,
			wantTotal:         200*inRate + 800*cachedRate + 200*outRate,
			wantNaiveOverbill: true,
		},
		{
			name:   "fully cached prompt costs the cached rate only",
			prompt: 1000, cached: 1000, completion: 0,
			wantFullTokens: 0, wantCachedTokens: 1000,
			wantTotal:         1000 * cachedRate,
			wantNaiveOverbill: true,
		},
		{
			name:   "one cached token",
			prompt: 1000, cached: 1, completion: 1,
			wantFullTokens: 999, wantCachedTokens: 1,
			wantTotal:         999*inRate + 1*cachedRate + 1*outRate,
			wantNaiveOverbill: true,
		},
		{
			name:   "empty usage costs nothing",
			prompt: 0, cached: 0, completion: 0,
			wantFullTokens: 0, wantCachedTokens: 0, wantTotal: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			u := &apiv1.Usage{
				PromptTokens:     tc.prompt,
				CompletionTokens: tc.completion,
				TotalTokens:      tc.prompt + tc.completion,
			}
			if tc.cached > 0 {
				u.PromptTokensDetails = &apiv1.PromptTokensDetails{CachedTokens: tc.cached}
			}
			b, err := tbl.Cost("gpt-4o", u)
			if err != nil {
				t.Fatalf("Cost: %v", err)
			}
			if b.InputTokens != tc.wantFullTokens {
				t.Fatalf("InputTokens = %d, want %d (prompt minus cached)", b.InputTokens, tc.wantFullTokens)
			}
			if b.CachedInputTokens != tc.wantCachedTokens {
				t.Fatalf("CachedInputTokens = %d, want %d", b.CachedInputTokens, tc.wantCachedTokens)
			}
			if b.Total != tc.wantTotal {
				t.Fatalf("Total = %s, want %s", money.FormatUSD(b.Total), money.FormatUSD(tc.wantTotal))
			}
			// The two billed prompt components must reconstruct prompt_tokens
			// exactly, which is the invariant reconciliation depends on.
			if b.PromptTokens() != int64(tc.prompt) {
				t.Fatalf("PromptTokens() = %d, want %d", b.PromptTokens(), tc.prompt)
			}
			if b.ComponentsSum() != b.Total {
				t.Fatalf("components sum %s != Total %s",
					money.FormatUSD(b.ComponentsSum()), money.FormatUSD(b.Total))
			}
			naive := money.Pico(int64(tc.prompt)*inRate + int64(tc.cached)*cachedRate +
				int64(tc.completion)*outRate)
			if tc.wantNaiveOverbill && naive <= b.Total {
				t.Fatalf("test case is not discriminating: naive total %s <= correct total %s",
					money.FormatUSD(naive), money.FormatUSD(b.Total))
			}
		})
	}
}

func TestValidateUsage(t *testing.T) {
	tests := []struct {
		name    string
		usage   *apiv1.Usage
		wantErr string // substring; empty means the record must be accepted
	}{
		{
			name:  "plain usage",
			usage: &apiv1.Usage{PromptTokens: 10, CompletionTokens: 5},
		},
		{
			name: "cached equal to prompt is legal",
			usage: &apiv1.Usage{
				PromptTokens:        10,
				PromptTokensDetails: &apiv1.PromptTokensDetails{CachedTokens: 10},
			},
		},
		{
			name: "reasoning equal to completion is legal",
			usage: &apiv1.Usage{
				CompletionTokens:        7,
				CompletionTokensDetails: &apiv1.CompletionTokensDetails{ReasoningTokens: 7},
			},
		},
		{
			name:  "all zeroes is legal",
			usage: &apiv1.Usage{},
		},
		{
			name:    "nil usage",
			usage:   nil,
			wantErr: "usage is absent",
		},
		{
			name:    "negative prompt tokens",
			usage:   &apiv1.Usage{PromptTokens: -1},
			wantErr: "negative token counts",
		},
		{
			name:    "negative completion tokens",
			usage:   &apiv1.Usage{CompletionTokens: -5},
			wantErr: "negative token counts",
		},
		{
			name: "cached exceeds prompt by one",
			usage: &apiv1.Usage{
				PromptTokens:        10,
				PromptTokensDetails: &apiv1.PromptTokensDetails{CachedTokens: 11},
			},
			wantErr: "exceeds prompt_tokens",
		},
		{
			name: "cached present with a zero prompt",
			usage: &apiv1.Usage{
				PromptTokens:        0,
				PromptTokensDetails: &apiv1.PromptTokensDetails{CachedTokens: 500},
			},
			wantErr: "exceeds prompt_tokens",
		},
		{
			name: "negative cached tokens",
			usage: &apiv1.Usage{
				PromptTokens:        10,
				PromptTokensDetails: &apiv1.PromptTokensDetails{CachedTokens: -3},
			},
			wantErr: "negative cached_tokens",
		},
		{
			name: "reasoning exceeds completion",
			usage: &apiv1.Usage{
				CompletionTokens:        10,
				CompletionTokensDetails: &apiv1.CompletionTokensDetails{ReasoningTokens: 11},
			},
			wantErr: "exceeds completion_tokens",
		},
		{
			name: "negative reasoning tokens",
			usage: &apiv1.Usage{
				CompletionTokens:        10,
				CompletionTokensDetails: &apiv1.CompletionTokensDetails{ReasoningTokens: -1},
			},
			wantErr: "negative reasoning_tokens",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateUsage(tc.usage)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("valid usage rejected: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected an error")
			}
			if !errors.Is(err, ErrInvalidUsage) {
				t.Fatalf("error %v does not match ErrInvalidUsage", err)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not mention %q", err, tc.wantErr)
			}
		})
	}
}

// TestCostRejectsSpoofedCacheCreditRatherThanCreditingIt is the boundary that
// matters most about the cached-tokens check: without it, a body claiming more
// cached tokens than prompt tokens produces a NEGATIVE input component, which
// nets out as a credit on the invoice.
func TestCostRejectsSpoofedCacheCreditRatherThanCreditingIt(t *testing.T) {
	tbl := testTable(t)
	u := &apiv1.Usage{
		PromptTokens:        10,
		CompletionTokens:    0,
		PromptTokensDetails: &apiv1.PromptTokensDetails{CachedTokens: 1_000_000},
	}
	b, err := tbl.Cost("gpt-4o", u)
	if err == nil {
		t.Fatalf("spoofed usage priced successfully: total %s", money.FormatUSD(b.Total))
	}
	if !errors.Is(err, ErrInvalidUsage) {
		t.Fatalf("error %v does not match ErrInvalidUsage", err)
	}
	if !strings.Contains(err.Error(), "gpt-4o") {
		t.Fatalf("error %q should name the model", err)
	}
	// Sanity: had the check been absent, the input component would have been
	// negative. Prove the arithmetic that the check is protecting against.
	rates := mustRates(t, "2.50", "1.25", "10.00")
	unchecked, mulErr := money.Mul(rates.Input, 10)
	if mulErr != nil {
		t.Fatal(mulErr)
	}
	spoofed := unchecked - money.Pico(1_000_000)*rates.Input
	if spoofed >= 0 {
		t.Fatal("test case is not discriminating: the unchecked charge is not negative")
	}
}

func TestCostReportsOverflowInsteadOfWrapping(t *testing.T) {
	tbl := testTable(t)
	tests := []struct {
		name  string
		usage *apiv1.Usage
		field string
	}{
		{
			name:  "absurd prompt",
			usage: &apiv1.Usage{PromptTokens: 1 << 55},
			field: "input cost",
		},
		{
			name: "absurd cached prompt",
			usage: &apiv1.Usage{
				PromptTokens:        1 << 55,
				PromptTokensDetails: &apiv1.PromptTokensDetails{CachedTokens: 1 << 55},
			},
			field: "cached input cost",
		},
		{
			name:  "absurd completion",
			usage: &apiv1.Usage{CompletionTokens: 1 << 55},
			field: "output cost",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b, err := tbl.Cost("gpt-4o", tc.usage)
			if err == nil {
				t.Fatalf("expected overflow, got total %s (%d pico)", money.FormatUSD(b.Total), b.Total)
			}
			if !errors.Is(err, money.ErrOverflow) {
				t.Fatalf("error %v does not match money.ErrOverflow", err)
			}
			if !strings.Contains(err.Error(), tc.field) {
				t.Fatalf("error %q does not identify the component %q", err, tc.field)
			}
			if b.Total != 0 {
				t.Fatalf("expected a zero Breakdown on overflow, got total %d", b.Total)
			}
		})
	}
}

func TestBreakdownCarriesResolutionProvenance(t *testing.T) {
	tbl := testTable(t)
	b, err := tbl.Cost("openai/gpt-4o-2024-08-06", &apiv1.Usage{PromptTokens: 4, CompletionTokens: 2})
	if err != nil {
		t.Fatalf("Cost: %v", err)
	}
	if b.Model != "openai/gpt-4o-2024-08-06" {
		t.Fatalf("Model = %q, want the requested name", b.Model)
	}
	if b.PricedAs != "gpt-4o" {
		t.Fatalf("PricedAs = %q, want %q", b.PricedAs, "gpt-4o")
	}
	if b.Rule != RuleVendorPrefixAndDatedSnapshot {
		t.Fatalf("Rule = %v, want %v", b.Rule, RuleVendorPrefixAndDatedSnapshot)
	}
	if b.Rule.Exact() {
		t.Fatal("a normalised match must not claim to be exact")
	}
	if b.Rates != (Rates{Input: 2_500_000, CachedInput: 1_250_000, Output: 10_000_000}) {
		t.Fatalf("Rates = %+v, want the gpt-4o rates", b.Rates)
	}
	// A stored Breakdown must be recomputable from its own fields.
	recomputed := money.Pico(b.InputTokens)*b.Rates.Input +
		money.Pico(b.CachedInputTokens)*b.Rates.CachedInput +
		money.Pico(b.OutputTokens)*b.Rates.Output
	if recomputed != b.Total {
		t.Fatalf("hand-recomputed total %d != Total %d", recomputed, b.Total)
	}
}

func TestRuleStrings(t *testing.T) {
	tests := []struct {
		rule Rule
		want string
	}{
		{RuleExact, "exact"},
		{RuleVendorPrefix, "vendor_prefix_stripped"},
		{RuleDatedSnapshot, "dated_snapshot_stripped"},
		{RuleVendorPrefixAndDatedSnapshot, "vendor_prefix_and_dated_snapshot_stripped"},
		{Rule(99), "invalid"},
	}
	seen := make(map[string]bool, len(tests))
	for _, tc := range tests {
		if got := tc.rule.String(); got != tc.want {
			t.Fatalf("Rule(%d).String() = %q, want %q", tc.rule, got, tc.want)
		}
		if seen[tc.want] {
			t.Fatalf("duplicate rule label %q would make metrics ambiguous", tc.want)
		}
		seen[tc.want] = true
	}
	if !RuleExact.Exact() {
		t.Fatal("RuleExact must be exact")
	}
}

// TestEstimateNeverUnderchargesTheRealCost is the test that would catch an
// estimator that is "never wrong because it is never called": it exercises
// Estimate against the actual Cost of a spread of usages, including the case
// that would break a naive estimator (a cache hit, which is cheaper than the
// estimate assumed) and the case a wrong estimator would get away with (no
// caching at all, where estimate and cost coincide exactly).
func TestEstimateNeverUnderchargesTheRealCost(t *testing.T) {
	tbl := testTable(t)
	const promptTokens, maxOut = 1000, 500

	est, err := tbl.Estimate("gpt-4o", promptTokens, maxOut)
	if err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	tests := []struct {
		name       string
		cached     int
		completion int
		wantEqual  bool // the estimate should be exactly the cost
	}{
		{name: "worst case: no cache, cap fully consumed", cached: 0, completion: maxOut, wantEqual: true},
		{name: "short completion", cached: 0, completion: 1},
		{name: "cache hit makes the real cost lower", cached: 900, completion: maxOut},
		{name: "fully cached and short", cached: promptTokens, completion: 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			u := &apiv1.Usage{PromptTokens: promptTokens, CompletionTokens: tc.completion}
			if tc.cached > 0 {
				u.PromptTokensDetails = &apiv1.PromptTokensDetails{CachedTokens: tc.cached}
			}
			b, err := tbl.Cost("gpt-4o", u)
			if err != nil {
				t.Fatalf("Cost: %v", err)
			}
			if est < b.Total {
				t.Fatalf("estimate %s under-charges the real cost %s; a budget check built on it "+
					"would admit a request it cannot pay for",
					money.FormatUSD(est), money.FormatUSD(b.Total))
			}
			if tc.wantEqual && est != b.Total {
				t.Fatalf("estimate %s should equal the worst-case cost %s exactly",
					money.FormatUSD(est), money.FormatUSD(b.Total))
			}
		})
	}

	// A completion that blows past the cap is the documented limit of the
	// guarantee, and the estimate is genuinely lower there. Asserting it keeps
	// the docstring honest.
	over, err := tbl.Cost("gpt-4o", &apiv1.Usage{PromptTokens: promptTokens, CompletionTokens: maxOut * 10})
	if err != nil {
		t.Fatalf("Cost: %v", err)
	}
	if est >= over.Total {
		t.Fatal("test case is not discriminating: an over-cap completion should exceed the estimate")
	}
}

func TestEstimateErrors(t *testing.T) {
	tbl := testTable(t)
	tests := []struct {
		name            string
		model           string
		prompt, maxOut  int64
		wantErrSentinel error
	}{
		{name: "negative prompt", model: "gpt-4o", prompt: -1, maxOut: 10, wantErrSentinel: ErrInvalidUsage},
		{name: "negative cap", model: "gpt-4o", prompt: 10, maxOut: -1, wantErrSentinel: ErrInvalidUsage},
		{name: "unpriced model", model: "nope", prompt: 10, maxOut: 10, wantErrSentinel: ErrUnpriced},
		{name: "overflowing prompt", model: "gpt-4o", prompt: 1 << 55, maxOut: 0, wantErrSentinel: money.ErrOverflow},
		{name: "overflowing cap", model: "gpt-4o", prompt: 0, maxOut: 1 << 55, wantErrSentinel: money.ErrOverflow},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tbl.Estimate(tc.model, tc.prompt, tc.maxOut)
			if err == nil {
				t.Fatalf("expected an error, got %s", money.FormatUSD(got))
			}
			if !errors.Is(err, tc.wantErrSentinel) {
				t.Fatalf("error %v does not match %v", err, tc.wantErrSentinel)
			}
			if got != 0 {
				t.Fatalf("expected a zero estimate alongside the error, got %d", got)
			}
		})
	}
	// A zero-token estimate is free, and must not be mistaken for an error.
	if got, err := tbl.Estimate("gpt-4o", 0, 0); err != nil || got != 0 {
		t.Fatalf("Estimate(0,0) = %d, %v; want 0, nil", got, err)
	}
}

func TestLoadValidSheet(t *testing.T) {
	const sheet = `{
	  "currency": "USD",
	  "source": "unit test",
	  "models": [
	    {
	      "model": "m-exact",
	      "aliases": ["m-alias"],
	      "input_per_1m_tokens": "0.15",
	      "cached_input_per_1m_tokens": "0.075",
	      "output_per_1m_tokens": "0.60",
	      "note": "cheap"
	    },
	    {
	      "model": "m-no-cached-rate",
	      "input_per_1m_tokens": "3.00",
	      "output_per_1m_tokens": "6.00"
	    },
	    {
	      "model": "m-free-cache",
	      "input_per_1m_tokens": "3.00",
	      "cached_input_per_1m_tokens": "0",
	      "output_per_1m_tokens": "6.00"
	    }
	  ]
	}`
	tbl, err := Load(strings.NewReader(sheet))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tbl.Len() != 3 {
		t.Fatalf("Len = %d, want 3", tbl.Len())
	}
	want := []string{"m-exact", "m-free-cache", "m-no-cached-rate"}
	got := tbl.Models()
	if len(got) != len(want) {
		t.Fatalf("Models = %q, want %q (sorted)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Models = %q, want %q (sorted)", got, want)
		}
	}
	if m, err := tbl.Lookup("m-alias"); err != nil || m.PricedAs != "m-exact" {
		t.Fatalf("alias lookup = %+v, %v", m, err)
	}

	// The default for an omitted cached rate must be the FULL input rate. Zero
	// would make every cache hit free and silently under-bill it.
	m, err := tbl.Lookup("m-no-cached-rate")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if m.Rates.CachedInput != m.Rates.Input {
		t.Fatalf("omitted cached rate = %d, want the input rate %d (never zero)",
			m.Rates.CachedInput, m.Rates.Input)
	}
	if m.Rates.CachedInput == 0 {
		t.Fatal("omitted cached rate defaulted to free")
	}
	// An explicit "0" is honoured, so a genuinely free cache read is expressible.
	free, err := tbl.Lookup("m-free-cache")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if free.Rates.CachedInput != 0 {
		t.Fatalf("explicit zero cached rate = %d, want 0", free.Rates.CachedInput)
	}
}

func TestLoadRejectsBadSheets(t *testing.T) {
	tests := []struct {
		name       string
		sheet      string
		wantErrHas []string
	}{
		{
			name: "duplicate model entry",
			sheet: `{"models":[
			  {"model":"dup","input_per_1m_tokens":"1","output_per_1m_tokens":"2"},
			  {"model":"dup","input_per_1m_tokens":"9","output_per_1m_tokens":"9"}]}`,
			wantErrHas: []string{"dup", "more than once"},
		},
		{
			name:       "negative input price",
			sheet:      `{"models":[{"model":"neg","input_per_1m_tokens":"-1","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"neg", "input_per_1m_tokens", "negative"},
		},
		{
			name: "negative cached price",
			sheet: `{"models":[{"model":"negc","input_per_1m_tokens":"1",
			  "cached_input_per_1m_tokens":"-0.5","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"negc", "cached_input_per_1m_tokens", "negative"},
		},
		{
			name:       "negative output price",
			sheet:      `{"models":[{"model":"nego","input_per_1m_tokens":"1","output_per_1m_tokens":"-2"}]}`,
			wantErrHas: []string{"nego", "output_per_1m_tokens", "negative"},
		},
		{
			name:       "inexact price is rejected rather than rounded",
			sheet:      `{"models":[{"model":"fine","input_per_1m_tokens":"0.0000001","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"fine", "input price", "6 decimal places"},
		},
		{
			name:       "price as a JSON number is rejected",
			sheet:      `{"models":[{"model":"num","input_per_1m_tokens":0.15,"output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"decoding price sheet"},
		},
		{
			name:       "missing input price",
			sheet:      `{"models":[{"model":"noin","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"noin", "'input_per_1m_tokens' is required"},
		},
		{
			name:       "missing output price",
			sheet:      `{"models":[{"model":"noout","input_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"noout", "'output_per_1m_tokens' is required"},
		},
		{
			name:       "empty model name",
			sheet:      `{"models":[{"model":"  ","input_per_1m_tokens":"1","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"models[0]", "empty"},
		},
		{
			name:       "no models",
			sheet:      `{"currency":"USD","models":[]}`,
			wantErrHas: []string{"no models"},
		},
		{
			name:       "wrong currency",
			sheet:      `{"currency":"EUR","models":[{"model":"m","input_per_1m_tokens":"1","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"EUR", "USD only"},
		},
		{
			name: "alias collides with another model's canonical name",
			sheet: `{"models":[
			  {"model":"a","input_per_1m_tokens":"1","output_per_1m_tokens":"2"},
			  {"model":"b","aliases":["a"],"input_per_1m_tokens":"9","output_per_1m_tokens":"9"}]}`,
			wantErrHas: []string{"alias \"a\"", "already used by"},
		},
		{
			name: "canonical name collides with an earlier alias",
			sheet: `{"models":[
			  {"model":"a","aliases":["b"],"input_per_1m_tokens":"1","output_per_1m_tokens":"2"},
			  {"model":"b","input_per_1m_tokens":"9","output_per_1m_tokens":"9"}]}`,
			wantErrHas: []string{"collides with an alias"},
		},
		{
			name: "duplicate alias across two models",
			sheet: `{"models":[
			  {"model":"a","aliases":["shared"],"input_per_1m_tokens":"1","output_per_1m_tokens":"2"},
			  {"model":"b","aliases":["shared"],"input_per_1m_tokens":"9","output_per_1m_tokens":"9"}]}`,
			wantErrHas: []string{"shared", "already used by"},
		},
		{
			name:       "empty alias",
			sheet:      `{"models":[{"model":"a","aliases":[""],"input_per_1m_tokens":"1","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"empty alias"},
		},
		{
			name:       "misspelled price key is not silently ignored",
			sheet:      `{"models":[{"model":"a","input_per_1m_tokens":"1","cached_input_per_1m":"0","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"cached_input_per_1m"},
		},
		{
			name:       "unparseable price string",
			sheet:      `{"models":[{"model":"a","input_per_1m_tokens":"one dollar","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"input_per_1m_tokens", "one dollar"},
		},
		{
			name:       "too many decimal places for a picodollar",
			sheet:      `{"models":[{"model":"a","input_per_1m_tokens":"1.0000000000001","output_per_1m_tokens":"2"}]}`,
			wantErrHas: []string{"decimal places"},
		},
		{
			name:       "trailing content after the sheet",
			sheet:      `{"models":[{"model":"a","input_per_1m_tokens":"1","output_per_1m_tokens":"2"}]} {"models":[]}`,
			wantErrHas: []string{"trailing content"},
		},
		{
			name:       "not an object",
			sheet:      `[]`,
			wantErrHas: []string{"decoding price sheet"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tbl, err := Load(strings.NewReader(tc.sheet))
			if err == nil {
				t.Fatalf("sheet loaded successfully with %d models; expected rejection", tbl.Len())
			}
			if tbl != nil {
				t.Fatal("a rejected sheet must not yield a table")
			}
			for _, want := range tc.wantErrHas {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("error %q does not mention %q", err, want)
				}
			}
		})
	}
}

func TestLoadFile(t *testing.T) {
	dir := t.TempDir()
	good := dir + "/prices.json"
	body, err := json.Marshal(DefaultSheet())
	if err != nil {
		t.Fatal(err)
	}
	if err := writeFile(good, body); err != nil {
		t.Fatal(err)
	}
	tbl, err := LoadFile(good)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if tbl.Len() != DefaultTable().Len() {
		t.Fatalf("round-tripped sheet has %d models, want %d", tbl.Len(), DefaultTable().Len())
	}

	bad := dir + "/broken.json"
	if err := writeFile(bad, []byte(`{"models":[]}`)); err != nil {
		t.Fatal(err)
	}
	// The path must appear in the error: "price sheet contains no models" with
	// three sheets configured is not an actionable startup log line.
	if _, err := LoadFile(bad); err == nil || !strings.Contains(err.Error(), bad) {
		t.Fatalf("error %v should name the offending path %s", err, bad)
	}

	if _, err := LoadFile(dir + "/does-not-exist.json"); err == nil {
		t.Fatal("expected an error for a missing file")
	}
}

func writeFile(path string, body []byte) error {
	return os.WriteFile(path, body, 0o600)
}

// TestDefaultTable is what makes the DefaultTable panic unreachable in
// production: if an edit to the demonstration sheet introduces an inexact price
// or a duplicate, this fails at test time rather than at process start.
func TestDefaultTable(t *testing.T) {
	if _, err := defaultSheet.Table(); err != nil {
		t.Fatalf("built-in demonstration sheet is invalid: %v", err)
	}
	tbl := DefaultTable()
	if tbl.Len() < 4 {
		t.Fatalf("demonstration sheet has only %d models", tbl.Len())
	}
	if DefaultTable() != tbl {
		t.Fatal("DefaultTable should return the same shared table")
	}
	for _, name := range tbl.Models() {
		m, err := tbl.Lookup(name)
		if err != nil {
			t.Fatalf("Lookup(%q): %v", name, err)
		}
		if !m.Rule.Exact() {
			t.Fatalf("canonical name %q did not match exactly", name)
		}
		if m.Rates.Output == 0 && m.Rates.Input == 0 {
			t.Fatalf("model %q is priced entirely free, which is almost certainly a typo", name)
		}
		if m.Rates.CachedInput > m.Rates.Input {
			t.Fatalf("model %q charges more for a cache read (%d) than for a fresh token (%d)",
				name, m.Rates.CachedInput, m.Rates.Input)
		}
		// Every demonstration model must actually price a request.
		b, err := tbl.Cost(name, &apiv1.Usage{PromptTokens: 1000, CompletionTokens: 100})
		if err != nil {
			t.Fatalf("Cost(%q): %v", name, err)
		}
		if b.Total <= 0 {
			t.Fatalf("model %q priced 1100 tokens at %s", name, money.FormatUSD(b.Total))
		}
	}

	// The sheet must survive a JSON round trip, since that is how an operator
	// will fork it.
	body, err := json.Marshal(DefaultSheet())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Load(strings.NewReader(string(body))); err != nil {
		t.Fatalf("marshalled default sheet does not load: %v", err)
	}
}

func TestDefaultSheetCopyIsDeep(t *testing.T) {
	a := DefaultSheet()
	// Find a row with aliases; mutating it must not be visible in a later copy.
	idx := -1
	for i := range a.Models {
		if len(a.Models[i].Aliases) > 0 {
			idx = i
			break
		}
	}
	if idx < 0 {
		t.Skip("demonstration sheet has no aliased model")
	}
	original := a.Models[idx].Aliases[0]
	a.Models[idx].Aliases[0] = "mutated"
	a.Models[idx].Model = "mutated-model"

	b := DefaultSheet()
	if b.Models[idx].Model == "mutated-model" {
		t.Fatal("mutating a returned sheet's model name leaked into the package sheet")
	}
	if b.Models[idx].Aliases[0] != original {
		t.Fatalf("mutating a returned sheet's aliases leaked into the package sheet: %q",
			b.Models[idx].Aliases[0])
	}
}

// TestTableIsSafeForConcurrentUse runs under -race. The Table is immutable after
// construction and the request path takes no lock, so this is the test that
// would catch a future "optimisation" that memoised a lookup miss into the map.
func TestTableIsSafeForConcurrentUse(t *testing.T) {
	tbl := testTable(t)
	names := []string{
		"gpt-4o", "openai/gpt-4o", "gpt-4o-2024-08-06", "legacy-0613",
		"unknown-model", "openai/unknown-2024-01-01", "thinker-1",
	}
	u := &apiv1.Usage{
		PromptTokens:            1000,
		CompletionTokens:        200,
		PromptTokensDetails:     &apiv1.PromptTokensDetails{CachedTokens: 400},
		CompletionTokensDetails: &apiv1.CompletionTokensDetails{ReasoningTokens: 150},
	}
	// Compute the expected total once, single-threaded, so the goroutines can
	// assert a value and not merely fail to crash.
	wantTotal := make(map[string]money.Pico, len(names))
	for _, n := range names {
		if b, err := tbl.Cost(n, u); err == nil {
			wantTotal[n] = b.Total
		}
	}

	const workers, iters = 16, 400
	var wg sync.WaitGroup
	errs := make(chan string, workers)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				n := names[(w+i)%len(names)]
				b, err := tbl.Cost(n, u)
				if err != nil {
					if _, priced := wantTotal[n]; priced {
						errs <- "unexpected error for " + n + ": " + err.Error()
						return
					}
					continue
				}
				if b.Total != wantTotal[n] {
					errs <- "inconsistent total for " + n
					return
				}
				if _, err := tbl.Estimate(n, 1000, 500); err != nil {
					if _, priced := wantTotal[n]; priced {
						errs <- "unexpected estimate error for " + n
						return
					}
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatal(e)
	}
	if tbl.Len() != 4 {
		t.Fatalf("table grew to %d entries under concurrent lookups", tbl.Len())
	}
}

// TestRequestPathDoesNotAllocate pins the claim in Lookup's comment. It is a
// test and not just a benchmark because an allocation regression here is a
// silent throughput loss that nobody runs a benchmark to notice.
func TestRequestPathDoesNotAllocate(t *testing.T) {
	tbl := testTable(t)
	u := &apiv1.Usage{
		PromptTokens:            4096,
		CompletionTokens:        512,
		PromptTokensDetails:     &apiv1.PromptTokensDetails{CachedTokens: 2048},
		CompletionTokensDetails: &apiv1.CompletionTokensDetails{ReasoningTokens: 256},
	}
	cases := []struct {
		name string
		fn   func()
	}{
		{"exact lookup", func() { _, _ = tbl.Lookup("gpt-4o") }},
		{"normalised lookup", func() { _, _ = tbl.Lookup("openai/gpt-4o-2024-08-06") }},
		{"cost", func() { _, _ = tbl.Cost("gpt-4o", u) }},
		{"estimate", func() { _, _ = tbl.Estimate("gpt-4o", 4096, 512) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := testing.AllocsPerRun(200, tc.fn); got != 0 {
				t.Fatalf("%s allocates %.1f times per call; the request path must not allocate", tc.name, got)
			}
		})
	}
}

func BenchmarkTableCost(b *testing.B) {
	tbl := DefaultTable()
	u := &apiv1.Usage{
		PromptTokens:            4096,
		CompletionTokens:        512,
		TotalTokens:             4608,
		PromptTokensDetails:     &apiv1.PromptTokensDetails{CachedTokens: 2048},
		CompletionTokensDetails: &apiv1.CompletionTokensDetails{ReasoningTokens: 256},
	}
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := tbl.Cost("gpt-4o", u); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkTableLookupMiss bounds the cost of the miss path. Two allocations are
// expected and unavoidable — the *UnpricedError and its Tried slice — so this
// benchmark exists to catch the case where candidate generation starts
// allocating too, which is what would make a client hammering a bad model name a
// meaningful garbage source.
func BenchmarkTableLookupMiss(b *testing.B) {
	tbl := DefaultTable()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := tbl.Lookup("openai/nope-2024-08-06"); err == nil {
			b.Fatal("expected a miss")
		}
	}
}

// BenchmarkTableLookupNormalised is the allocation assertion that matters: a hit
// via a fallback rule is on the request path and must be allocation-free.
func BenchmarkTableLookupNormalised(b *testing.B) {
	tbl := DefaultTable()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := tbl.Lookup("openai/gpt-4o-2024-08-06"); err != nil {
			b.Fatal(err)
		}
	}
}
