package tokens

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

func TestRatioTokens(t *testing.T) {
	tests := []struct {
		name  string
		ratio Ratio
		chars int
		want  int
	}{
		{"empty", 400, 0, 0},
		{"negative chars", 400, -5, 0},
		{"one char rounds up to one token", 400, 1, 1},
		{"exactly one token", 400, 4, 1},
		{"one over rounds up", 400, 5, 2},
		{"boundary below", 400, 8, 2},
		{"boundary above", 400, 9, 3},
		{"fractional ratio", 365, 365, 100},
		{"fractional ratio rounds up", 365, 366, 101},
		{"zero ratio falls back to default", 0, 380, 100},
		{"negative ratio falls back to default", -1, 380, 100},
		// A large-but-plausible prompt: 4MB of text. The point is that it does
		// not overflow or go negative, which is what an int32 intermediate or a
		// (chars*100) in a 32-bit world would do.
		{"multi-megabyte prompt", 400, 4 << 20, 1048576},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.ratio.Tokens(tc.chars); got != tc.want {
				t.Fatalf("Ratio(%d).Tokens(%d) = %d, want %d", tc.ratio, tc.chars, got, tc.want)
			}
		})
	}
}

// TestRatioTokensNeverZeroForNonEmpty is the boundary that matters for
// admission: any non-empty text must cost at least one token, or a caller could
// admit an unbounded number of one-character messages for free.
func TestRatioTokensNeverZeroForNonEmpty(t *testing.T) {
	for _, r := range []Ratio{1, 100, 355, 400, 410, 10000} {
		for chars := 1; chars < 32; chars++ {
			if got := r.Tokens(chars); got < 1 {
				t.Fatalf("Ratio(%d).Tokens(%d) = %d, want >= 1", r, chars, got)
			}
		}
	}
}

func TestRatioTextTokens(t *testing.T) {
	const r Ratio = 400
	tests := []struct {
		name string
		in   string
		want int
	}{
		{"empty", "", 0},
		// 11 letters+spaces at 4.00 -> 3.
		{"prose", "hello world", 3},
		// 8 digits at the dense ratio (2.40) -> 4.
		{"digits are dense", "12345678", 4},
		// Populations round separately: 1 letter -> 1, and 6 dense characters
		// ({, ", ", :, 1, }) at 2.40 -> 3.
		{"json fragment", `{"a":1}`, 4},
		// Ideographic: 3 runes at 1.25 tokens each -> ceil(3.75) = 4.
		{"cjk", "日本語", 4},
		// Accented Latin is charged by byte length, so 5 runes / 7 bytes -> 2.
		{"accented latin", "naïve", 2},
		{"single char", "x", 1},
		{"single ideograph", "日", 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := r.TextTokens(tc.in); got != tc.want {
				t.Fatalf("TextTokens(%q) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}

// TestTextTokensDoesNotUnderCountIdeographic is the regression guard for the
// mistake this function exists to prevent. A byte-length estimator divides
// Japanese by 4 and under-counts by ~3x; a rune-count estimator that ignored the
// script would still under-count by ~25%. Either would silently under-reserve
// every CJK prompt, so the test asserts the estimate is at least one token per
// ideographic rune, which is the floor every published tokenizer respects.
func TestTextTokensDoesNotUnderCountIdeographic(t *testing.T) {
	for _, s := range []string{"日本語", cjkText, strings.Repeat("漢字", 500), "안녕하세요", "こんにちは世界"} {
		runes := len([]rune(s))
		for _, r := range []Ratio{355, 380, 400, 410} {
			got := r.TextTokens(s)
			if got < runes {
				t.Errorf("Ratio(%d).TextTokens(%q...) = %d tokens for %d ideographic runes; "+
					"under-counting CJK is the bug this function exists to prevent",
					r, firstRunes(s, 8), got, runes)
			}
		}
	}
}

func firstRunes(s string, n int) string {
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return string(rs[:n])
}

func TestRatioFor(t *testing.T) {
	tests := []struct {
		model      string
		wantRatio  Ratio
		wantFamily string
	}{
		{"gpt-4o", 410, "o200k"},
		{"gpt-4o-mini", 410, "o200k"},
		{"gpt-4o-2024-08-06", 410, "o200k"},
		{"gpt-5", 410, "o200k"},
		// Longest-prefix: "gpt-4.1" must not be swallowed by "gpt-4".
		{"gpt-4.1-mini", 410, "o200k"},
		{"gpt-4-turbo", 400, "cl100k"},
		{"gpt-3.5-turbo", 400, "cl100k"},
		{"claude-sonnet-4-5", 365, "anthropic"},
		{"gemini-2.5-flash", 400, "gemini"},
		{"llama-3.3-70b", 385, "llama"},
		{"mistral-large", 355, "mistral"},
		{"mixtral-8x7b", 355, "mistral"},
		// Routing prefixes are stripped, matching pricing.Lookup.
		{"openai/gpt-4o", 410, "o200k"},
		{"openrouter/anthropic/claude-haiku-4-5", 365, "anthropic"},
		// Case folding.
		{"GPT-4o", 410, "o200k"},
		{"Claude-Sonnet-4-5", 365, "anthropic"},
		// Unknown models get the pessimistic default and say so.
		{"", DefaultRatio, DefaultFamily},
		{"some-new-model-2027", DefaultRatio, DefaultFamily},
		{"gpt", DefaultRatio, DefaultFamily},
		// A trailing slash is not a prefix; the name is used whole and misses.
		{"openai/", DefaultRatio, DefaultFamily},
	}
	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			r, fam := RatioFor(tc.model)
			if r != tc.wantRatio || fam != tc.wantFamily {
				t.Fatalf("RatioFor(%q) = (%d, %q), want (%d, %q)",
					tc.model, r, fam, tc.wantRatio, tc.wantFamily)
			}
		})
	}
}

// TestRatioForFamilyLabelIsHonest catches the failure mode where the family
// label silently claims a match that did not happen. A metric built on this
// label is the only signal an operator gets that a newly deployed model is being
// estimated at the default ratio, so a label that said "o200k" for an unmatched
// name would make the signal useless.
func TestRatioForFamilyLabelIsHonest(t *testing.T) {
	for _, model := range []string{"unknown-model", "", "x", "grok-4", "command-r-plus"} {
		r, fam := RatioFor(model)
		if fam != DefaultFamily {
			t.Errorf("RatioFor(%q) claimed family %q for a model with no table entry", model, fam)
		}
		if r != DefaultRatio {
			t.Errorf("RatioFor(%q) = %d, want DefaultRatio %d", model, r, DefaultRatio)
		}
	}
	// And the converse: every table entry must report its own family, so the
	// label can never be the default for a name that did match.
	for _, f := range familyRatios {
		r, fam := RatioFor(f.prefix)
		if fam != f.family || r != f.ratio {
			t.Errorf("RatioFor(%q) = (%d, %q), want (%d, %q)", f.prefix, r, fam, f.ratio, f.family)
		}
	}
}

func TestEstimatePrompt(t *testing.T) {
	tests := []struct {
		name string
		req  *apiv1.ChatRequest
		want int
	}{
		{
			name: "nil request",
			req:  nil,
			want: 0,
		},
		{
			// Priming(3) + framing(3) + role(1) + text: "hi" is 2 prose chars
			// at 4.10 -> 1. Total 8.
			name: "single tiny message",
			req: &apiv1.ChatRequest{
				Model:    "gpt-4o",
				Messages: []apiv1.Message{msg(apiv1.RoleUser, "hi")},
			},
			want: 8,
		},
		{
			// A message with no content still costs its framing: an empty
			// message is not free, and an estimator that returned 3 here would
			// be undercounting the wrapper the provider bills for.
			name: "empty content still costs framing",
			req: &apiv1.ChatRequest{
				Model:    "gpt-4o",
				Messages: []apiv1.Message{{Role: apiv1.RoleUser}},
			},
			want: 7,
		},
		{
			// Same text, two messages instead of one: the difference must be
			// exactly PerMessageTokens + one role token.
			name: "framing scales with message count",
			req: &apiv1.ChatRequest{
				Model: "gpt-4o",
				Messages: []apiv1.Message{
					msg(apiv1.RoleUser, "hi"),
					msg(apiv1.RoleUser, "hi"),
				},
			},
			want: 8 + PerMessageTokens + 1 + 1,
		},
		{
			// Name costs PerNameTokens plus the name's own text.
			name: "named message",
			req: &apiv1.ChatRequest{
				Model:    "gpt-4o",
				Messages: []apiv1.Message{{Role: apiv1.RoleUser, Name: "alice", Content: apiv1.NewTextContent("hi")}},
			},
			want: 8 + PerNameTokens + 2,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := EstimatePrompt(tc.req, "gpt-4o")
			if got.TokenCount != tc.want {
				t.Fatalf("TokenCount = %d, want %d", got.TokenCount, tc.want)
			}
			if got.Unknown {
				t.Fatalf("Unknown = true, want false: %s", got.UnknownReason)
			}
		})
	}
}

// TestEstimatePromptCountsTools is the test that would catch tool schemas being
// silently dropped — an omission that is invisible on chat traffic and enormous
// on agent traffic, where the schemas outweigh the messages.
func TestEstimatePromptCountsTools(t *testing.T) {
	base := &apiv1.ChatRequest{
		Model:    "gpt-4o",
		Messages: []apiv1.Message{msg(apiv1.RoleUser, "Any refunds?")},
	}
	withTools := &apiv1.ChatRequest{
		Model:    base.Model,
		Messages: base.Messages,
		Tools:    json.RawMessage(toolSchema),
	}

	bare := EstimatePrompt(base, "gpt-4o").TokenCount
	tooled := EstimatePrompt(withTools, "gpt-4o").TokenCount

	if tooled <= bare+ToolsFramingTokens {
		t.Fatalf("tools added only %d tokens over the bare request (framing alone is %d); "+
			"the %d-byte schema is not being counted", tooled-bare, ToolsFramingTokens, len(toolSchema))
	}
	// And the schema must be counted at the dense ratio, not the prose one: at
	// prose density this schema would come out ~30%% cheaper than it is.
	ratio, _ := RatioFor("gpt-4o")
	if prose := ratio.Tokens(len(toolSchema)); tooled-bare-ToolsFramingTokens <= prose {
		t.Fatalf("schema counted as %d tokens; prose density would give %d, so the "+
			"dense-character adjustment is not being applied",
			tooled-bare-ToolsFramingTokens, prose)
	}
}

func TestEstimatePromptCountsToolCallsAndIDs(t *testing.T) {
	toolCalls := json.RawMessage(`[{"id":"call_abc123","type":"function","function":{"name":"search_orders","arguments":"{\"customer_id\":\"cust_812\"}"}}]`)
	req := &apiv1.ChatRequest{
		Model: "gpt-4o",
		Messages: []apiv1.Message{
			msg(apiv1.RoleUser, "Any refunds?"),
			{Role: apiv1.RoleAssistant, ToolCalls: toolCalls},
			{Role: apiv1.RoleTool, ToolCallID: "call_abc123", Content: apiv1.NewTextContent(`{"orders":[]}`)},
		},
	}
	stripped := &apiv1.ChatRequest{
		Model: "gpt-4o",
		Messages: []apiv1.Message{
			msg(apiv1.RoleUser, "Any refunds?"),
			{Role: apiv1.RoleAssistant},
			{Role: apiv1.RoleTool, Content: apiv1.NewTextContent(`{"orders":[]}`)},
		},
	}
	full := EstimatePrompt(req, "gpt-4o").TokenCount
	without := EstimatePrompt(stripped, "gpt-4o").TokenCount
	if full <= without {
		t.Fatalf("tool_calls and tool_call_id contributed %d tokens; a replayed tool call "+
			"is prompt input on the next turn and must be counted", full-without)
	}
}

// TestEstimatePromptUnknownOnNonTextParts is the crux of the Unknown contract.
// The estimator must NOT silently count an image as zero tokens.
func TestEstimatePromptUnknownOnNonTextParts(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantUnknown bool
		reasonHas   string
	}{
		{
			name:        "string content is knowable",
			body:        `{"model":"gpt-4o","messages":[{"role":"user","content":"describe this"}]}`,
			wantUnknown: false,
		},
		{
			name:        "text-only array content is knowable",
			body:        `{"model":"gpt-4o","messages":[{"role":"user","content":[{"type":"text","text":"describe this"}]}]}`,
			wantUnknown: false,
		},
		{
			name: "image part makes the estimate untrustworthy",
			body: `{"model":"gpt-4o","messages":[{"role":"user","content":[` +
				`{"type":"text","text":"describe this"},` +
				`{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}]}`,
			wantUnknown: true,
			reasonHas:   "messages[0]",
		},
		{
			name: "audio part too",
			body: `{"model":"gpt-4o","messages":[{"role":"user","content":[` +
				`{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}]}]}`,
			wantUnknown: true,
			reasonHas:   "non-text parts",
		},
		{
			name: "reason names the first offending message and the count",
			body: `{"model":"gpt-4o","messages":[` +
				`{"role":"user","content":"hello"},` +
				`{"role":"user","content":[{"type":"image_url","image_url":{"url":"a"}}]},` +
				`{"role":"user","content":[{"type":"image_url","image_url":{"url":"b"}}]}]}`,
			wantUnknown: true,
			reasonHas:   "2 message(s)",
		},
		{
			name:        "null content is knowable, not unknown",
			body:        `{"model":"gpt-4o","messages":[{"role":"assistant","content":null}]}`,
			wantUnknown: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := mustUnmarshalReq(t, tc.body)
			got := EstimatePrompt(req, "gpt-4o")
			if got.Unknown != tc.wantUnknown {
				t.Fatalf("Unknown = %v, want %v (reason %q)", got.Unknown, tc.wantUnknown, got.UnknownReason)
			}
			if got.Unknown && got.UnknownReason == "" {
				t.Fatal("Unknown is set but UnknownReason is empty; the caller cannot log a policy decision")
			}
			if !got.Unknown && got.UnknownReason != "" {
				t.Fatalf("UnknownReason %q set while Unknown is false", got.UnknownReason)
			}
			if tc.reasonHas != "" && !strings.Contains(got.UnknownReason, tc.reasonHas) {
				t.Fatalf("UnknownReason = %q, want it to mention %q", got.UnknownReason, tc.reasonHas)
			}
			// Even when Unknown, the text that WAS present must still be
			// counted: the number is a floor, not a zero.
			if got.TokenCount <= 0 {
				t.Fatalf("TokenCount = %d; an unknown estimate must still count the text it could read", got.TokenCount)
			}
			// The String form must carry the caveat, so a log line cannot
			// launder the number.
			if got.Unknown && !strings.Contains(got.String(), "INCOMPLETE") {
				t.Fatalf("String() = %q, want it to flag the estimate as incomplete", got.String())
			}
		})
	}
}

// TestEstimatePromptUsesRoutedModelNotRequestedModel guards the aliasing case:
// the client asks for one name, the router picks another, and the estimate must
// follow the model that will actually do the tokenizing.
func TestEstimatePromptUsesRoutedModelNotRequestedModel(t *testing.T) {
	req := &apiv1.ChatRequest{
		Model:    "fast-chat", // an alias with no table entry
		Messages: []apiv1.Message{msg(apiv1.RoleUser, strings.Repeat("word ", 200))},
	}
	viaAlias := EstimatePrompt(req, req.Model).TokenCount
	viaRouted := EstimatePrompt(req, "claude-haiku-4-5").TokenCount
	if viaAlias == viaRouted {
		t.Fatal("the estimate ignored the routed model; an alias and a real model with " +
			"different ratios must not produce the same count")
	}
	// Anthropic's lower ratio must give MORE tokens for the same text.
	if viaRouted <= viaAlias {
		t.Fatalf("claude ratio (3.65) gave %d tokens vs default (3.80) %d; expected more", viaRouted, viaAlias)
	}
}

func TestEstimateCompletion(t *testing.T) {
	ptr := func(n int) *int { return &n }
	tests := []struct {
		name        string
		req         *apiv1.ChatRequest
		want        int
		wantUnknown bool
	}{
		{
			name: "max_tokens is used verbatim",
			req:  &apiv1.ChatRequest{MaxTokens: ptr(4096)},
			want: 4096,
		},
		{
			name: "max_completion_tokens wins over max_tokens",
			req:  &apiv1.ChatRequest{MaxTokens: ptr(100), MaxCompletionTok: ptr(2048)},
			want: 2048,
		},
		{
			// A hostile cap must not become a hostile reservation.
			name: "absurd cap is clamped",
			req:  &apiv1.ChatRequest{MaxTokens: ptr(1 << 30)},
			want: MaxCompletionReserve,
		},
		{
			name:        "no cap and no prior falls back to the documented default",
			req:         &apiv1.ChatRequest{},
			want:        DefaultCompletionReserve,
			wantUnknown: true,
		},
		{
			name:        "nil request",
			req:         nil,
			want:        DefaultCompletionReserve,
			wantUnknown: true,
		},
		{
			// max_tokens: 0 is "unset" per apiv1.EffectiveMaxTokens, not "zero
			// tokens". Treating it as a zero reserve would admit every request
			// for free.
			name:        "explicit zero cap is treated as unset",
			req:         &apiv1.ChatRequest{MaxTokens: ptr(0)},
			want:        DefaultCompletionReserve,
			wantUnknown: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := EstimateCompletion(tc.req, "gpt-4o", nil)
			if got.TokenCount != tc.want {
				t.Fatalf("TokenCount = %d, want %d", got.TokenCount, tc.want)
			}
			if got.Unknown != tc.wantUnknown {
				t.Fatalf("Unknown = %v, want %v", got.Unknown, tc.wantUnknown)
			}
		})
	}
}

// TestEstimateCompletionPrefersCapOverPrior asserts the direction of the
// conservatism. Even a well-warmed prior that says "these completions are 50
// tokens" must lose to an explicit max_tokens of 4096, because the cap is the
// only number the provider guarantees.
func TestEstimateCompletionPrefersCapOverPrior(t *testing.T) {
	p := warmPrior(t, "gpt-4o", 50, 30)
	if n, ok := p.Reserve(); !ok {
		t.Fatalf("prior should be trusted after warming, got reserve %d ok=%v", n, ok)
	}
	cap := 4096
	req := &apiv1.ChatRequest{MaxTokens: &cap}
	got := EstimateCompletion(req, "gpt-4o", p)
	if got.TokenCount != cap {
		t.Fatalf("TokenCount = %d, want the client cap %d; the prior must not be allowed to "+
			"under-reserve below a bound the provider will actually enforce", got.TokenCount, cap)
	}
	if got.Unknown {
		t.Fatal("a cap-derived estimate is a real bound, not an unknown")
	}
}

func TestEstimateCompletionUsesPriorWhenUncapped(t *testing.T) {
	p := warmPrior(t, "gpt-4o", 400, 30)
	got := EstimateCompletion(&apiv1.ChatRequest{}, "gpt-4o", p)
	if got.Unknown {
		t.Fatalf("a trusted prior should not be flagged unknown: %s", got.UnknownReason)
	}
	// 400 tokens * 1.5 safety = 600.
	if got.TokenCount != 600 {
		t.Fatalf("TokenCount = %d, want 600 (400-token mean scaled by SafetyPercent 150)", got.TokenCount)
	}
}

// TestEstimateCompletionIgnoresWrongModelPrior is the check that would silently
// always pass if the model comparison were dropped, so it is asserted directly:
// a prior for a chatty model must not be used to reserve for a different one.
func TestEstimateCompletionIgnoresWrongModelPrior(t *testing.T) {
	p := warmPrior(t, "gpt-4o-mini", 40, 30)
	got := EstimateCompletion(&apiv1.ChatRequest{}, "gpt-5", p)
	if got.TokenCount != DefaultCompletionReserve {
		t.Fatalf("TokenCount = %d, want the default %d; a prior for %q must not be used for %q",
			got.TokenCount, DefaultCompletionReserve, p.Model(), "gpt-5")
	}
	if !got.Unknown {
		t.Fatal("Unknown = false; falling back to a policy default is exactly the case the caller must be told about")
	}
}

func TestEstimateCombineAndTotal(t *testing.T) {
	tests := []struct {
		name        string
		a, b        Estimate
		wantCount   int
		wantUnknown bool
		reasonHas   []string
	}{
		{
			name:      "both known",
			a:         Estimate{TokenCount: 10},
			b:         Estimate{TokenCount: 5},
			wantCount: 15,
		},
		{
			name:        "left unknown propagates",
			a:           Estimate{TokenCount: 10, Unknown: true, UnknownReason: "image"},
			b:           Estimate{TokenCount: 5},
			wantCount:   15,
			wantUnknown: true,
			reasonHas:   []string{"image"},
		},
		{
			name:        "right unknown propagates",
			a:           Estimate{TokenCount: 10},
			b:           Estimate{TokenCount: 5, Unknown: true, UnknownReason: "no prior"},
			wantCount:   15,
			wantUnknown: true,
			reasonHas:   []string{"no prior"},
		},
		{
			name:        "both unknown keeps both reasons",
			a:           Estimate{TokenCount: 10, Unknown: true, UnknownReason: "image"},
			b:           Estimate{TokenCount: 5, Unknown: true, UnknownReason: "no prior"},
			wantCount:   15,
			wantUnknown: true,
			reasonHas:   []string{"image", "no prior"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.a.Combine(tc.b)
			if got.TokenCount != tc.wantCount {
				t.Fatalf("TokenCount = %d, want %d", got.TokenCount, tc.wantCount)
			}
			if got.Unknown != tc.wantUnknown {
				t.Fatalf("Unknown = %v, want %v", got.Unknown, tc.wantUnknown)
			}
			for _, want := range tc.reasonHas {
				if !strings.Contains(got.UnknownReason, want) {
					t.Errorf("UnknownReason = %q, want it to mention %q", got.UnknownReason, want)
				}
			}
		})
	}
}

func TestEstimateAll(t *testing.T) {
	req := mustUnmarshalReq(t, `{"model":"gpt-4o","max_tokens":256,"messages":[{"role":"user","content":[`+
		`{"type":"text","text":"what is in this picture"},`+
		`{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}]}`)
	all := EstimateAll(req, "gpt-4o", nil)
	if !all.Prompt.Unknown {
		t.Fatal("prompt side should be unknown (image part)")
	}
	if all.Completion.Unknown {
		t.Fatal("completion side has an explicit cap and should be known")
	}
	total := all.Total()
	if total.TokenCount != all.Prompt.TokenCount+all.Completion.TokenCount {
		t.Fatalf("Total = %d, want %d", total.TokenCount, all.Prompt.TokenCount+all.Completion.TokenCount)
	}
	if !total.Unknown {
		t.Fatal("Total must stay unknown when either side is; a budget decision made on this " +
			"number needs to know the image was not counted")
	}
}

// --- error bound -----------------------------------------------------------

// Measured error of the estimator against the reference tokenizer over the
// fixture set, as of the ratio table in tokens.go:
//
//	aggregate error over all fixtures : +4.2%  (the estimator over-counts)
//	mean absolute error per fixture   : 10.1%
//	worst single fixture              : +23.4% (cjk, over-counting)
//	worst UNDER-count                 : -9.9%  (code-review, 10 tokens)
//
// The bounds below are set just outside those measurements. They are asymmetric
// on purpose: over-counting is the safe direction for admission (it can only
// reject a request that would have fit), while under-counting admits a request
// the budget cannot pay for, so the under-count bound is the tighter of the two.
//
// This is a calibration test, not a correctness test: the reference tokenizer is
// not ground truth about any real provider (see reference.go), it is a stable
// yardstick. What the test actually protects is that a future edit to
// familyRatios, to DenseRatioPercent, or to the framing constants cannot move
// the estimator without someone noticing and updating the numbers above.
const (
	maxAggregateErrorPct    = 8.0
	maxMeanAbsoluteErrorPct = 13.0
	maxOverCountPct         = 28.0
	maxUnderCountPct        = 13.0
)

func TestEstimatorErrorBound(t *testing.T) {
	fixtures := promptFixtures()
	if len(fixtures) < 6 {
		t.Fatalf("only %d fixtures; the bound is not meaningful over a small set", len(fixtures))
	}

	var sumRef, sumEst int
	var sumAbsPct float64
	for _, f := range fixtures {
		ref := Reference.CountRequest(f.req)
		if ref == 0 {
			t.Fatalf("fixture %q counts as zero reference tokens", f.name)
		}
		est := EstimatePrompt(f.req, f.req.Model)
		if est.Unknown {
			t.Fatalf("fixture %q is flagged unknown; the error bound must be measured on "+
				"prompts the estimator claims to understand", f.name)
		}
		pct := 100 * float64(est.TokenCount-ref) / float64(ref)
		t.Logf("%-24s reference=%4d estimate=%4d error=%+7.2f%%", f.name, ref, est.TokenCount, pct)

		if pct > maxOverCountPct {
			t.Errorf("fixture %q over-counts by %+.2f%% (reference %d, estimate %d), bound is %+.1f%%",
				f.name, pct, ref, est.TokenCount, maxOverCountPct)
		}
		if -pct > maxUnderCountPct {
			t.Errorf("fixture %q UNDER-counts by %.2f%% (reference %d, estimate %d), bound is %.1f%%. "+
				"Under-counting is the dangerous direction: it admits requests the budget cannot pay for.",
				f.name, -pct, ref, est.TokenCount, maxUnderCountPct)
		}
		sumRef += ref
		sumEst += est.TokenCount
		sumAbsPct += absF(pct)
	}

	aggregate := 100 * float64(sumEst-sumRef) / float64(sumRef)
	meanAbs := sumAbsPct / float64(len(fixtures))
	t.Logf("MEASURED: aggregate error %+.2f%% over %d reference tokens; mean absolute error %.2f%% over %d fixtures",
		aggregate, sumRef, meanAbs, len(fixtures))

	if absF(aggregate) > maxAggregateErrorPct {
		t.Errorf("aggregate error %+.2f%% exceeds the stated bound of +/-%.1f%%; "+
			"the ratio table has drifted", aggregate, maxAggregateErrorPct)
	}
	if meanAbs > maxMeanAbsoluteErrorPct {
		t.Errorf("mean absolute error %.2f%% exceeds the stated bound of %.1f%%",
			meanAbs, maxMeanAbsoluteErrorPct)
	}
	// Direction check: the estimator is supposed to lean high in aggregate.
	if aggregate < 0 {
		t.Errorf("aggregate error is %+.2f%%: the estimator now UNDER-counts on average, "+
			"which inverts the safety property the package doc claims", aggregate)
	}
}

// TestEstimatorBeatsNaiveCharDivision demonstrates that the framing constants
// and dense-character handling are earning their complexity. A naive
// len(text)/4 estimator is the obvious alternative; if it were no worse, this
// package should be deleted.
func TestEstimatorBeatsNaiveCharDivision(t *testing.T) {
	var ours, naive float64
	fixtures := promptFixtures()
	for _, f := range fixtures {
		ref := Reference.CountRequest(f.req)
		chars := 0
		for i := range f.req.Messages {
			chars += len(f.req.Messages[i].Content.Text())
		}
		chars += len(f.req.Tools) + len(f.req.ToolChoice)
		naiveEst := chars / 4

		ours += absF(100 * float64(EstimatePrompt(f.req, f.req.Model).TokenCount-ref) / float64(ref))
		naive += absF(100 * float64(naiveEst-ref) / float64(ref))
	}
	n := float64(len(fixtures))
	t.Logf("MEASURED: mean absolute error — this package %.2f%%, naive len/4 %.2f%%", ours/n, naive/n)
	if ours/n >= naive/n {
		t.Errorf("this estimator's mean absolute error (%.2f%%) is no better than naive "+
			"len(text)/4 (%.2f%%); the framing constants and density handling are not paying for themselves",
			ours/n, naive/n)
	}
}

func absF(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

// --- Prior -----------------------------------------------------------------

// fakeClock is a manually advanced clock, so staleness is tested by moving time
// rather than by sleeping.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{t: time.Date(2026, 7, 30, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

// warmPrior returns a prior with n observations of exactly tok tokens, which is
// enough to be trusted and settles the EWMA on tok exactly.
func warmPrior(t *testing.T, model string, tok, n int) *Prior {
	t.Helper()
	p, err := NewPrior("acme", model, DefaultPriorConfig())
	if err != nil {
		t.Fatalf("NewPrior: %v", err)
	}
	for i := 0; i < n; i++ {
		p.Observe(tok)
	}
	return p
}

func TestPriorConfigValidate(t *testing.T) {
	ok := DefaultPriorConfig()
	tests := []struct {
		name   string
		mutate func(*PriorConfig)
		errHas string
	}{
		{"default is valid", func(*PriorConfig) {}, ""},
		{"zero smoothing", func(c *PriorConfig) { c.Smoothing = 0 }, "Smoothing"},
		{"zero min samples", func(c *PriorConfig) { c.MinSamples = 0 }, "MinSamples"},
		{"negative max age", func(c *PriorConfig) { c.MaxAge = -time.Second }, "MaxAge"},
		{"zero default", func(c *PriorConfig) { c.Default = 0 }, "Default"},
		{"default above hard cap", func(c *PriorConfig) { c.Default = MaxCompletionReserve + 1 }, "MaxCompletionReserve"},
		{"negative floor", func(c *PriorConfig) { c.Floor = -1 }, "Floor"},
		{"zero ceiling", func(c *PriorConfig) { c.Ceiling = 0 }, "Ceiling"},
		{"floor above ceiling", func(c *PriorConfig) { c.Floor, c.Ceiling = 100, 50 }, "Floor"},
		{"ceiling above hard cap", func(c *PriorConfig) { c.Ceiling = MaxCompletionReserve + 1 }, "MaxCompletionReserve"},
		{"safety below 100", func(c *PriorConfig) { c.SafetyPercent = 99 }, "SafetyPercent"},
		{"zero max keys", func(c *PriorConfig) { c.MaxKeys = 0 }, "MaxKeys"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := ok
			tc.mutate(&cfg)
			err := cfg.Validate()
			switch {
			case tc.errHas == "" && err != nil:
				t.Fatalf("Validate() = %v, want nil", err)
			case tc.errHas != "" && err == nil:
				t.Fatalf("Validate() = nil, want an error mentioning %q", tc.errHas)
			case tc.errHas != "" && !strings.Contains(err.Error(), tc.errHas):
				t.Fatalf("Validate() = %v, want it to mention %q", err, tc.errHas)
			}
			// NewPrior and NewPriors must share the validation, so a bad config
			// cannot sneak in through either door.
			if _, err := NewPrior("t", "m", cfg); (err != nil) != (tc.errHas != "") {
				t.Errorf("NewPrior error = %v, inconsistent with Validate", err)
			}
			if _, err := NewPriors(cfg); (err != nil) != (tc.errHas != "") {
				t.Errorf("NewPriors error = %v, inconsistent with Validate", err)
			}
		})
	}
}

// TestPriorColdIsNotTrusted is the most important prior test. An online
// estimator that trusts its first few samples is a budget bypass: three short
// answers teach it to reserve 40 tokens, and then a 4000-token answer is admitted
// against a reservation that cannot cover it.
func TestPriorColdIsNotTrusted(t *testing.T) {
	cfg := DefaultPriorConfig()
	p, err := NewPrior("acme", "gpt-4o", cfg)
	if err != nil {
		t.Fatal(err)
	}

	if n, ok := p.Reserve(); ok || n != cfg.Default {
		t.Fatalf("empty prior: Reserve() = (%d, %v), want (%d, false)", n, ok, cfg.Default)
	}

	for i := int64(1); i < cfg.MinSamples; i++ {
		p.Observe(40)
		n, ok := p.Reserve()
		if ok {
			t.Fatalf("prior trusted after %d samples (MinSamples is %d); it would reserve %d",
				i, cfg.MinSamples, n)
		}
		if n != cfg.Default {
			t.Fatalf("untrusted prior returned %d, want the configured default %d", n, cfg.Default)
		}
	}
	// The MinSamples-th observation flips it.
	p.Observe(40)
	n, ok := p.Reserve()
	if !ok {
		t.Fatalf("prior still untrusted at exactly MinSamples=%d samples", cfg.MinSamples)
	}
	// 40 * 1.5 = 60, below the floor of 64, so the floor applies.
	if n != cfg.Floor {
		t.Fatalf("Reserve() = %d, want the floor %d (40-token mean scaled to 60 is below it)", n, cfg.Floor)
	}
}

func TestPriorSampleCountIsVisible(t *testing.T) {
	p, err := NewPrior("acme", "gpt-4o", DefaultPriorConfig())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		p.Observe(100)
	}
	three := p.Snapshot()
	for i := 0; i < 2997; i++ {
		p.Observe(100)
	}
	threeK := p.Snapshot()

	if three.Samples != 3 || threeK.Samples != 3000 {
		t.Fatalf("Samples = %d and %d, want 3 and 3000; a caller must be able to tell a "+
			"3-observation prior from a 3000-observation one", three.Samples, threeK.Samples)
	}
	if three.Trusted {
		t.Error("3-sample prior reports Trusted")
	}
	if !threeK.Trusted {
		t.Error("3000-sample prior reports untrusted")
	}
}

func TestPriorEWMATracksAndHighWaterRemembers(t *testing.T) {
	cfg := DefaultPriorConfig()
	cfg.Smoothing = 4
	cfg.Floor = 1
	p, err := NewPrior("acme", "gpt-4o", cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Settle on 100.
	for i := int64(0); i < cfg.MinSamples+40; i++ {
		p.Observe(100)
	}
	if got := p.Snapshot().Mean; got < 99 || got > 101 {
		t.Fatalf("Mean = %d after 100-token observations, want ~100", got)
	}

	// One 5000-token outlier. The mean must move only a little, but the
	// high-water mark must remember all of it, and Reserve must follow the
	// high-water mark — that is the whole reason it exists.
	p.Observe(5000)
	st := p.Snapshot()
	if st.HighWater != 5000 {
		t.Fatalf("HighWater = %d, want 5000", st.HighWater)
	}
	if st.Mean > 2000 {
		t.Fatalf("Mean = %d; a single outlier should not dominate a smoothed mean", st.Mean)
	}
	if st.Reserve != 5000 {
		t.Fatalf("Reserve = %d, want 5000: the high-water mark must dominate when it exceeds "+
			"the scaled mean, or the tenant that emits one long answer per thousand requests "+
			"gets a reservation that cannot cover it", st.Reserve)
	}

	// Then the workload shifts upward and stays there. The mean must track it,
	// which the high-water mark alone could never do downward or upward beyond
	// its own maximum.
	for i := 0; i < 200; i++ {
		p.Observe(9000)
	}
	if got := p.Snapshot().Mean; got < 8900 {
		t.Fatalf("Mean = %d after a sustained shift to 9000; the EWMA is not tracking", got)
	}
}

func TestPriorReserveBounds(t *testing.T) {
	tests := []struct {
		name    string
		observe int
		floor   int
		ceiling int
		want    int
	}{
		{"floor applies to tiny completions", 1, 64, 8192, 64},
		{"ceiling applies to huge completions", 100000, 64, 8192, 8192},
		{"in between is the scaled mean", 1000, 64, 8192, 1500},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultPriorConfig()
			cfg.Floor, cfg.Ceiling = tc.floor, tc.ceiling
			p, err := NewPrior("acme", "gpt-4o", cfg)
			if err != nil {
				t.Fatal(err)
			}
			for i := int64(0); i < cfg.MinSamples+20; i++ {
				p.Observe(tc.observe)
			}
			n, ok := p.Reserve()
			if !ok {
				t.Fatal("prior not trusted after warming")
			}
			if n != tc.want {
				t.Fatalf("Reserve() = %d, want %d", n, tc.want)
			}
		})
	}
}

// TestPriorIgnoresNonPositiveObservations covers the failure-path case. A
// cancelled or failed request produced no evidence about completion length, and
// folding a zero in would drag the reserve down exactly when a provider is
// unhealthy and requests are dying early.
func TestPriorIgnoresNonPositiveObservations(t *testing.T) {
	p := warmPrior(t, "gpt-4o", 1000, 40)
	before := p.Snapshot()
	for i := 0; i < 500; i++ {
		p.Observe(0)
		p.Observe(-1)
		p.ObserveUsage(nil)
		p.ObserveUsage(&apiv1.Usage{CompletionTokens: 0})
	}
	after := p.Snapshot()
	if after.Samples != before.Samples {
		t.Fatalf("Samples went %d -> %d; non-positive observations must not count",
			before.Samples, after.Samples)
	}
	if after.Mean != before.Mean || after.Reserve != before.Reserve {
		t.Fatalf("mean/reserve moved from (%d, %d) to (%d, %d) on zero observations",
			before.Mean, before.Reserve, after.Mean, after.Reserve)
	}
}

// TestPriorClampsHostileObservation guards against a buggy or malicious upstream
// poisoning the prior. A reported completion of 10^9 tokens, taken at face
// value, would make every subsequent request from that tenant unaffordable.
func TestPriorClampsHostileObservation(t *testing.T) {
	cfg := DefaultPriorConfig()
	cfg.Ceiling = MaxCompletionReserve
	p, err := NewPrior("acme", "gpt-4o", cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := int64(0); i < cfg.MinSamples; i++ {
		p.Observe(1 << 30)
	}
	st := p.Snapshot()
	if st.HighWater > MaxCompletionReserve {
		t.Fatalf("HighWater = %d, want it clamped to %d", st.HighWater, MaxCompletionReserve)
	}
	if st.Reserve > MaxCompletionReserve {
		t.Fatalf("Reserve = %d, want <= %d", st.Reserve, MaxCompletionReserve)
	}
	if st.Reserve <= 0 {
		t.Fatalf("Reserve = %d; a clamp that overflowed to zero or negative would be a budget bypass", st.Reserve)
	}
}

func TestPriorObserveUsageIncludesReasoningTokens(t *testing.T) {
	p, err := NewPrior("acme", "o3", DefaultPriorConfig())
	if err != nil {
		t.Fatal(err)
	}
	// A thinking model: 40 visible tokens, 3960 reasoning tokens, all of which
	// the provider already folded into CompletionTokens.
	u := &apiv1.Usage{
		CompletionTokens:        4000,
		CompletionTokensDetails: &apiv1.CompletionTokensDetails{ReasoningTokens: 3960},
	}
	for i := int64(0); i < DefaultPriorConfig().MinSamples; i++ {
		p.ObserveUsage(u)
	}
	st := p.Snapshot()
	if st.HighWater != 4000 {
		t.Fatalf("HighWater = %d, want 4000; a reserve that excluded reasoning tokens would "+
			"under-reserve a thinking model by 100x", st.HighWater)
	}
}

func TestPriorStaleness(t *testing.T) {
	clk := newFakeClock()
	cfg := DefaultPriorConfig()
	cfg.Now = clk.Now
	cfg.MaxAge = time.Hour
	cfg.Floor = 1
	p, err := NewPrior("acme", "gpt-4o", cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := int64(0); i < cfg.MinSamples; i++ {
		p.Observe(1000)
	}
	if n, ok := p.Reserve(); !ok || n != 1500 {
		t.Fatalf("fresh prior: Reserve() = (%d, %v), want (1500, true)", n, ok)
	}

	// Exactly at MaxAge is still fresh; strictly past it is stale.
	clk.Advance(cfg.MaxAge)
	if n, ok := p.Reserve(); !ok || n != 1500 {
		t.Fatalf("prior at exactly MaxAge: Reserve() = (%d, %v), want (1500, true)", n, ok)
	}
	clk.Advance(time.Nanosecond)
	if n, ok := p.Reserve(); ok || n != cfg.Default {
		t.Fatalf("stale prior: Reserve() = (%d, %v), want (%d, false)", n, ok, cfg.Default)
	}

	// A fresh observation revives it, keeping the accumulated evidence.
	p.Observe(1000)
	if n, ok := p.Reserve(); !ok || n != 1500 {
		t.Fatalf("revived prior: Reserve() = (%d, %v), want (1500, true)", n, ok)
	}

	// MaxAge = 0 disables expiry entirely.
	cfg2 := cfg
	cfg2.MaxAge = 0
	p2, err := NewPrior("acme", "gpt-4o", cfg2)
	if err != nil {
		t.Fatal(err)
	}
	for i := int64(0); i < cfg2.MinSamples; i++ {
		p2.Observe(1000)
	}
	clk.Advance(365 * 24 * time.Hour)
	if _, ok := p2.Reserve(); !ok {
		t.Fatal("MaxAge=0 should disable expiry, but the prior went stale")
	}
}

// TestPriorStaleFallsBackThroughEstimateCompletion closes the loop: a stale
// prior must not merely report untrusted, it must actually cause
// EstimateCompletion to reserve the conservative default. A prior whose
// staleness was computed but ignored is exactly the check that silently always
// passes.
func TestPriorStaleFallsBackThroughEstimateCompletion(t *testing.T) {
	clk := newFakeClock()
	cfg := DefaultPriorConfig()
	cfg.Now = clk.Now
	cfg.MaxAge = time.Hour
	p, err := NewPrior("acme", "gpt-4o", cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := int64(0); i < cfg.MinSamples; i++ {
		p.Observe(120)
	}
	fresh := EstimateCompletion(&apiv1.ChatRequest{}, "gpt-4o", p)
	if fresh.Unknown {
		t.Fatalf("fresh prior produced an unknown estimate: %s", fresh.UnknownReason)
	}

	clk.Advance(2 * time.Hour)
	stale := EstimateCompletion(&apiv1.ChatRequest{}, "gpt-4o", p)
	if stale.TokenCount != DefaultCompletionReserve || !stale.Unknown {
		t.Fatalf("stale prior: got (%d, unknown=%v), want (%d, true)",
			stale.TokenCount, stale.Unknown, DefaultCompletionReserve)
	}
}

// TestNilPriorIsUsable covers the contract that makes Priors.Lookup safe to call
// without a nil check at every site. Lookup returns nil by design, so a nil
// prior must behave as "no evidence" rather than panicking somewhere downstream.
func TestNilPriorIsUsable(t *testing.T) {
	var p *Prior
	if got := p.Tenant(); got != "" {
		t.Errorf("nil.Tenant() = %q, want empty", got)
	}
	if got := p.Model(); got != "" {
		t.Errorf("nil.Model() = %q, want empty", got)
	}
	p.Observe(500)      // must not panic
	p.ObserveUsage(nil) // must not panic
	p.ObserveUsage(&apiv1.Usage{CompletionTokens: 500})
	n, ok := p.Reserve()
	if ok {
		t.Error("nil prior reports Trusted")
	}
	if n != DefaultCompletionReserve {
		t.Errorf("nil.Reserve() = %d, want %d", n, DefaultCompletionReserve)
	}
	est := EstimateCompletion(&apiv1.ChatRequest{}, "gpt-4o", p)
	if est.TokenCount != DefaultCompletionReserve || !est.Unknown {
		t.Errorf("EstimateCompletion with a nil prior = (%d, unknown=%v), want (%d, true)",
			est.TokenCount, est.Unknown, DefaultCompletionReserve)
	}
}

// TestPriorEWMAConvergesExactly pins the claim in Observe that thousandths of a
// token is fine enough resolution for the recurrence not to stall. With
// whole-token resolution the integer division would truncate to zero while still
// up to Smoothing-1 tokens short of the target, and the mean would sit
// permanently wrong — a bug that is invisible unless asserted, because the mean
// would still look plausible.
func TestPriorEWMAConvergesExactly(t *testing.T) {
	for _, smoothing := range []int{1, 2, 8, 32} {
		cfg := DefaultPriorConfig()
		cfg.Smoothing = smoothing
		p, err := NewPrior("acme", "gpt-4o", cfg)
		if err != nil {
			t.Fatal(err)
		}
		// Seed low, then feed a constant target forever.
		p.Observe(1)
		for i := 0; i < 5000; i++ {
			p.Observe(1000)
		}
		if got := p.Snapshot().Mean; got != 1000 {
			t.Errorf("Smoothing=%d: mean converged to %d, not the constant target 1000; "+
				"the integer recurrence stalled", smoothing, got)
		}

		// And downward, which truncation-toward-zero could break asymmetrically.
		for i := 0; i < 5000; i++ {
			p.Observe(100)
		}
		if got := p.Snapshot().Mean; got != 100 {
			t.Errorf("Smoothing=%d: mean converged to %d on the way down, not 100", smoothing, got)
		}
	}
}

func TestPriorsLookupNeverCreates(t *testing.T) {
	ps, err := NewPriors(DefaultPriorConfig())
	if err != nil {
		t.Fatal(err)
	}
	// The attack this guards: `model` comes from the request body, so a
	// get-or-create on the admission path is a remote memory-growth primitive.
	for i := 0; i < 10000; i++ {
		if got := ps.Lookup("acme", fmt.Sprintf("attacker-model-%d", i)); got != nil {
			t.Fatalf("Lookup created a prior for %d", i)
		}
	}
	if n := ps.Len(); n != 0 {
		t.Fatalf("Len = %d after 10000 lookups, want 0; Lookup must never allocate", n)
	}
}

func TestPriorsBoundedByMaxKeys(t *testing.T) {
	cfg := DefaultPriorConfig()
	cfg.MaxKeys = 4
	ps, err := NewPriors(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		if err := ps.Observe("acme", fmt.Sprintf("m%d", i), 100); err != nil {
			t.Fatalf("Observe %d: %v", i, err)
		}
	}
	if n := ps.Len(); n != 4 {
		t.Fatalf("Len = %d, want 4", n)
	}

	// New keys are refused, and the refusal is countable rather than silent.
	for i := 4; i < 20; i++ {
		err := ps.Observe("acme", fmt.Sprintf("m%d", i), 100)
		if err == nil {
			t.Fatalf("Observe of new key m%d succeeded past MaxKeys", i)
		}
		if !strings.Contains(err.Error(), "full") {
			t.Fatalf("error = %v, want it to mention the table being full", err)
		}
	}
	if n := ps.Len(); n != 4 {
		t.Fatalf("Len = %d after 16 refused observations, want 4", n)
	}
	if d := ps.Dropped(); d != 16 {
		t.Fatalf("Dropped = %d, want 16; a saturated prior table must be visible as a metric", d)
	}
	// Existing keys keep working while the table is full — saturation must not
	// stop the priors that already exist from tracking.
	if err := ps.Observe("acme", "m0", 200); err != nil {
		t.Fatalf("Observe on an existing key while full: %v", err)
	}
	if got := ps.Lookup("acme", "m0").Snapshot().Samples; got != 2 {
		t.Fatalf("existing prior has %d samples, want 2", got)
	}
}

func TestPriorsKeyedByTenantAndModel(t *testing.T) {
	ps, err := NewPriors(DefaultPriorConfig())
	if err != nil {
		t.Fatal(err)
	}
	for i := int64(0); i < DefaultPriorConfig().MinSamples; i++ {
		_ = ps.Observe("acme", "gpt-4o", 100)
		_ = ps.Observe("acme", "gpt-5", 4000)
		_ = ps.Observe("globex", "gpt-4o", 2000)
	}
	cases := []struct {
		tenant, model string
		wantHighWater int
	}{
		{"acme", "gpt-4o", 100},
		{"acme", "gpt-5", 4000},
		{"globex", "gpt-4o", 2000},
	}
	for _, tc := range cases {
		p := ps.Lookup(tc.tenant, tc.model)
		if p == nil {
			t.Fatalf("Lookup(%q, %q) = nil", tc.tenant, tc.model)
		}
		if got := p.Snapshot().HighWater; got != tc.wantHighWater {
			t.Errorf("Lookup(%q, %q).HighWater = %d, want %d; priors must not bleed across "+
				"tenant or model", tc.tenant, tc.model, got, tc.wantHighWater)
		}
	}
	if ps.Lookup("nobody", "gpt-4o") != nil {
		t.Error("Lookup for an unknown tenant returned a prior")
	}
	if n := len(ps.Snapshot()); n != 3 {
		t.Errorf("Snapshot has %d entries, want 3", n)
	}
}

func TestPriorsObserveUsage(t *testing.T) {
	ps, err := NewPriors(DefaultPriorConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := ps.ObserveUsage("acme", "gpt-4o", nil); err != nil {
		t.Fatalf("nil usage should be a no-op, got %v", err)
	}
	if n := ps.Len(); n != 0 {
		t.Fatalf("nil usage created %d priors", n)
	}
	if err := ps.ObserveUsage("acme", "gpt-4o", &apiv1.Usage{CompletionTokens: 512}); err != nil {
		t.Fatal(err)
	}
	if got := ps.Lookup("acme", "gpt-4o").Snapshot().HighWater; got != 512 {
		t.Fatalf("HighWater = %d, want 512", got)
	}
}

// TestPriorConcurrentUse is the race test. Run under -race it covers the two
// places a race is possible: the per-prior counters, and the prior table's map.
func TestPriorConcurrentUse(t *testing.T) {
	cfg := DefaultPriorConfig()
	cfg.MaxKeys = 32
	ps, err := NewPriors(cfg)
	if err != nil {
		t.Fatal(err)
	}

	const goroutines = 24
	const iterations = 400
	var wg sync.WaitGroup
	start := make(chan struct{})

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			<-start
			for i := 0; i < iterations; i++ {
				model := fmt.Sprintf("m%d", i%8)
				switch g % 4 {
				case 0:
					_ = ps.Observe("acme", model, 100+i%500)
				case 1:
					if p := ps.Lookup("acme", model); p != nil {
						_, _ = p.Reserve()
					}
				case 2:
					if p := ps.Lookup("acme", model); p != nil {
						_ = p.Snapshot()
					}
				case 3:
					_ = EstimateCompletion(&apiv1.ChatRequest{}, model, ps.Lookup("acme", model))
					_ = ps.Snapshot()
					_ = ps.Len()
					_ = ps.Dropped()
				}
			}
		}(g)
	}
	close(start)
	wg.Wait()

	// Every observation landed somewhere: the total sample count across the
	// table must equal the number of observing goroutines' iterations. This is
	// what catches a lost update that -race alone would not see, because a
	// racy-but-lucky increment is not a data race, it is just wrong.
	var total int64
	for _, st := range ps.Snapshot() {
		total += st.Samples
	}
	observers := int64(0)
	for g := 0; g < goroutines; g++ {
		if g%4 == 0 {
			observers++
		}
	}
	want := observers * iterations
	if total != want {
		t.Fatalf("total samples across the table = %d, want %d; observations were lost", total, want)
	}
}

// TestEstimatePromptConcurrentUse asserts the request-path function is safe and
// pure. EstimatePrompt takes a *ChatRequest and must not mutate it, or two
// concurrent estimates of the same cached request would corrupt each other.
func TestEstimatePromptConcurrentUse(t *testing.T) {
	req := mustUnmarshalReq(t, `{"model":"gpt-4o","messages":[`+
		`{"role":"system","content":"be brief"},`+
		`{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"u"}}]}]}`)
	want := EstimatePrompt(req, "gpt-4o")

	var wg sync.WaitGroup
	for g := 0; g < 32; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				got := EstimatePrompt(req, "gpt-4o")
				if got.TokenCount != want.TokenCount || got.Unknown != want.Unknown {
					t.Errorf("concurrent estimate = %+v, want %+v", got, want)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// TestEstimatePromptAllocations pins the request-path allocation cost. String
// content — the overwhelmingly common shape — must not allocate at all, because
// this runs before every single request.
func TestEstimatePromptAllocations(t *testing.T) {
	req := &apiv1.ChatRequest{
		Model: "gpt-4o",
		Messages: []apiv1.Message{
			msg(apiv1.RoleSystem, loremProse),
			msg(apiv1.RoleUser, userProse),
		},
	}
	// Warm any lazy initialisation before measuring.
	EstimatePrompt(req, "gpt-4o")
	got := testing.AllocsPerRun(200, func() {
		if EstimatePrompt(req, "gpt-4o").TokenCount == 0 {
			t.Fatal("unexpected zero estimate")
		}
	})
	if got != 0 {
		t.Fatalf("EstimatePrompt allocated %.1f times per call on string-form content; "+
			"the pre-flight estimator runs on every request and must be allocation-free here", got)
	}
}

func BenchmarkEstimatePrompt(b *testing.B) {
	req := &apiv1.ChatRequest{
		Model: "gpt-4o",
		Messages: []apiv1.Message{
			msg(apiv1.RoleSystem, loremProse),
			msg(apiv1.RoleUser, userProse),
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = EstimatePrompt(req, "gpt-4o")
	}
}

func BenchmarkPriorObserveContended(b *testing.B) {
	ps, err := NewPriors(DefaultPriorConfig())
	if err != nil {
		b.Fatal(err)
	}
	_ = ps.Observe("acme", "gpt-4o", 100)
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = ps.Observe("acme", "gpt-4o", 250)
		}
	})
}
