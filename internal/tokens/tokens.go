// Package tokens produces the pre-flight token estimate that budget admission
// runs on, and — this is the whole design commitment — it never pretends that
// estimate is a count.
//
// The gateway cannot know the true token count before the request runs:
//
//   - It cannot ship a real BPE tokenizer per model. cl100k_base alone is ~1.6MB
//     of vocabulary, o200k_base is larger, each model family needs its own, and
//     the vocabularies for several commercially interesting models are simply not
//     published. A gateway that bundled four of them would still guess on the
//     fifth, while paying megabytes of binary and a cold-start cost for the
//     illusion of exactness.
//   - The completion length is not merely unpublished, it is *unknowable*: it
//     does not exist until generation finishes.
//
// So this package returns an Estimate that carries its own trustworthiness
// (Unknown, UnknownReason), and the two sides of the estimate are deliberately
// biased in opposite-but-safe ways:
//
//	prompt:     characters / chars-per-token + the structural framing every chat
//	            format imposes. Approximate in both directions; the ratio table
//	            below is pinned by a test that asserts a measured error bound so
//	            a regression in it fails the build.
//	completion: the client's max_tokens when it set one (a real upper bound the
//	            provider enforces), else an observed per-(tenant, model) prior.
//	            Deliberately CONSERVATIVE — it over-reserves.
//
// Over-reserving is the correct direction of error for a budget: the worst it
// can do is reject a request that would have fit, which the tenant sees and can
// act on. Under-reserving admits a request the budget cannot pay for, which
// nobody sees until the invoice. See EstimateCompletion.
//
// Reconciliation against the real provider counts happens after the fact in
// internal/ledger; the estimate is admission control, never billing.
package tokens

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
)

// Estimate is a token count plus an explicit statement of whether it can be
// trusted.
//
// Unknown is not "we failed"; it is "this number is a floor, not an estimate,
// because the input contained something whose token cost is not derivable from
// the text we can see". The canonical case is an image part: counting it as zero
// under-admits nothing and over-admits everything, so the caller must be able to
// see the difference and apply a policy (reject, charge a fixed image
// surcharge, or admit and reconcile after the fact). A bool the caller can
// ignore is still better than a silent lie, and UnknownReason means the policy
// decision can be logged with the reason attached.
type Estimate struct {
	// TokenCount is the estimated tokens. When Unknown is true it is a lower
	// bound on the truth (the text parts were counted; the rest was not).
	TokenCount int
	// Unknown reports that part of the input could not be estimated from text.
	Unknown bool
	// UnknownReason is a short human-readable reason, empty iff !Unknown.
	UnknownReason string
}

// Combine adds two estimates, propagating untrustworthiness. Used to fold the
// prompt and completion sides into the single number admission compares against
// a budget.
func (e Estimate) Combine(o Estimate) Estimate {
	out := Estimate{TokenCount: e.TokenCount + o.TokenCount}
	switch {
	case e.Unknown && o.Unknown:
		out.Unknown, out.UnknownReason = true, e.UnknownReason+"; "+o.UnknownReason
	case e.Unknown:
		out.Unknown, out.UnknownReason = true, e.UnknownReason
	case o.Unknown:
		out.Unknown, out.UnknownReason = true, o.UnknownReason
	}
	return out
}

// String renders the estimate for logs, keeping the caveat attached to the
// number so it cannot be copied into a log line without it.
func (e Estimate) String() string {
	if !e.Unknown {
		return fmt.Sprintf("%d tokens (estimated)", e.TokenCount)
	}
	return fmt.Sprintf("%d tokens (estimated, INCOMPLETE: %s)", e.TokenCount, e.UnknownReason)
}

// Structural overhead of the chat format.
//
// Source: OpenAI's published counting guidance — the num_tokens_from_messages
// recipe in the "How to count tokens with tiktoken" cookbook. Every message is
// wrapped in framing tokens (<|start|>role<|message|>...<|end|>), and the
// request ends with a priming sequence for the assistant turn.
//
// These are APPROXIMATIONS, and knowingly so:
//
//   - The cookbook's own numbers are version-dependent: 3 per message for
//     gpt-3.5-turbo-0613 and later, 4 per message (and -1 per name) for
//     gpt-3.5-turbo-0301.
//   - Anthropic and Google do not publish their framing at all. Their formats
//     are of the same shape and order of magnitude, so the same constants are
//     applied rather than inventing a second unverifiable table.
//
// They matter most for the case they are least accurate on: a long conversation
// of very short messages, where framing is a large fraction of the total. That
// is exactly the shape a chat UI produces, so dropping them (the tempting
// simplification) would systematically under-reserve the most common traffic.
const (
	// PerMessageTokens is the framing cost of one message.
	PerMessageTokens = 3
	// PerNameTokens is the extra cost of a message carrying a `name`.
	PerNameTokens = 1
	// ReplyPrimingTokens is the once-per-request cost of priming the assistant
	// turn.
	ReplyPrimingTokens = 3
	// ToolsFramingTokens is the fixed preamble cost of sending any tool
	// definitions at all, on top of the serialized schemas themselves. The
	// least certain constant here: OpenAI injects a namespace wrapper whose
	// exact text is not published, and community measurements land in the
	// 10-20 range. Rounded up, because over-reserving is the safe direction.
	ToolsFramingTokens = 16
)

// Ratio is a chars-per-token ratio in hundredths of a character (400 = 4.00
// chars per token).
//
// Hundredths of a char rather than a float64 for the same reason internal/money
// is integer: the estimate is compared against a budget, and two gateway
// replicas doing the same float division in a different order must not admit
// different requests. Integer arithmetic also keeps the hot path free of any
// rounding-mode argument — Tokens rounds one way, up, always.
type Ratio int

// DefaultRatio is used for models with no table entry.
//
// 3.80 sits at the pessimistic end of the 3.5-4.3 band that published English
// tokenizers occupy, which means an unknown model's prompt is over-counted
// rather than under-counted. For admission that is the direction we want; see
// the package doc.
const DefaultRatio Ratio = 380

// DefaultFamily is the family label reported for a model that matched no entry.
const DefaultFamily = "default"

// DenseRatioPercent scales a family ratio down for "dense" characters —
// punctuation, symbols, digits, and line structure — as a percentage (60 = the
// family ratio times 0.60).
//
// Every ratio in familyRatios is measured on English prose, and applying a prose
// ratio to text that is not prose is this estimator's largest systematic
// under-count. It is not a small one, and the direction is the dangerous one.
//
// Where 0.60 comes from: prose averages ~4.0 characters per token, while the
// characters this class covers average ~2.4. Digits are the clearest case,
// because both cl100k_base and o200k_base document three-digit grouping exactly
// (1 token per 3 characters). Punctuation clusters in JSON are similar — the
// vocabularies hold `":"` and `},` as units, so a `{"type":"string"}` run spends
// a token every two or three characters. Indentation is the same again: a
// newline plus a tab is one token for two characters, and formatted source code
// is a quarter indentation by volume.
//
// Agent traffic is mostly schema and code by volume, so pricing it as prose
// under-reserves exactly the workload that costs the most.
//
// The scale is applied to whichever family ratio is in force rather than being
// an absolute chars-per-token number, so the relationship between families
// survives an edit to the table above.
const DenseRatioPercent = 60

// Dense returns the ratio to use for punctuation, symbols and digits at this
// family's efficiency.
func (r Ratio) Dense() Ratio {
	if r <= 0 {
		r = DefaultRatio
	}
	d := Ratio(int(r) * DenseRatioPercent / 100)
	if d < 1 {
		d = 1
	}
	return d
}

// Tokens converts a character count to tokens, rounding up, and never returning
// zero for non-empty input.
//
// Rounding up per counted segment (rather than summing characters across the
// whole request and dividing once) is deliberate: real tokenizers cannot merge
// across message boundaries either, so per-segment rounding is both closer to
// the truth and biased upward.
func (r Ratio) Tokens(chars int) int {
	if chars <= 0 {
		return 0
	}
	if r <= 0 {
		r = DefaultRatio
	}
	// int64 throughout: a 4GB prompt is not a realistic request, but a
	// silently negative token count from an overflowed multiply would be a
	// budget bypass, and the widening is free on the platforms this runs on.
	n := (int64(chars)*100 + int64(r) - 1) / int64(r)
	if n < 1 {
		n = 1
	}
	return int(n)
}

// CharsPerToken renders the ratio as a float for human-facing output only.
// Nothing on the request path uses it.
func (r Ratio) CharsPerToken() float64 { return float64(r) / 100 }

// TextTokens estimates the tokens of a string at this ratio.
//
// It is deliberately not len(s)/ratio, because of two mistakes that would make.
//
// First, the byte length. Every ratio in the table is measured on English, where
// one character is one byte. Dividing Japanese — three bytes per character, and
// roughly one *token* per character in every published tokenizer — by 4.0
// under-counts a CJK prompt by about 3x. Under-counting is the one direction of
// error this package may not have, and "it's fine for our traffic" is not an
// argument a gateway gets to make about a whole writing system.
//
// Second, the uniform ratio. A prose ratio applied to JSON or code under-counts
// by a third; see DenseRatioPercent.
//
// So the string is classified into three populations in one pass:
//
//	letters and spaces      — divided by the family ratio, as measured.
//	punctuation and digits  — divided by the Dense() ratio.
//	ideographic runes       — IdeographicTokensPerRuneX100 hundredths of a token
//	                          each, independent of the ratio, because that cost
//	                          is a property of the script rather than of the
//	                          vocabulary's efficiency on English.
//
// Each population is rounded up separately, which biases the total upward by up
// to two tokens per string. That is intentional and it is smaller than the
// ratio's own error.
func (r Ratio) TextTokens(s string) int {
	if s == "" {
		return 0
	}
	prose, dense, ideo := 0, 0, 0
	if !hasNonASCII(s) {
		// Fast path: no decoding, and it covers nearly all real traffic.
		for i := 0; i < len(s); i++ {
			if isDenseASCII(s[i]) {
				dense++
			} else {
				prose++
			}
		}
	} else {
		for _, ru := range s {
			switch {
			case ru < utf8.RuneSelf:
				if isDenseASCII(byte(ru)) {
					dense++
				} else {
					prose++
				}
			case ru >= ideographicFloor:
				ideo++
			default:
				// Accented Latin, Greek, Cyrillic, Arabic, Devanagari. Real
				// tokenizers are meaningfully less efficient on these than on
				// ASCII, and the UTF-8 byte length is a serviceable proxy for
				// how much less — it errs high, which is the safe way to be
				// wrong.
				prose += utf8.RuneLen(ru)
			}
		}
	}
	n := r.Tokens(prose) + r.Dense().Tokens(dense) + (ideo*IdeographicTokensPerRuneX100+99)/100
	if n < 1 {
		n = 1
	}
	return n
}

// isDenseASCII reports whether an ASCII byte is one real tokenizers spend
// tokens on fastest: punctuation, symbols, digits, and line structure.
//
// The space is the deliberate exception. A space merges into the word that
// follows it (" the" is one token in every GPT-family vocabulary), so it is
// nearly free and belongs with the prose. A newline or a tab does not merge into
// anything — it is a token, or part of a short indentation token — which is why
// the estimate on a code block is otherwise a quarter low: indentation is a
// large fraction of the characters in formatted source and none of it is free.
func isDenseASCII(c byte) bool {
	switch c {
	case ' ':
		return false
	case '\n', '\r', '\t', '\f', '\v':
		return true
	}
	switch {
	case c < ' ':
		return true
	case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z':
		return false
	default:
		return true
	}
}

// IdeographicTokensPerRuneX100 is the tokens-per-rune cost of ideographic
// script, in hundredths of a token.
//
// 1.25 sits above the ~1.0 that o200k_base achieves on Japanese and below the
// ~1.5-2.0 that cl100k_base needs for the same text, so it over-reserves on the
// newest models and roughly matches the older ones. Erring high is the point;
// see the package doc.
const IdeographicTokensPerRuneX100 = 125

// hasNonASCII reports whether s contains any byte outside ASCII. A plain byte
// loop rather than strings.IndexFunc, which would decode every rune.
func hasNonASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return true
		}
	}
	return false
}

// familyRatio maps a model-name prefix to a tokenizer family and its ratio.
//
// Matching is longest-prefix, because "gpt-4o-mini" must not be captured by the
// "gpt-4" entry when a more specific one exists. Values are approximate
// averages over English prose and are the thing the error-bound test pins:
// change one and TestEstimatorErrorBound tells you by how much it moved.
var familyRatios = []struct {
	prefix string
	family string
	ratio  Ratio
}{
	// o200k_base (gpt-4o, gpt-5, o-series). A larger vocabulary than cl100k,
	// so slightly more characters per token on English.
	{"gpt-5", "o200k", 410},
	{"gpt-4o", "o200k", 410},
	{"gpt-4.1", "o200k", 410},
	{"o1", "o200k", 410},
	{"o3", "o200k", 410},
	{"o4", "o200k", 410},
	// cl100k_base (gpt-4, gpt-3.5-turbo, text-embedding-3-*).
	{"gpt-4", "cl100k", 400},
	{"gpt-3.5", "cl100k", 400},
	{"text-embedding-3", "cl100k", 400},
	// Anthropic does not publish its vocabulary. Its counting endpoint
	// consistently reports more tokens than cl100k for the same English text,
	// hence the lower ratio.
	{"claude", "anthropic", 365},
	// Google documents "roughly 4 characters per token" for Gemini.
	{"gemini", "gemini", 400},
	// SentencePiece-based open models (Llama 3's 128k vocab, Mistral's 32k).
	// Mistral's smaller vocabulary is meaningfully less efficient.
	{"llama", "llama", 385},
	{"mistral", "mistral", 355},
	{"mixtral", "mistral", 355},
	{"qwen", "qwen", 380},
	{"deepseek", "deepseek", 380},
}

// RatioFor resolves a model name to its chars-per-token ratio and the family
// label it matched, which is DefaultFamily when nothing did.
//
// The family label is returned rather than kept private so that the caller can
// put it on a metric: a sudden shift in the share of traffic estimated at the
// default ratio is the signal that a new model was deployed and nobody updated
// this table.
//
// A routing prefix is stripped ("openai/gpt-4o", "openrouter/openai/gpt-4o"),
// mirroring internal/pricing.Lookup, so the two do not disagree about which
// family a decorated name belongs to. Matching is case-insensitive on the ASCII
// range only, and allocation-free for the overwhelmingly common already-lower
// name.
func RatioFor(model string) (Ratio, string) {
	if model == "" {
		return DefaultRatio, DefaultFamily
	}
	name := model
	if i := strings.LastIndexByte(name, '/'); i >= 0 && i < len(name)-1 {
		name = name[i+1:]
	}
	best, bestLen := DefaultRatio, -1
	bestFamily := DefaultFamily
	for _, f := range familyRatios {
		if len(f.prefix) > bestLen && hasASCIIFoldPrefix(name, f.prefix) {
			best, bestFamily, bestLen = f.ratio, f.family, len(f.prefix)
		}
	}
	return best, bestFamily
}

// hasASCIIFoldPrefix reports whether s starts with prefix, folding ASCII case.
// prefix is assumed already lower-case (every entry in familyRatios is).
func hasASCIIFoldPrefix(s, prefix string) bool {
	if len(s) < len(prefix) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != prefix[i] {
			return false
		}
	}
	return true
}

// roleTokens is the cost of a role string.
//
// The four legal roles are a closed set that every published vocabulary holds as
// a single token — they are the most frequent strings in the training data of a
// chat model. Dividing "assistant" by 4.1 characters per token and rounding up
// says 3, which is wrong on every message in every request: at 3 tokens per
// message the error compounds with conversation length, which is precisely the
// axis a chat gateway scales along. An unknown role (a provider extension) falls
// back to the ratio, where being approximate is acceptable because it is rare.
func roleTokens(role string, ratio Ratio) int {
	switch role {
	case apiv1.RoleSystem, apiv1.RoleUser, apiv1.RoleAssistant, apiv1.RoleTool, "developer", "function":
		return 1
	case "":
		return 0
	default:
		return ratio.TextTokens(role)
	}
}

// EstimatePrompt estimates the prompt tokens of a request.
//
// It walks the messages, counts each one's text through Ratio.TextTokens (which
// applies the family's chars-per-token ratio, with the density and script
// adjustments documented there), and adds the structural framing documented
// above. Tool definitions, replayed tool calls, and response_format are counted
// too: they are prompt input the provider bills for, and on agent traffic they
// outweigh the messages.
//
// model is passed separately from req.Model because the gateway routes aliases:
// the client asks for "fast-chat" and the estimate must use the tokenizer of
// whatever upstream model the router actually chose. Estimating against the alias
// would quietly use the default ratio for every aliased route.
//
// Pure and allocation-free for string-form content, which is the common case, so
// it is safe to call concurrently on a shared request. Array-form content costs
// one JSON parse in HasNonTextParts — the price of not silently counting an
// image as zero.
func EstimatePrompt(req *apiv1.ChatRequest, model string) Estimate {
	if req == nil {
		return Estimate{}
	}
	ratio, _ := RatioFor(model)

	var est Estimate
	total := ReplyPrimingTokens
	nonTextIdx := -1
	nonTextCount := 0

	for i := range req.Messages {
		m := &req.Messages[i]
		total += PerMessageTokens
		total += roleTokens(m.Role, ratio)
		if m.Name != "" {
			total += PerNameTokens + ratio.TextTokens(m.Name)
		}
		total += ratio.TextTokens(m.Content.Text())
		if m.ToolCallID != "" {
			total += ratio.TextTokens(m.ToolCallID)
		}
		// An assistant turn's tool_calls are replayed to the model as prompt on
		// the following turn, so they are prompt tokens now. Estimated from the
		// raw JSON we received, which is not byte-identical to what the provider
		// re-serializes, but is the right order of magnitude.
		if len(m.ToolCalls) > 0 {
			total += ratio.TextTokens(string(m.ToolCalls))
		}
		if !m.Content.IsString() && m.Content.HasNonTextParts() {
			nonTextCount++
			if nonTextIdx < 0 {
				nonTextIdx = i
			}
		}
	}

	// Tool definitions are billed as prompt tokens by every provider that
	// supports them, and a realistic agent's tool schemas are frequently larger
	// than its messages. Omitting them is the single biggest under-count
	// available to a naive estimator.
	if len(req.Tools) > 0 {
		total += ToolsFramingTokens + ratio.TextTokens(string(req.Tools))
	}
	if len(req.ToolChoice) > 0 {
		total += ratio.TextTokens(string(req.ToolChoice))
	}
	if len(req.ResponseFormat) > 0 {
		total += ratio.TextTokens(string(req.ResponseFormat))
	}

	est.TokenCount = total
	if nonTextCount > 0 {
		est.Unknown = true
		est.UnknownReason = fmt.Sprintf(
			"%d message(s) carry non-text parts (first at messages[%d]); image/audio token cost is model-specific and not derivable from the request",
			nonTextCount, nonTextIdx)
	}
	return est
}

// DefaultCompletionReserve is the blind fallback used when the client set no
// cap and there is no usable prior.
//
// 1024 is not a measurement, it is a policy choice, and it is documented as one:
// it is large enough that a typical chat answer fits under it (so the reserve is
// genuinely conservative for the common case) and small enough that a single
// uncapped request from a new tenant cannot exhaust a day's budget on its own.
// Operators who care should set MaxTokens on their clients or let the prior
// warm up; both replace this number with something defensible.
const DefaultCompletionReserve = 1024

// MaxCompletionReserve bounds every completion estimate.
//
// The bound exists because the inputs are hostile in both directions: a client
// may send max_tokens=2^31-1, and a buggy provider may report a completion of
// 10^9 tokens which would then poison the prior. Reserving a number that large
// would reject every subsequent request from the tenant — a self-inflicted
// outage caused by a field the caller controls.
const MaxCompletionReserve = 262_144

// EstimateCompletion estimates the completion tokens a request will produce.
//
// The order of preference is the point of this function:
//
//  1. The client's max_tokens / max_completion_tokens, when set. This is not an
//     estimate at all, it is a hard ceiling the provider enforces, so using it
//     is the only choice that cannot be wrong in the dangerous direction. It is
//     usually a large over-estimate — a client that sets max_tokens=4096 and
//     receives 80 tokens has reserved 51x what it spent — and that is accepted
//     on purpose. Admission reserves; internal/ledger bills the real usage; the
//     difference is released. A budget that over-reserves rejects requests that
//     would have fit, which is visible and recoverable. A budget that
//     under-reserves admits requests it cannot pay for, which is neither.
//  2. The observed prior for this (tenant, model), once it has enough samples
//     to be worth trusting and is recent enough not to be stale.
//  3. DefaultCompletionReserve, flagged Unknown so the caller can see that the
//     number is a policy default and not evidence.
//
// prior may be nil. A prior belonging to a different model is ignored rather
// than used: reserving a summarizer's 60-token history against a code-generation
// model is worse than having no prior at all, and a caller that fetches the
// wrong prior should not be silently rewarded for it.
func EstimateCompletion(req *apiv1.ChatRequest, model string, prior *Prior) Estimate {
	if req != nil {
		if hardCap := req.EffectiveMaxTokens(); hardCap > 0 {
			if hardCap > MaxCompletionReserve {
				hardCap = MaxCompletionReserve
			}
			return Estimate{TokenCount: hardCap}
		}
	}
	if prior != nil {
		if pm := prior.Model(); pm == "" || pm == model {
			if n, ok := prior.Reserve(); ok {
				return Estimate{TokenCount: n}
			}
		}
	}
	return Estimate{
		TokenCount: DefaultCompletionReserve,
		Unknown:    true,
		UnknownReason: fmt.Sprintf(
			"no max_tokens and no trusted prior for model %q; reserved the default %d tokens",
			model, DefaultCompletionReserve),
	}
}

// EstimateRequest is both sides of the estimate plus their sum, which is what
// budget admission actually needs.
type EstimateRequest struct {
	Prompt     Estimate
	Completion Estimate
}

// Total folds the two sides together, propagating untrustworthiness from either.
func (r EstimateRequest) Total() Estimate { return r.Prompt.Combine(r.Completion) }

// EstimateAll estimates both sides of a request in one call.
func EstimateAll(req *apiv1.ChatRequest, model string, prior *Prior) EstimateRequest {
	return EstimateRequest{
		Prompt:     EstimatePrompt(req, model),
		Completion: EstimateCompletion(req, model, prior),
	}
}

// PriorConfig parameterises the observed-completion-length estimator.
type PriorConfig struct {
	// Smoothing is the EWMA divisor N: each observation moves the mean by
	// 1/N of its distance. N=8 gives a mean dominated by roughly the last 8-20
	// requests, which tracks a tenant changing its prompt template within a
	// deploy rather than over a week. Must be >= 1; 1 means "last value wins".
	Smoothing int

	// MinSamples is how many observations a prior needs before it is trusted at
	// all. Below it, Reserve falls back to Default.
	//
	// This is the guard against the failure mode that makes online priors
	// dangerous: the first three requests of the day are short, the prior says
	// "reserve 40 tokens", and the fourth request generates 4000 tokens that
	// the budget already spent. A cold prior must fail toward the conservative
	// default, never toward the tiny sample it happens to hold.
	MinSamples int64

	// MaxAge makes a prior expire. A prior last updated a week ago describes a
	// workload that may no longer exist; trusting it is trusting a memory.
	// Zero disables expiry.
	MaxAge time.Duration

	// Default is the reserve used when the prior is cold or stale.
	Default int

	// Floor is the smallest reserve a trusted prior may produce, so that a
	// tenant whose answers are all one word does not end up with an effectively
	// zero reservation and thereby unlimited admissions.
	Floor int

	// Ceiling caps a trusted prior's reserve.
	Ceiling int

	// SafetyPercent scales the EWMA mean before it is used, expressed as a
	// percentage (150 = mean * 1.5). Completion lengths are right-skewed — the
	// mean of a distribution with a long tail systematically under-predicts the
	// requests in the tail — so the mean alone is the wrong statistic for a
	// reservation. Must be >= 100.
	SafetyPercent int

	// MaxKeys bounds the number of (tenant, model) pairs tracked.
	//
	// This is a security bound, not a tuning knob: `model` arrives in the
	// request body, so an unbounded map keyed by it is a remote memory-growth
	// primitive. When the table is full, observations for new keys are dropped
	// and counted, and lookups for them miss — which degrades to the
	// conservative Default rather than to a wrong reserve.
	MaxKeys int

	// Now is the clock, defaulting to time.Now. Injected so staleness is tested
	// by advancing a variable rather than by sleeping.
	Now func() time.Time
}

// DefaultPriorConfig returns a configuration tuned for interactive chat
// traffic.
func DefaultPriorConfig() PriorConfig {
	return PriorConfig{
		Smoothing:     8,
		MinSamples:    20,
		MaxAge:        6 * time.Hour,
		Default:       DefaultCompletionReserve,
		Floor:         64,
		Ceiling:       8192,
		SafetyPercent: 150,
		MaxKeys:       4096,
	}
}

// Validate rejects a configuration that would produce a prior that cannot be
// trusted or cannot be bounded. Called by NewPrior and NewPriors; exported so
// config loading fails at startup rather than at the first uncapped request.
func (c PriorConfig) Validate() error {
	switch {
	case c.Smoothing < 1:
		return fmt.Errorf("tokens: PriorConfig.Smoothing must be >= 1, got %d", c.Smoothing)
	case c.MinSamples < 1:
		return fmt.Errorf("tokens: PriorConfig.MinSamples must be >= 1, got %d", c.MinSamples)
	case c.MaxAge < 0:
		return fmt.Errorf("tokens: PriorConfig.MaxAge must be >= 0, got %v", c.MaxAge)
	case c.Default <= 0:
		return fmt.Errorf("tokens: PriorConfig.Default must be > 0, got %d", c.Default)
	case c.Default > MaxCompletionReserve:
		return fmt.Errorf("tokens: PriorConfig.Default (%d) exceeds MaxCompletionReserve (%d)",
			c.Default, MaxCompletionReserve)
	case c.Floor < 0:
		return fmt.Errorf("tokens: PriorConfig.Floor must be >= 0, got %d", c.Floor)
	case c.Ceiling <= 0:
		return fmt.Errorf("tokens: PriorConfig.Ceiling must be > 0, got %d", c.Ceiling)
	case c.Floor > c.Ceiling:
		return fmt.Errorf("tokens: PriorConfig.Floor (%d) must be <= Ceiling (%d)", c.Floor, c.Ceiling)
	case c.Ceiling > MaxCompletionReserve:
		return fmt.Errorf("tokens: PriorConfig.Ceiling (%d) exceeds MaxCompletionReserve (%d)",
			c.Ceiling, MaxCompletionReserve)
	case c.SafetyPercent < 100:
		return fmt.Errorf("tokens: PriorConfig.SafetyPercent must be >= 100, got %d", c.SafetyPercent)
	case c.MaxKeys < 1:
		return fmt.Errorf("tokens: PriorConfig.MaxKeys must be >= 1, got %d", c.MaxKeys)
	}
	return nil
}

func (c *PriorConfig) now() time.Time {
	if c.Now != nil {
		return c.Now()
	}
	return time.Now()
}

// Prior is an online estimator of observed completion length for one
// (tenant, model) pair.
//
// It holds three numbers on purpose:
//
//	ewma       — a smoothed central tendency, so the reserve follows a workload
//	             that changes shape.
//	highWater  — the largest completion ever observed, which the EWMA by
//	             construction cannot represent. Reserving max(scaled mean,
//	             high-water) is what makes the prior safe rather than merely
//	             typical: the tenant that emits one 6000-token answer per
//	             thousand requests still gets 6000 reserved.
//	samples    — exposed, because a prior with 3 observations and a prior with
//	             3000 are different objects and a caller that cannot tell them
//	             apart will trust the wrong one. MinSamples enforces that
//	             distinction here; Snapshot lets an operator see it.
//
// Safe for concurrent use. The critical sections are a handful of integer
// operations under a plain Mutex: an RWMutex would add work to the (frequent)
// read path to speed up a section too short for the reader-parallelism to pay
// for itself.
type Prior struct {
	cfg    PriorConfig
	tenant string
	model  string

	mu sync.Mutex
	// ewmaX1000 is the smoothed mean in thousandths of a token, so that the
	// smoothing recurrence keeps resolution without floating point. A single
	// observation of 1 token still moves it.
	ewmaX1000 int64
	highWater int
	samples   int64
	lastObs   time.Time
}

// NewPrior builds a standalone prior, for callers (and tests) that manage a
// single (tenant, model) pair themselves. Most callers want Priors.
func NewPrior(tenant, model string, cfg PriorConfig) (*Prior, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Prior{cfg: cfg, tenant: tenant, model: model}, nil
}

// A nil *Prior is a valid value meaning "no prior exists". Every method below
// tolerates it, because Priors.Lookup deliberately returns nil rather than
// creating an entry (see Priors) and threading a nil check through every caller
// of a nil-returning lookup is how one of them ends up missing it. Reserve on a
// nil prior reports untrusted, which is the conservative answer.

// Tenant returns the tenant this prior belongs to.
func (p *Prior) Tenant() string {
	if p == nil {
		return ""
	}
	return p.tenant
}

// Model returns the model this prior belongs to, or "" if it is nil or
// unattributed.
func (p *Prior) Model() string {
	if p == nil {
		return ""
	}
	return p.model
}

// Observe folds one real completion length into the prior.
//
// Non-positive observations are ignored rather than recorded as zero: a failed
// or cancelled request produced no evidence about how long a completion is, and
// averaging zeros in would drag the reserve down precisely when a provider is
// unhealthy and requests are dying early. Observations above
// MaxCompletionReserve are clamped, not discarded — the fact that something
// enormous happened is real information, its exact magnitude past the clamp is
// not usable.
func (p *Prior) Observe(completionTokens int) {
	if p == nil || completionTokens <= 0 {
		return
	}
	if completionTokens > MaxCompletionReserve {
		completionTokens = MaxCompletionReserve
	}
	now := p.cfg.now()

	p.mu.Lock()
	if p.samples == 0 {
		// Seed with the first observation instead of easing up from zero.
		// Starting at zero would make the first several reserves meaningless,
		// and MinSamples already prevents them from being trusted.
		p.ewmaX1000 = int64(completionTokens) * 1000
	} else {
		// The step is at least one milli-token whenever the mean is not already
		// at the target. Without that nudge the integer division truncates to
		// zero once the gap is smaller than Smoothing, and the mean parks
		// permanently short of a constant input — at Smoothing=8 it converges to
		// 999.992 for a steady 1000, which then *displays* as 999 because the
		// milli-to-token conversion truncates too. A mean that is permanently
		// one token wrong is not a bug anybody would notice by looking, which is
		// exactly why TestPriorEWMAConvergesExactly asserts it.
		target := int64(completionTokens) * 1000
		delta := (target - p.ewmaX1000) / int64(p.cfg.Smoothing)
		if delta == 0 {
			switch {
			case target > p.ewmaX1000:
				delta = 1
			case target < p.ewmaX1000:
				delta = -1
			}
		}
		p.ewmaX1000 += delta
	}
	if completionTokens > p.highWater {
		p.highWater = completionTokens
	}
	p.samples++
	p.lastObs = now
	p.mu.Unlock()
}

// ObserveUsage folds a provider-reported usage record into the prior.
//
// CompletionTokens is used as-is because every provider that emits reasoning
// tokens already includes them in it (see apiv1.Usage): a reserve that excluded
// hidden reasoning would under-reserve a thinking model by more than an order
// of magnitude, which is the exact mistake apiv1 warns about.
func (p *Prior) ObserveUsage(u *apiv1.Usage) {
	if u == nil {
		return
	}
	p.Observe(u.CompletionTokens)
}

// Reserve returns the completion tokens to reserve and whether the prior was
// trusted to produce it. When trusted is false the returned value is the
// configured default, not a computed one.
func (p *Prior) Reserve() (int, bool) {
	if p == nil {
		return DefaultCompletionReserve, false
	}
	now := p.cfg.now()

	p.mu.Lock()
	samples, ewma, hwm, last := p.samples, p.ewmaX1000, p.highWater, p.lastObs
	p.mu.Unlock()

	if samples < p.cfg.MinSamples {
		return p.cfg.Default, false
	}
	if p.cfg.MaxAge > 0 && now.Sub(last) > p.cfg.MaxAge {
		return p.cfg.Default, false
	}
	// Scale the mean up before comparing with the high-water mark; take the
	// larger. Both are conservative and neither dominates: the scaled mean wins
	// on a workload whose length is drifting upward, the high-water mark wins on
	// one with a rare long tail.
	scaled := (ewma * int64(p.cfg.SafetyPercent)) / (100 * 1000)
	v := int(scaled)
	if hwm > v {
		v = hwm
	}
	if v < p.cfg.Floor {
		v = p.cfg.Floor
	}
	if v > p.cfg.Ceiling {
		v = p.cfg.Ceiling
	}
	return v, true
}

// PriorStat is a point-in-time view of a prior, for /metrics and for operators
// asking why a reserve is what it is.
type PriorStat struct {
	Tenant string
	Model  string
	// Mean is the smoothed observed completion length, in tokens.
	Mean int
	// HighWater is the largest completion observed.
	HighWater int
	// Samples is how many observations the prior has folded in.
	Samples int64
	// LastObserved is when the most recent observation arrived; zero if none.
	LastObserved time.Time
	// Reserve and Trusted are what Reserve() would return right now.
	Reserve int
	Trusted bool
}

// Snapshot returns the prior's current state.
func (p *Prior) Snapshot() PriorStat {
	p.mu.Lock()
	st := PriorStat{
		Tenant:       p.tenant,
		Model:        p.model,
		Mean:         int(p.ewmaX1000 / 1000),
		HighWater:    p.highWater,
		Samples:      p.samples,
		LastObserved: p.lastObs,
	}
	p.mu.Unlock()
	st.Reserve, st.Trusted = p.Reserve()
	return st
}

// PriorKey identifies a prior.
type PriorKey struct {
	Tenant string
	Model  string
}

// ErrPriorTableFull is reported by Priors.Observe when the table has reached
// MaxKeys and the observation is for a key it does not already track.
var ErrPriorTableFull = errors.New("tokens: prior table is full")

// Priors is the bounded, concurrency-safe collection of per-(tenant, model)
// priors.
//
// Lookup and Observe are split rather than being one get-or-create call, and
// that split is a security decision, not an aesthetic one: `model` comes from
// the request body, so a get-or-create on the admission path would let an
// attacker allocate a prior per request with a random model name. Lookup never
// creates. Only Observe — which runs after a request has actually completed
// upstream, i.e. after the client has paid for the privilege — may allocate, and
// even then only up to MaxKeys.
type Priors struct {
	cfg PriorConfig

	mu      sync.RWMutex
	m       map[PriorKey]*Prior
	dropped int64
}

// NewPriors builds a prior table.
func NewPriors(cfg PriorConfig) (*Priors, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Priors{cfg: cfg, m: make(map[PriorKey]*Prior)}, nil
}

// Lookup returns the prior for a (tenant, model) pair, or nil if none exists.
// Never allocates a prior. nil is a valid argument to EstimateCompletion, which
// then uses the conservative default.
func (p *Priors) Lookup(tenant, model string) *Prior {
	p.mu.RLock()
	pr := p.m[PriorKey{Tenant: tenant, Model: model}]
	p.mu.RUnlock()
	return pr
}

// Observe folds a completion length into the prior for a (tenant, model) pair,
// creating it if the table has room. It returns ErrPriorTableFull when the key
// is new and the table is at MaxKeys, so that the caller can surface the
// saturation as a metric instead of it being invisible.
func (p *Priors) Observe(tenant, model string, completionTokens int) error {
	if completionTokens <= 0 {
		return nil
	}
	if pr := p.Lookup(tenant, model); pr != nil {
		pr.Observe(completionTokens)
		return nil
	}
	key := PriorKey{Tenant: tenant, Model: model}

	p.mu.Lock()
	pr, ok := p.m[key]
	if !ok {
		if len(p.m) >= p.cfg.MaxKeys {
			p.dropped++
			p.mu.Unlock()
			return fmt.Errorf("%w (%d keys); dropped observation for tenant %q model %q",
				ErrPriorTableFull, p.cfg.MaxKeys, tenant, model)
		}
		pr = &Prior{cfg: p.cfg, tenant: tenant, model: model}
		p.m[key] = pr
	}
	p.mu.Unlock()

	// Deliberately outside the table lock: the prior has its own mutex, and
	// holding the table's write lock across an update would serialise every
	// tenant's accounting behind one map.
	pr.Observe(completionTokens)
	return nil
}

// ObserveUsage is Observe with a provider usage record.
func (p *Priors) ObserveUsage(tenant, model string, u *apiv1.Usage) error {
	if u == nil {
		return nil
	}
	return p.Observe(tenant, model, u.CompletionTokens)
}

// Len returns the number of tracked (tenant, model) pairs.
func (p *Priors) Len() int {
	p.mu.RLock()
	n := len(p.m)
	p.mu.RUnlock()
	return n
}

// Dropped returns how many observations were discarded because the table was
// full.
func (p *Priors) Dropped() int64 {
	p.mu.RLock()
	n := p.dropped
	p.mu.RUnlock()
	return n
}

// Snapshot returns a stat per tracked prior. Allocates; intended for the
// metrics scrape and the admin surface, not the request path.
func (p *Priors) Snapshot() []PriorStat {
	p.mu.RLock()
	out := make([]PriorStat, 0, len(p.m))
	priors := make([]*Prior, 0, len(p.m))
	for _, pr := range p.m {
		priors = append(priors, pr)
	}
	p.mu.RUnlock()

	// Snapshot each prior outside the table lock, for the same reason Observe
	// updates outside it.
	for _, pr := range priors {
		out = append(out, pr.Snapshot())
	}
	return out
}
