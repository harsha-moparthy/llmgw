// Package mockprov is a programmable, OpenAI-compatible mock LLM provider.
//
// It is the instrument every measurement in this project is taken with, which
// makes its correctness more important than the gateway's: a bug here does not
// fail a test, it produces a wrong number that a reader then trusts. Two
// properties are therefore load-bearing.
//
// # Determinism
//
// A given request produces byte-identical output every time. The reply text is
// generated from a seed derived from the request content, with no global RNG and
// no wall clock in the response path, so a benchmark can be re-run and compared
// and the cost reconciliation has a stable ground truth. Any nondeterminism here
// would show up as flaky reconciliation and unrepeatable latency numbers.
//
// # Independent cost accounting
//
// The mock computes token counts (via tokens.Reference) and cost (via its OWN
// price list, in this package) completely independently of the gateway's ledger
// and pricing table. That independence is the entire point of the
// reconciliation: if both sides shared a cost model, a bug in it would cancel
// out and the reconciliation would pass while every invoice was wrong. The
// mock's request log is one side of the diff; the gateway's ledger is the other.
package mockprov

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/harsha-moparthy/llmgw/internal/apiv1"
	"github.com/harsha-moparthy/llmgw/internal/money"
	"github.com/harsha-moparthy/llmgw/internal/tokens"
)

// ModelConfig describes one mock model's behaviour and price.
type ModelConfig struct {
	TTFBMillis       int `json:"ttfb_ms"`
	InterTokenMillis int `json:"inter_token_ms"`
	// CompletionTokens is the target completion length. The generated text is
	// trimmed or extended to hit this exactly, so the reported count is
	// predictable.
	CompletionTokens int `json:"completion_tokens"`
	// ReasoningTokens, when the model is a reasoning model, are added to the
	// completion count and reported in completion_tokens_details — matching how
	// real reasoning models bill. They are the trap the gateway's pricing tests
	// exist for: they are ALREADY inside completion_tokens.
	ReasoningTokens int  `json:"reasoning_tokens"`
	Reasoning       bool `json:"reasoning"`
	// CachedPromptFraction, in [0,1], is the share of prompt tokens reported as
	// served from the provider's prefix cache.
	CachedPromptFraction float64 `json:"cached_prompt_fraction"`

	PromptPricePerM       string `json:"prompt_price_per_1m"`
	CachedPromptPricePerM string `json:"cached_prompt_price_per_1m"`
	CompletionPricePerM   string `json:"completion_price_per_1m"`

	// Resolved prices, computed at load time from the strings above.
	promptRate money.Pico
	cachedRate money.Pico
	outputRate money.Pico
}

// Faults are the programmable failure modes. Every field defaults to off, and
// each is switchable at runtime through the admin endpoint so a load test can
// break a provider mid-flight — which is exactly the failover demonstration the
// spec asks for.
type Faults struct {
	// Down makes the listener refuse or reset connections. The most basic
	// outage, and the one the failover benchmark toggles.
	Down bool `json:"down"`
	// StatusCode, when non-zero, makes every request return this HTTP status.
	StatusCode int    `json:"status_code"`
	StatusBody string `json:"status_body"`
	// RetryAfterSeconds populates a Retry-After header on the error, exercising
	// the adapter's parsing of it.
	RetryAfterSeconds int `json:"retry_after_s"`
	// AddedLatencyMillis adds constant latency to every response.
	AddedLatencyMillis int `json:"added_latency_ms"`
	// P99SpikeMillis and P99SpikeRate inject a tail: a fraction P99SpikeRate of
	// requests take an extra P99SpikeMillis. This is what lets the benchmark
	// observe a realistic p99 rather than a flat latency.
	P99SpikeMillis int     `json:"p99_spike_ms"`
	P99SpikeRate   float64 `json:"p99_spike_rate"`
	// MidStreamAbortAfter emits this many chunks then closes the connection with
	// no terminating [DONE]. The single most important fault in the project: it
	// is the one that tests whether the gateway can fail over — or at least fail
	// honestly — on a streaming request.
	MidStreamAbortAfter int `json:"mid_stream_abort_after"`
	// MidStreamStallAfter emits this many chunks then hangs, to exercise read
	// deadlines.
	MidStreamStallAfter int `json:"mid_stream_stall_after"`
	// MalformedSSE emits a frame that violates SSE framing.
	MalformedSSE bool `json:"malformed_sse"`
	// MalformedJSON emits a valid SSE frame whose data is not valid JSON.
	MalformedJSON bool `json:"malformed_json"`
	// TruncateAtLength ends the response with finish_reason "length".
	TruncateAtLength bool `json:"truncate_at_length"`
	// ErrorRate, in [0,1], makes a random-by-seed fraction of requests fail with
	// StatusCode (or 503 if unset).
	ErrorRate float64 `json:"error_rate"`
}

// Config is the mock provider's configuration.
type Config struct {
	Listen       string                  `json:"listen"`
	AdminListen  string                  `json:"admin_listen"`
	LogPath      string                  `json:"log_path"`
	DefaultModel string                  `json:"default_model"`
	Models       map[string]*ModelConfig `json:"models"`
	Faults       Faults                  `json:"faults"`
}

// resolve computes per-token rates from the string prices, rejecting inexact
// ones exactly as the gateway's pricing package does — so the two independent
// price lists cannot disagree merely because one rounded and the other did not.
func (m *ModelConfig) resolve(name string) error {
	parseRate := func(field, s string) (money.Pico, error) {
		if s == "" {
			return 0, nil
		}
		usd, err := money.ParseUSD(s)
		if err != nil {
			return 0, fmt.Errorf("model %q %s %q: %w", name, field, s, err)
		}
		rate, err := money.PerToken(usd)
		if err != nil {
			return 0, fmt.Errorf("model %q %s: %w", name, field, err)
		}
		return rate, nil
	}
	var err error
	if m.promptRate, err = parseRate("prompt_price_per_1m", m.PromptPricePerM); err != nil {
		return err
	}
	if m.CachedPromptPricePerM != "" {
		if m.cachedRate, err = parseRate("cached_prompt_price_per_1m", m.CachedPromptPricePerM); err != nil {
			return err
		}
	} else {
		// No cached rate given defaults to the full input rate — no discount —
		// rather than to zero, which would make cached prompts free and
		// under-bill every cache hit.
		m.cachedRate = m.promptRate
	}
	if m.outputRate, err = parseRate("completion_price_per_1m", m.CompletionPricePerM); err != nil {
		return err
	}
	if m.CachedPromptFraction < 0 || m.CachedPromptFraction > 1 {
		return fmt.Errorf("model %q cached_prompt_fraction %g is not in [0,1]", name, m.CachedPromptFraction)
	}
	if m.CompletionTokens <= 0 {
		m.CompletionTokens = 32
	}
	return nil
}

// Validate resolves prices and applies defaults. Called once at load time.
func (c *Config) Validate() error {
	if len(c.Models) == 0 {
		return fmt.Errorf("mockprov: no models configured")
	}
	if c.DefaultModel == "" {
		return fmt.Errorf("mockprov: default_model is required")
	}
	if _, ok := c.Models[c.DefaultModel]; !ok {
		return fmt.Errorf("mockprov: default_model %q is not in the models map", c.DefaultModel)
	}
	names := make([]string, 0, len(c.Models))
	for name := range c.Models {
		names = append(names, name)
	}
	sort.Strings(names) // deterministic error order
	for _, name := range names {
		if err := c.Models[name].resolve(name); err != nil {
			return err
		}
	}
	return nil
}

// RequestRecord is one row of the mock's independent request log — the provider
// side of the cost reconciliation.
//
// Its JSON shape is deliberately EXACTLY ledger.ProviderRecord: request_id,
// attempt, model, a nested tokens object, and cost_pico. That is what lets the
// reconciliation harness decode this log directly with ledger.DecodeProviderLog
// and compare it to the gateway's ledger with no shape translation in between —
// a translation step would be one more place the two representations could
// silently diverge. The extra fields (stream, fault, finish_reason, timing) are
// diagnostics the reconciler ignores.
type RequestRecord struct {
	RequestID      string      `json:"request_id"`
	Attempt        int         `json:"attempt"`
	Model          string      `json:"model"`
	Tokens         TokenCounts `json:"tokens"`
	CostPico       int64       `json:"cost_pico"`
	Stream         bool        `json:"stream"`
	Fault          string      `json:"fault,omitempty"`
	FinishReason   string      `json:"finish_reason"`
	ReceivedAt     string      `json:"received_at"`
	DurationMicros int64       `json:"duration_micros"`
}

// TokenCounts mirrors ledger.TokenCounts so the mock's log is byte-compatible
// with the provider-log schema the reconciler reads. Duplicated rather than
// imported because the mock is meant to be an INDEPENDENT accounting of the same
// request — sharing the type would be a small step toward sharing the code, and
// the whole value of the reconciliation is that the two sides are computed
// separately.
type TokenCounts struct {
	Prompt     int `json:"prompt"`
	Cached     int `json:"cached"`
	Completion int `json:"completion"`
	Reasoning  int `json:"reasoning"`
}

// usage builds the OpenAI usage record for a generated response.
//
// The subtlety encoded here is the two "already included" relationships that
// every real provider has and that the gateway's pricing tests check:
//   - completion_tokens INCLUDES reasoning_tokens
//   - prompt_tokens INCLUDES cached_tokens
//
// Getting these wrong in the instrument would make the gateway look correct
// while both were wrong together, so the mock is the authority on them.
func (m *ModelConfig) usage(promptTokens, completionTextTokens int) *apiv1.Usage {
	reasoning := 0
	if m.Reasoning {
		reasoning = m.ReasoningTokens
	}
	cached := int(float64(promptTokens) * m.CachedPromptFraction)
	if cached > promptTokens {
		cached = promptTokens
	}
	completion := completionTextTokens + reasoning
	u := &apiv1.Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completion,
		TotalTokens:      promptTokens + completion,
	}
	if cached > 0 {
		u.PromptTokensDetails = &apiv1.PromptTokensDetails{CachedTokens: cached}
	}
	if reasoning > 0 {
		u.CompletionTokensDetails = &apiv1.CompletionTokensDetails{ReasoningTokens: reasoning}
	}
	return u
}

// cost computes the charge for a usage record from the mock's own rates.
//
// This mirrors the gateway's pricing math deliberately, and independently: full-
// rate input is (prompt - cached), cached tokens bill at the cached rate, and
// reasoning tokens are NOT added on top of completion (they are already in it).
// The reconciliation checks that this independent computation matches the
// gateway's to the picodollar.
func (m *ModelConfig) cost(u *apiv1.Usage) money.Pico {
	cached := u.CachedPromptTokens()
	fullInput := int64(u.PromptTokens - cached)
	if fullInput < 0 {
		fullInput = 0
	}
	inputCost, _ := money.Mul(m.promptRate, fullInput)
	cachedCost, _ := money.Mul(m.cachedRate, int64(cached))
	// Completion tokens already include reasoning; charge them once at the
	// output rate.
	outputCost, _ := money.Mul(m.outputRate, int64(u.CompletionTokens))
	total, _ := money.Add(inputCost, cachedCost)
	total, _ = money.Add(total, outputCost)
	return total
}

// generateReply produces deterministic reply text of at least n reference
// tokens, seeded from the request so the same request yields identical bytes.
//
// The text is drawn from a fixed lexicon indexed by a seeded counter. It is not
// meant to be coherent — the mock is a load and correctness instrument, not a
// model — only reproducible and tokenisable to a known count.
func generateReply(seed uint64, targetTokens int) string {
	if targetTokens <= 0 {
		targetTokens = 1
	}
	var sb strings.Builder
	x := seed | 1 // avoid a zero state
	for tokenCount(sb.String()) < targetTokens {
		// xorshift64: deterministic, no allocation, no global state.
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		w := lexicon[x%uint64(len(lexicon))]
		if sb.Len() > 0 {
			sb.WriteByte(' ')
		}
		sb.WriteString(w)
	}
	// Trim to exactly targetTokens reference tokens so the reported count is
	// exact rather than "at least".
	toks := tokens.Reference.Tokenize(sb.String())
	if len(toks) > targetTokens {
		toks = toks[:targetTokens]
	}
	return tokens.Reference.Detokenize(toks)
}

func tokenCount(s string) int { return tokens.Reference.Count(s) }

// lexicon is a small fixed word list. Fixed, not random, so the reply is a pure
// function of the seed.
var lexicon = []string{
	"the", "gateway", "routes", "requests", "across", "providers", "with",
	"failover", "caching", "and", "budgets", "under", "load", "streaming",
	"tokens", "through", "a", "single", "compatible", "interface", "that",
	"normalises", "responses", "counts", "usage", "enforces", "limits",
	"reconciles", "cost", "against", "logs", "cleanly", "handling", "outages",
	"within", "bounded", "windows", "while", "preserving", "backpressure",
}

// requestSeed derives a stable seed from the request's semantic content, so two
// identical requests generate the same reply and two different ones almost
// certainly do not.
func requestSeed(req *apiv1.ChatRequest, model string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(model))
	for i := range req.Messages {
		_, _ = h.Write([]byte(req.Messages[i].Role))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(req.Messages[i].Content.Text()))
		_, _ = h.Write([]byte{0})
	}
	if req.Seed != nil {
		var b [8]byte
		for i := 0; i < 8; i++ {
			b[i] = byte(*req.Seed >> (8 * i))
		}
		_, _ = h.Write(b[:])
	}
	return h.Sum64()
}

// faultSnapshot is an atomically-swappable copy of the fault config, so the
// admin endpoint can change faults without locking the request path.
type faultSnapshot struct {
	f Faults
}

// FaultStore holds the current faults behind an RWMutex. Reads are on the hot
// path and vastly outnumber writes (which come only from the admin endpoint), so
// an RWMutex is the right trade rather than a plain Mutex.
type FaultStore struct {
	mu sync.RWMutex
	f  Faults
}

// NewFaultStore returns a store seeded with the given faults.
func NewFaultStore(f Faults) *FaultStore { return &FaultStore{f: f} }

// Load returns the current faults.
func (s *FaultStore) Load() Faults {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.f
}

// Set replaces one fault field by name, returning an error for an unknown field.
// Used by the admin endpoint.
func (s *FaultStore) Set(field, value string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return applyFault(&s.f, field, value)
}

// Replace swaps the whole fault set.
func (s *FaultStore) Replace(f Faults) {
	s.mu.Lock()
	s.f = f
	s.mu.Unlock()
}

// deterministicJitter maps a seed to a value in [0,1) without touching a global
// RNG, so that error-rate and spike decisions are reproducible per request.
func deterministicJitter(seed uint64) float64 {
	// Take the top 53 bits for a uniform double, the standard trick.
	return float64(seed>>11) / float64(1<<53)
}

// marshalChunk renders a streaming chunk with the given fields.
func marshalChunk(id, model string, created int64, role, content string, finish *string, usage *apiv1.Usage) ([]byte, error) {
	delta := &apiv1.Message{}
	if role != "" {
		delta.Role = role
	}
	if content != "" || (role == "" && finish == nil) {
		delta.Content = apiv1.NewTextContent(content)
	}
	chunk := apiv1.ChatChunk{
		ID:      id,
		Object:  apiv1.ObjectChatCompletionChunk,
		Created: created,
		Model:   model,
		Choices: []apiv1.Choice{{Index: 0, Delta: delta, FinishReason: finish}},
		Usage:   usage,
	}
	return json.Marshal(chunk)
}

// nowFunc is the clock, injectable for tests. The response PATH never reads it
// (determinism), but the log's timestamp and duration do.
type clock struct {
	now func() time.Time
}

func (c clock) Now() time.Time {
	if c.now != nil {
		return c.now()
	}
	return time.Now()
}
